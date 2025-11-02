# chat/ai_service.py
import google.generativeai as genai
import os
import json
from django.conf import settings
from .models import ChatMessage, Chat
from bots.models import Bot
import re
import time


# Configura a API key
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada no ficheiro .env")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")


def generate_suggestions_for_bot(prompt: str):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        instruction = f"""
        Based on the following bot's instructions, generate exactly three short, engaging, and distinct conversation starters (under 10 words each).
        The user will see these as suggestion chips to start the conversation.
        Return the result as a valid JSON array of strings. For example: ["Suggestion 1", "Suggestion 2", "Suggestion 3"].


        Bot Instructions: "{prompt}"
        """
        
        response = model.generate_content(instruction)
        
        # Limpa a resposta antes de fazer o parse do JSON
        cleaned_response = re.sub(r'^``````$', '', response.text.strip(), flags=re.MULTILINE)
        suggestions = json.loads(cleaned_response)
        
        if isinstance(suggestions, list) and len(suggestions) > 0 and all(isinstance(s, str) for s in suggestions):
            # Retorna apenas as 3 primeiras, caso a IA envie mais
            return suggestions[:3]
        
    except Exception as e:
        print(f"Could not generate suggestions: {e}")
    
    return ["Tell me more.", "What can you do?", "Give me an example."]


def get_ai_response(chat_id: int, user_message_text: str, user_message_obj: ChatMessage = None):
    """
    Obtém uma resposta ESTRUTURADA (resposta + sugestões) do modelo Gemini.
    AGORA SUPORTA IMAGENS e ARQUIVOS via upload_file.
    Retorna um dicionário: {'content': '...', 'suggestions': [...]}
    """
    try:
        chat = Chat.objects.select_related('bot').get(id=chat_id)
        bot = chat.bot

        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=10000,
            response_mime_type="application/json"
        )

        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        model = genai.GenerativeModel(
            model_name='gemini-2.5-pro',
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=bot.prompt
        )

        # --- Lógica do Histórico ---
        history_qs = ChatMessage.objects.filter(chat_id=chat_id).order_by('created_at')
        if user_message_obj:
            history_qs = history_qs.exclude(id=user_message_obj.id)

        history = history_qs.order_by('-created_at')[:10].values('role', 'content', 'original_filename', 'attachment_type')
        history = reversed(history) 
        
        gemini_history = []
        for msg in history:
            if "An unexpected error occurred" in msg.get('content', ''):
                continue
            role = 'user' if msg.get('role') == 'user' else 'model'
            
            parts = []
            if msg.get('content'):
                parts.append({'text': msg.get('content')})
            
            if msg.get('attachment_type') == 'image' and msg.get('original_filename'):
                parts.append({'text': f"[Anexo de imagem anterior: {msg.get('original_filename')}]"})
            elif msg.get('attachment_type') == 'file' and msg.get('original_filename'):
                parts.append({'text': f"[Anexo de arquivo anterior: {msg.get('original_filename')}]"})
            
            if parts:
                gemini_history.append({'role': role, 'parts': parts})
        # --- Fim da Lógica do Histórico ---

        chat_session = model.start_chat(history=gemini_history)

        # --- CONSTRÓI O PROMPT MULTIMODAL (UPLOAD UNIFICADO) ---
        input_parts_for_ai = []
        
        # Upload unificado para IMAGENS e ARQUIVOS
        if user_message_obj and user_message_obj.attachment:
            file_path = user_message_obj.attachment.path
            
            try:
                print(f"[AI Service] Fazendo upload: {user_message_obj.original_filename}...")
                uploaded_file = genai.upload_file(path=file_path)
                
                # Aguarda processamento (importante para vídeos/PDFs grandes)
                max_wait_time = 60  # Timeout de 60 segundos
                wait_time = 0
                
                while uploaded_file.state.name == "PROCESSING":
                    if wait_time >= max_wait_time:
                        print(f"[AI Service] Timeout ao processar arquivo após {max_wait_time}s")
                        raise TimeoutError(f"File processing timeout after {max_wait_time}s")
                    
                    print(f"[AI Service] Aguardando processamento... ({wait_time}s)")
                    time.sleep(2)
                    wait_time += 2
                    uploaded_file = genai.get_file(uploaded_file.name)
                
                if uploaded_file.state.name == "ACTIVE":
                    input_parts_for_ai.append(uploaded_file)
                    print(f"[AI Service] Upload concluído: {uploaded_file.state.name}")
                else:
                    raise ValueError(f"File in unexpected state: {uploaded_file.state.name}")
                    
            except Exception as e:
                print(f"[AI Service] Erro no upload: {e}")
                attachment_type_name = "imagem" if user_message_obj.attachment_type == "image" else "arquivo"
                input_parts_for_ai.append(f"[Erro ao processar {attachment_type_name}: {user_message_obj.original_filename}]")

        # Adiciona o texto do usuário
        input_parts_for_ai.append(user_message_text) 

        # Adiciona as instruções de formatação JSON
        json_instruction_prompt = """
        Based on the user's message (and any attached image/file) and the conversation history, provide a helpful response.
        If a file or image was provided, base your response on its content (e.g., summarize the PDF, describe the image, answer questions about the text file).
        Also, generate exactly two distinct, short, and relevant follow-up suggestions (under 10 words each) that the user might ask next.

        Respond with a valid JSON object with the following structure:
        {
          "response": "Your main answer to the user's message goes here.",
          "suggestions": ["First suggestion", "Second suggestion"]
        }
        """
        input_parts_for_ai.append(json_instruction_prompt)
        
        # Envia tudo junto
        print("[AI Service] Enviando request multimodal para o Gemini...")
        response = chat_session.send_message(input_parts_for_ai)
        # --- FIM DAS MUDANÇAS NO PROMPT ---

        if response.parts:
            try:
                cleaned_text = re.sub(r'^``````$', '', response.text.strip(), flags=re.MULTILINE)
                response_data = json.loads(cleaned_text)
                return {
                    'content': response_data.get('response', "Sorry, I couldn't generate a response."),
                    'suggestions': response_data.get('suggestions', [])
                }
            except json.JSONDecodeError as json_err:
                print(f"Gemini JSON parse error: {json_err}. Raw text: {response.text}")
                return {
                    'content': response.text,
                    'suggestions': []
                }
        else:
            reason = "UNKNOWN"
            if response.candidates and response.candidates[0].finish_reason:
                reason = response.candidates[0].finish_reason.name
            print(f"Gemini response was blocked. Finish Reason: {reason}")
            return {
                'content': "My response was blocked. Please try a different message.",
                'suggestions': []
            }

    except Exception as e:
        print(f"An error occurred in Gemini AI service: {e}")
        return {
            'content': "An unexpected error occurred while generating a response. Please try again.",
            'suggestions': []
        }
