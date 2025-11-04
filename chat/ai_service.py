# chat/ai_service.py

from google import genai
from google.genai import types
import os
import json
import mimetypes
from django.conf import settings
from .models import ChatMessage, Chat
from bots.models import Bot
import re
import time


# Configuração do cliente
# Para Google AI Studio (desenvolvimento)
def get_ai_client():
    """
    Retorna o cliente configurado.
    Para migrar para Vertex AI, basta alterar esta função.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada")
    
    # Cliente para Google AI Studio
    client = genai.Client(api_key=api_key)
    return client


# Para migração futura para Vertex AI, substitua a função acima por:
# def get_ai_client():
#     """Cliente para Vertex AI (produção)"""
#     project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
#     location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
#     
#     client = genai.Client(
#         vertexai=True,
#         project=project_id,
#         location=location
#     )
#     return client



def generate_suggestions_for_bot(prompt: str):
    try:
        client = get_ai_client()
        
        instruction = f"""
Based on the following bot's instructions, generate exactly three short, engaging, and distinct conversation starters (under 10 words each).
The user will see these as suggestion chips to start the conversation.
Return the result as a valid JSON array of strings. For example: ["Suggestion 1", "Suggestion 2", "Suggestion 3"].

Bot Instructions: "{prompt}"
"""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=instruction,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
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
    AGORA SUPORTA IMAGENS e ARQUIVOS enviados inline como bytes.
    Retorna um dicionário: {'content': '...', 'suggestions': [...]}
    """
    try:
        client = get_ai_client()
        chat = Chat.objects.select_related('bot').get(id=chat_id)
        bot = chat.bot
        
        # Configurações de geração
        generation_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=10000,
            response_mime_type="application/json",
            system_instruction=bot.prompt,
            safety_settings=[
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE'
                ),
            ]
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
                parts.append({"text": msg.get('content')})
            
            if msg.get('attachment_type') == 'image' and msg.get('original_filename'):
                parts.append({"text": f"[Anexo de imagem anterior: {msg.get('original_filename')}]"})
            elif msg.get('attachment_type') == 'file' and msg.get('original_filename'):
                parts.append({"text": f"[Anexo de arquivo anterior: {msg.get('original_filename')}]"})
            
            if parts:
                gemini_history.append({
                    "role": role,
                    "parts": parts
                })
        
        # --- Fim da Lógica do Histórico ---
        
        # --- CONSTRÓI O PROMPT MULTIMODAL (INLINE BYTES) ---
        input_parts_for_ai = []
        
        # ✅ SOLUÇÃO: Envia imagem/arquivo como bytes inline
        if user_message_obj and user_message_obj.attachment:
            file_path = user_message_obj.attachment.path
            original_filename = user_message_obj.original_filename
            
            try:
                print(f"[AI Service] Processando: {original_filename}...")
                
                # Detecta o MIME type
                mime_type, _ = mimetypes.guess_type(original_filename)
                
                if not mime_type and user_message_obj.attachment_type == 'image':
                    _, ext = os.path.splitext(original_filename)
                    ext_lower = ext.lower()
                    mime_map = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp',
                        '.bmp': 'image/bmp'
                    }
                    mime_type = mime_map.get(ext_lower, 'image/jpeg')
                
                print(f"[AI Service] MIME type: {mime_type}")
                
                # Lê o arquivo como bytes
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                # Cria Part inline com bytes
                input_parts_for_ai.append(
                    types.Part.from_bytes(
                        data=file_data,
                        mime_type=mime_type
                    )
                )
                print(f"[AI Service] Arquivo adicionado inline ({len(file_data)} bytes)")
                    
            except Exception as e:
                print(f"[AI Service] Erro ao processar arquivo: {e}")
                attachment_type_name = "imagem" if user_message_obj.attachment_type == "image" else "arquivo"
                input_parts_for_ai.append({"text": f"[Erro ao processar {attachment_type_name}: {original_filename}]"})
        
        # Adiciona o texto do usuário
        input_parts_for_ai.append({"text": user_message_text})
        
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
        input_parts_for_ai.append({"text": json_instruction_prompt})
        
        # Prepara o conteúdo final com estrutura correta
        contents = gemini_history + [{
            "role": "user",
            "parts": input_parts_for_ai
        }]
        
        # Envia tudo junto
        print("[AI Service] Enviando request multimodal para o Gemini...")
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=generation_config
        )
        
        # --- PROCESSA A RESPOSTA ---
        if response.text:
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
            if response.candidates and len(response.candidates) > 0:
                reason = response.candidates[0].finish_reason
            
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
