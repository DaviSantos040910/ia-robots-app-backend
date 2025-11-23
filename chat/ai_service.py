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
from django.core.files.uploadedfile import UploadedFile
from django.core.files import File  # Importante para salvar arquivos gerados no Django
import tempfile
from pathlib import Path
from django.utils import timezone 
from datetime import timedelta 
from django.db import transaction 
import uuid

# Configuração do cliente
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


def generate_suggestions_for_bot(prompt: str):
    """
    Gera 3 sugestões de conversa para um bot baseado em seu prompt
    """
    try:
        client = get_ai_client()
        instruction = f"""
Based on the following bot's instructions, generate exactly three short, engaging, and distinct conversation starters (under 10 words each).
The user will see these as suggestion chips to start the conversation.
Return the result as a valid JSON array of strings. For example: ["Suggestion 1", "Suggestion 2", "Suggestion 3"].

Bot Instructions: "{prompt}"
"""
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=instruction,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
        # Limpa a resposta de forma mais robusta
        cleaned_response = re.sub(r'^```(json)?\n', '', response.text.strip(), flags=re.MULTILINE)
        cleaned_response = re.sub(r'\n```$', '', cleaned_response.strip(), flags=re.MULTILINE)
        
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
    Suporta múltiplos anexos e histórico de conversa.
    
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
        
        # 1. Encontra a última mensagem do assistente para definir o contexto
        last_assistant_message = ChatMessage.objects.filter(
            chat_id=chat_id, 
            role=ChatMessage.Role.ASSISTANT
        ).order_by('-created_at').first()
        
        if last_assistant_message:
            since_timestamp = last_assistant_message.created_at
        else:
            since_timestamp = chat.created_at - timedelta(seconds=1) 

        # 2. Busca o HISTÓRICO (mensagens ANTES do ponto de corte)
        history_qs = ChatMessage.objects.filter(
            chat_id=chat_id, 
            created_at__lt=since_timestamp
        ).order_by('-created_at')[:10]
        
        history_values = reversed(history_qs.values('role', 'content'))
        
        gemini_history = []
        for msg in history_values:
            if not msg.get('content') or "An unexpected error" in msg.get('content'):
                continue
            
            role = 'user' if msg.get('role') == 'user' else 'model'
            parts = []
            if msg.get('content'):
                parts.append({"text": msg.get('content')})
            
            if parts:
                gemini_history.append({
                    "role": role,
                    "parts": parts
                })

        # 3. Busca o PROMPT ATUAL (mensagens DEPOIS do ponto de corte)
        prompt_messages_qs = ChatMessage.objects.filter(
            chat_id=chat_id,
            role=ChatMessage.Role.USER,
            created_at__gte=since_timestamp
        ).order_by('created_at')

        input_parts_for_ai = []
        
        for user_msg in prompt_messages_qs:
            # Processa ANEXOS
            if user_msg.attachment and user_msg.attachment.path:
                try:
                    file_path = user_msg.attachment.path
                    original_filename = user_msg.original_filename or "file"
                    
                    mime_type, _ = mimetypes.guess_type(original_filename)
                    
                    if not mime_type and user_msg.attachment_type == 'image':
                        mime_type = 'image/jpeg'
                    
                    # Se for áudio, não enviamos como imagem/arquivo para análise visual, 
                    # a menos que a IA suporte áudio nativo no contexto (Gemini 1.5 Pro suporta).
                    # Aqui assumimos que a transcrição já virou texto no content, 
                    # então pulamos o binário de áudio para economizar tokens, 
                    # a menos que seja imagem/pdf.
                    if mime_type and (mime_type.startswith('image/') or mime_type == 'application/pdf'):
                         with open(file_path, 'rb') as f:
                            file_data = f.read()
                            input_parts_for_ai.append(
                                types.Part.from_bytes(data=file_data, mime_type=mime_type)
                            )
                except Exception as e:
                    print(f"[AI Service] Erro ao processar arquivo: {e}")
            
            if user_msg.content:
                input_parts_for_ai.append({"text": user_msg.content})

        # Instrução final para formato JSON
        json_instruction_prompt = """
Based on the user's message and context, provide a helpful response.
Respond with a valid JSON object:
{
  "response": "Your main answer goes here.",
  "suggestions": ["Suggestion 1", "Suggestion 2"]
}
"""
        input_parts_for_ai.append({"text": json_instruction_prompt})
        
        contents = gemini_history + [{
            "role": "user",
            "parts": input_parts_for_ai
        }]
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=generation_config
        )
        
        if response.text:
            try:
                cleaned_text = response.text.strip()
                cleaned_text = re.sub(r'^```(json)?\s*', '', cleaned_text, flags=re.MULTILINE)
                cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE)
                cleaned_text = cleaned_text.strip()
                
                response_data = json.loads(cleaned_text)
                return {
                    'content': response_data.get('response', "Sorry, I couldn't generate a response."),
                    'suggestions': response_data.get('suggestions', [])
                }
            except json.JSONDecodeError:
                # Fallback se não vier JSON válido
                cleaned_text = re.sub(r'^```(json)?\n', '', response.text.strip(), flags=re.MULTILINE)
                cleaned_text = re.sub(r'\n```$', '', cleaned_text.strip(), flags=re.MULTILINE)
                return {'content': cleaned_text, 'suggestions': []}
        else:
            return {'content': "My response was blocked.", 'suggestions': []}
    
    except Exception as e:
        print(f"An error occurred in Gemini AI service: {e}")
        return {'content': "An unexpected error occurred while generating a response.", 'suggestions': []}


def transcribe_audio_gemini(audio_file: UploadedFile) -> dict:
    """
    Transcreve áudio usando Google Gemini API.
    Retorna apenas o texto.
    """
    try:
        client = get_ai_client()
        audio_bytes = audio_file.read()
        
        mime_type, _ = mimetypes.guess_type(audio_file.name)
        if not mime_type:
            # Fallback básico
            mime_type = 'audio/m4a'
        
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        prompt = "Generate a transcript of the speech in Portuguese. Return only the transcribed text, strictly without timestamps or speaker labels."
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, audio_part],
            config=types.GenerateContentConfig(temperature=0.2)
        )
        
        if response.text:
            return {'success': True, 'transcription': response.text.strip()}
        else:
            return {'success': False, 'error': 'No transcription returned'}
    
    except Exception as e:
        print(f"[Transcription Error] {e}")
        return {'success': False, 'error': str(e)}


def generate_tts_audio(message_text: str, output_path: str) -> dict:
    """
    Gera áudio TTS usando Gemini 2.5 Flash TTS e salva no output_path.
    """
    try:
        import wave
        client = get_ai_client()
        
        # Limitar tamanho do texto para evitar erros ou custos excessivos
        safe_text = message_text[:2000] 

        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-tts',
            contents=safe_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                    )
                )
            )
        )
        
        if not response.candidates or not response.candidates[0].content.parts:
            raise Exception("Nenhum áudio gerado pela API.")
        
        audio_part = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                audio_part = part.inline_data
                break
        
        if not audio_part:
            raise Exception("Nenhum dado de áudio encontrado (inline_data vazio).")
        
        pcm_data = audio_part.data
        sample_rate = 24000
        channels = 1
        sample_width = 2 # 16-bit
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        
        return {'success': True, 'file_path': output_path}
    
    except Exception as e:
        print(f"[TTS Error] {e}")
        return {'success': False, 'error': str(e)}


def handle_voice_interaction(chat_id: int, audio_file: UploadedFile, user) -> dict:
    """
    Orquestra o fluxo de chamada AO VIVO (Voice Call):
    1. Transcreve.
    2. Salva user message.
    3. Responde (texto).
    4. Salva AI message.
    Retorna dicionário para a UI tocar o texto via TTS nativo do app ou stream.
    """
    # Reutiliza a lógica base, sem gerar arquivo de áudio de resposta persistente no banco (para economizar storage em chamadas longas)
    # Se quiser salvar o áudio da chamada também, basta mudar reply_with_audio=True
    result = handle_voice_message(chat_id, audio_file, reply_with_audio=False, user=user)
    
    # Formata para o padrão que o frontend da VoiceCall espera
    return {
        "transcription": result['user_message'].content,
        "ai_response_text": result['ai_message'].content,
        "user_message": result['user_message'],
        "ai_messages": [result['ai_message']]
    }


def handle_voice_message(chat_id: int, user_audio_file: UploadedFile, reply_with_audio: bool, user) -> dict:
    """
    Orquestra o fluxo de MENSAGEM DE VOZ (estilo WhatsApp):
    1. Transcreve o áudio do usuário.
    2. Salva a mensagem do usuário no banco (com o arquivo de áudio anexado).
    3. Gera a resposta da IA em texto.
    4. (Opcional) Gera o áudio da resposta da IA e anexa à mensagem da IA.
    5. Salva a mensagem da IA.
    
    Retorna: {'user_message': ChatMessage, 'ai_message': ChatMessage}
    """
    
    with transaction.atomic():
        chat = Chat.objects.get(id=chat_id)

        # --- 1. Transcrição ---
        # Garante ponteiro no início para leitura
        user_audio_file.seek(0) 
        
        trans_result = transcribe_audio_gemini(user_audio_file)
        
        if trans_result['success']:
            transcription = trans_result['transcription']
        else:
            # Se falhar, salvamos um texto de fallback para não perder o áudio enviado
            transcription = "[Áudio recebido - Transcrição indisponível]"
            print(f"Falha na transcrição: {trans_result.get('error')}")

        # --- 2. Salvar Mensagem do Usuário ---
        # Resetamos o ponteiro novamente antes de salvar no modelo
        user_audio_file.seek(0)
        
        user_message = ChatMessage.objects.create(
            chat=chat,
            role=ChatMessage.Role.USER,
            content=transcription,
            attachment=user_audio_file,
            attachment_type='audio',
            original_filename=user_audio_file.name or "voice_message.m4a"
        )
        
        chat.last_message_at = timezone.now()
        chat.save()

        # --- 3. Gerar Resposta da IA (Texto) ---
        # Usamos o texto transcrito como prompt
        ai_response_data = get_ai_response(chat_id, transcription, user_message_obj=user_message)
        ai_text = ai_response_data.get('content', '')
        ai_suggestions = ai_response_data.get('suggestions', [])

        # Instancia a mensagem (ainda não salva) para poder anexar arquivo se necessário
        ai_message = ChatMessage(
            chat=chat,
            role=ChatMessage.Role.ASSISTANT,
            content=ai_text,
            suggestion1=ai_suggestions[0] if len(ai_suggestions) > 0 else None,
            suggestion2=ai_suggestions[1] if len(ai_suggestions) > 1 else None,
        )

        # --- 4. Gerar Áudio da Resposta (TTS) ---
        if reply_with_audio and ai_text and len(ai_text.strip()) > 0:
            try:
                # Cria arquivo temporário seguro
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    temp_path = temp_audio_file.name
                
                # Gera o áudio no arquivo temporário
                tts_result = generate_tts_audio(ai_text, temp_path)
                
                if tts_result['success']:
                    # Abre o arquivo temporário e salva no campo FileField do Django
                    with open(temp_path, 'rb') as f:
                        # Gera nome único para o arquivo no storage
                        filename = f"reply_tts_{uuid.uuid4().hex[:10]}.wav"
                        
                        # Salva o conteúdo no modelo. save=False pois salvaremos o objeto inteiro depois.
                        ai_message.attachment.save(filename, File(f), save=False)
                        ai_message.attachment_type = 'audio'
                        ai_message.original_filename = "voice_reply.wav"
                
                # Remove o arquivo temporário do sistema de arquivos local
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Erro ao gerar/salvar áudio de resposta: {e}")
                # Segue sem áudio se der erro

        # --- 5. Salvar Mensagem da IA ---
        ai_message.save()
        
        chat.last_message_at = timezone.now()
        chat.save()

        return {
            "user_message": user_message,
            "ai_message": ai_message
        }