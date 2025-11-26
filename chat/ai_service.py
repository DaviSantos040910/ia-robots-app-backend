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
from django.core.files import File
import tempfile
from pathlib import Path
from django.utils import timezone 
from datetime import timedelta 
from django.db import transaction 
import uuid
import wave # Necessário para ler metadados de WAV

def get_ai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada")
    client = genai.Client(api_key=api_key)
    return client

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
            model='gemini-2.5-flash',
            contents=instruction,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        cleaned_response = re.sub(r'^```(json)?\n', '', response.text.strip(), flags=re.MULTILINE)
        cleaned_response = re.sub(r'\n```$', '', cleaned_response.strip(), flags=re.MULTILINE)
        suggestions = json.loads(cleaned_response)
        if isinstance(suggestions, list) and len(suggestions) > 0 and all(isinstance(s, str) for s in suggestions):
            return suggestions[:3]
    except Exception as e:
        print(f"Could not generate suggestions: {e}")
    return ["Tell me more.", "What can you do?", "Give me an example."]

def get_ai_response(chat_id: int, user_message_text: str, user_message_obj: ChatMessage = None, reply_with_audio: bool = False):
    """
    Obtém resposta da IA. 
    Se reply_with_audio=True, gera o áudio TTS imediatamente.
    Retorna dict com 'audio_path' e 'duration_ms' se gerado.
    """
    try:
        client = get_ai_client()
        chat = Chat.objects.select_related('bot').get(id=chat_id)
        bot = chat.bot
        
        # Configuração do Modelo
        generation_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=10000,
            response_mime_type="application/json",
            system_instruction=bot.prompt,
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
            ]
        )
        
        # Histórico de mensagens
        last_assistant_message = ChatMessage.objects.filter(
            chat_id=chat_id, role=ChatMessage.Role.ASSISTANT
        ).order_by('-created_at').first()
        
        since_timestamp = last_assistant_message.created_at if last_assistant_message else chat.created_at - timedelta(seconds=1) 

        history_qs = ChatMessage.objects.filter(chat_id=chat_id, created_at__lt=since_timestamp).order_by('-created_at')[:10]
        history_values = reversed(history_qs.values('role', 'content'))
        
        gemini_history = []
        for msg in history_values:
            if not msg.get('content') or "An unexpected error" in msg.get('content'): continue
            role = 'user' if msg.get('role') == 'user' else 'model'
            if msg.get('content'):
                gemini_history.append({"role": role, "parts": [{"text": msg.get('content')}]})

        # Contexto Atual
        prompt_messages_qs = ChatMessage.objects.filter(chat_id=chat_id, role=ChatMessage.Role.USER, created_at__gte=since_timestamp).order_by('created_at')
        input_parts_for_ai = []
        
        for user_msg in prompt_messages_qs:
            # Tratamento de anexos
            if user_msg.attachment and user_msg.attachment.path:
                try:
                    file_path = user_msg.attachment.path
                    mime_type, _ = mimetypes.guess_type(user_msg.original_filename or "file")
                    if not mime_type and user_msg.attachment_type == 'image': mime_type = 'image/jpeg'
                    if mime_type and (mime_type.startswith('image/') or mime_type == 'application/pdf'):
                         with open(file_path, 'rb') as f:
                            input_parts_for_ai.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type))
                except Exception as e: print(f"[AI Service] Erro ao processar arquivo: {e}")
            
            if user_msg.content: input_parts_for_ai.append({"text": user_msg.content})

        json_instruction_prompt = """
Based on the user's message and context, provide a helpful response.
Respond with a valid JSON object:
{
  "response": "Your main answer goes here.",
  "suggestions": ["Suggestion 1", "Suggestion 2"]
}
"""
        input_parts_for_ai.append({"text": json_instruction_prompt})
        
        contents = gemini_history + [{"role": "user", "parts": input_parts_for_ai}]
        
        # Chamada Gemini
        print(f"[AI Service] Generating content... Reply with audio: {reply_with_audio}")
        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents, config=generation_config)
        
        result_data = {'content': "My response was blocked.", 'suggestions': [], 'audio_path': None, 'duration_ms': 0}

        if response.text:
            try:
                cleaned_text = re.sub(r'^```(json)?\s*', '', response.text.strip(), flags=re.MULTILINE)
                cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE).strip()
                response_data = json.loads(cleaned_text)
                
                result_data['content'] = response_data.get('response', "Sorry, I couldn't generate a response.")
                result_data['suggestions'] = response_data.get('suggestions', [])
            except json.JSONDecodeError:
                cleaned_text = re.sub(r'^```(json)?\n', '', response.text.strip(), flags=re.MULTILINE)
                result_data['content'] = re.sub(r'\n```$', '', cleaned_text.strip(), flags=re.MULTILINE)

        # --- TTS Generation (ATOMIC) ---
        # Se reply_with_audio for True, geramos o arquivo AGORA, antes de retornar.
        if reply_with_audio and result_data['content']:
            print("[AI Service] Generating TTS audio immediately...")
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    # Chama função que gera áudio E calcula duração
                    tts_result = generate_tts_audio(result_data['content'], temp_audio_file.name)
                    if tts_result['success']:
                        result_data['audio_path'] = tts_result['file_path']
                        result_data['duration_ms'] = tts_result.get('duration_ms', 0) # Salva duração
                        print(f"[AI Service] TTS Audio generated. Duration: {result_data['duration_ms']}ms")
                    else:
                        print(f"[AI Service] TTS Generation failed: {tts_result.get('error')}")
            except Exception as e:
                print(f"[AI Service] Exception during TTS generation: {e}")

        return result_data
    
    except Exception as e:
        print(f"An error occurred in Gemini AI service: {e}")
        return {'content': "An unexpected error occurred.", 'suggestions': [], 'audio_path': None, 'duration_ms': 0}

def transcribe_audio_gemini(audio_file: UploadedFile) -> dict:
    try:
        client = get_ai_client()
        audio_bytes = audio_file.read()
        mime_type, _ = mimetypes.guess_type(audio_file.name)
        if not mime_type: mime_type = 'audio/m4a'
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        prompt = "Generate a transcript of the speech in Portuguese. Return only the transcribed text, strictly without timestamps or speaker labels."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, audio_part], config=types.GenerateContentConfig(temperature=0.2))
        if response.text: return {'success': True, 'transcription': response.text.strip()}
        else: return {'success': False, 'error': 'No transcription returned'}
    except Exception as e:
        print(f"[Transcription Error] {e}")
        return {'success': False, 'error': str(e)}

def generate_tts_audio(message_text: str, output_path: str) -> dict:
    """
    Gera áudio TTS usando Gemini e calcula a duração do arquivo WAV.
    """
    try:
        import wave
        client = get_ai_client()
        safe_text = message_text[:2000] 
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-tts',
            contents=safe_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")))
            )
        )
        if not response.candidates or not response.candidates[0].content.parts: raise Exception("Nenhum áudio gerado.")
        audio_part = next((p.inline_data for p in response.candidates[0].content.parts if p.inline_data), None)
        if not audio_part: raise Exception("Nenhum dado de áudio encontrado.")
        
        # Salva os dados PCM como WAV e calcula duração
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_part.data)
            
            # Cálculo de duração
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_ms = int((frames / float(rate)) * 1000)

        return {'success': True, 'file_path': output_path, 'duration_ms': duration_ms}
    except Exception as e:
        print(f"[TTS Error] {e}")
        return {'success': False, 'error': str(e)}

def handle_voice_interaction(chat_id: int, audio_file: UploadedFile, user) -> dict:
    result = handle_voice_message(chat_id, audio_file, reply_with_audio=False, user=user)
    return {
        "transcription": result['user_message'].content,
        "ai_response_text": result['ai_message'].content,
        "user_message": result['user_message'],
        "ai_messages": [result['ai_message']]
    }

def handle_voice_message(chat_id: int, user_audio_file: UploadedFile, reply_with_audio: bool, user) -> dict:
    """
    Processa mensagem de voz do usuário e gera resposta.
    """
    with transaction.atomic():
        chat = Chat.objects.get(id=chat_id)
        
        # 1. Transcrever
        user_audio_file.seek(0) 
        trans_result = transcribe_audio_gemini(user_audio_file)
        transcription = trans_result['transcription'] if trans_result['success'] else "[Áudio - Transcrição indisponível]"
        user_audio_file.seek(0)
        
        # 2. Salvar mensagem do usuário (Duração deve ser atualizada na View com dado do frontend)
        user_message = ChatMessage.objects.create(
            chat=chat, role=ChatMessage.Role.USER, content=transcription,
            attachment=user_audio_file, attachment_type='audio', original_filename=user_audio_file.name or "voice_message.m4a"
        )
        chat.last_message_at = timezone.now()
        chat.save()

        # 3. Gerar resposta IA
        ai_response_data = get_ai_response(chat_id, transcription, user_message_obj=user_message, reply_with_audio=reply_with_audio)
        
        ai_text = ai_response_data.get('content', '')
        ai_suggestions = ai_response_data.get('suggestions', [])
        audio_path = ai_response_data.get('audio_path')
        duration_ms = ai_response_data.get('duration_ms', 0)

        # 4. Criar mensagem IA
        ai_message = ChatMessage(
            chat=chat, 
            role=ChatMessage.Role.ASSISTANT, 
            content=ai_text,
            suggestion1=ai_suggestions[0] if len(ai_suggestions) > 0 else None,
            suggestion2=ai_suggestions[1] if len(ai_suggestions) > 1 else None,
            duration=duration_ms # Salva duração do TTS
        )

        if audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, 'rb') as f:
                    filename = f"reply_tts_{uuid.uuid4().hex[:10]}.wav"
                    ai_message.attachment.save(filename, File(f), save=False)
                    ai_message.attachment_type = 'audio'
                    ai_message.original_filename = "voice_reply.wav"
                os.remove(audio_path)
            except Exception as e:
                print(f"[Handle Voice] Error attaching audio: {e}")
                ai_message.attachment_type = None 

        ai_message.save()
        chat.last_message_at = timezone.now()
        chat.save()

        return {"user_message": user_message, "ai_message": ai_message}