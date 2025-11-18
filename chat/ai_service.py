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
import tempfile
from pathlib import Path
from django.utils import timezone 
from datetime import timedelta 

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
        
        # ✅ CORREÇÃO 3: Limpa a resposta de forma mais robusta
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
    AGORA SUPORTA MÚLTIPLOS ANEXOS (imagens/arquivos) enviados
    antes do prompt de texto.
    
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
        
        # --- LÓGICA DE HISTÓRICO E PROMPT ATUALIZADA ---
        
        # 1. Encontra a última mensagem do assistente
        last_assistant_message = ChatMessage.objects.filter(
            chat_id=chat_id, 
            role=ChatMessage.Role.ASSISTANT
        ).order_by('-created_at').first()
        
        # Define o "ponto de corte": mensagens mais novas que isso são o PROMPT ATUAL
        if last_assistant_message:
            since_timestamp = last_assistant_message.created_at
        else:
            # Se a IA nunca falou, usa o início do chat
            since_timestamp = chat.created_at - timedelta(seconds=1) 

        # 2. Busca o HISTÓRICO (mensagens ANTES do ponto de corte)
        history_qs = ChatMessage.objects.filter(
            chat_id=chat_id, 
            created_at__lt=since_timestamp
        ).order_by('-created_at')[:10] # Pega as 10 mais antigas
        
        # ✅ CORREÇÃO 4: Não pegar mais 'original_filename' ou 'attachment_type'
        history_values = history_qs.values('role', 'content') 
        history_values = reversed(history_values) # Re-ordena para (mais antigo -> mais novo)
        
        gemini_history = []
        for msg in history_values:
            if "An unexpected error occurred" in msg.get('content', ''):
                continue
            
            role = 'user' if msg.get('role') == 'user' else 'model'
            parts = []
            
            # Só adiciona o conteúdo de TEXTO ao histórico
            if msg.get('content'):
                parts.append({"text": msg.get('content')})
            
            # ✅ CORREÇÃO 4: Lógica de placeholder de anexo REMOVIDA
            # Não adicionamos mais placeholders de anexos antigos.
            # Isso impede que a IA fique confusa em prompts futuros (como "Oi").
            
            if parts:
                gemini_history.append({
                    "role": role,
                    "parts": parts
                })

        # 3. Busca o PROMPT ATUAL (mensagens DEPOIS do ponto de corte)
        # Isso inclui todas as imagens/arquivos + a mensagem de texto final
        prompt_messages_qs = ChatMessage.objects.filter(
            chat_id=chat_id,
            role=ChatMessage.Role.USER,
            created_at__gte=since_timestamp
        ).order_by('created_at') # Garante a ordem correta (Img1, Img2, Texto)

        # --- CONSTRÓI O PROMPT MULTIMODAL (COM MÚLTIPLOS ARQUIVOS) ---
        input_parts_for_ai = []
        
        for user_msg in prompt_messages_qs:
            # Processa ANEXOS de CADA mensagem do prompt
            if user_msg.attachment and user_msg.attachment.path:
                file_path = user_msg.attachment.path
                original_filename = user_msg.original_filename
                
                try:
                    print(f"[AI Service] Processando anexo para prompt: {original_filename}...")
                    
                    mime_type, _ = mimetypes.guess_type(original_filename)
                    
                    if not mime_type and user_msg.attachment_type == 'image':
                        _, ext = os.path.splitext(original_filename)
                        ext_lower = ext.lower()
                        mime_map = {
                            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                            '.png': 'image/png', '.gif': 'image/gif',
                            '.webp': 'image/webp', '.bmp': 'image/bmp'
                        }
                        mime_type = mime_map.get(ext_lower, 'image/jpeg')
                    
                    print(f"[AI Service] MIME type: {mime_type}")
                    
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    input_parts_for_ai.append(
                        types.Part.from_bytes(
                            data=file_data,
                            mime_type=mime_type
                        )
                    )
                    print(f"[AI Service] Arquivo adicionado inline ({len(file_data)} bytes)")
                
                except Exception as e:
                    print(f"[AI Service] Erro ao processar arquivo: {e}")
                    attachment_type_name = "imagem" if user_msg.attachment_type == "image" else "arquivo"
                    input_parts_for_ai.append({"text": f"[Erro ao processar {attachment_type_name}: {original_filename}]"})
            
            # Adiciona o TEXTO da mensagem (se houver)
            # A mensagem final (ex: "O que são essas imagens?") será adicionada aqui.
            if user_msg.content:
                input_parts_for_ai.append({"text": user_msg.content})


        # Adiciona as instruções de formatação JSON no final
        json_instruction_prompt = """
Based on the user's message (and any attached image/file) and the conversation history, provide a helpful response.
If a file or image was provided, base your response on its content (e.g., summarize the PDF, describe the image, answer questions about the text file).
If MULTIPLE images/files were provided, analyze them all together to answer the user's text prompt.
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
            model='gemini-2.5-flash',
            contents=contents,
            config=generation_config
        )
        
        # --- PROCESSA A RESPOSTA ---
        if response.text:
            try:
                # ✅ CORREÇÃO 3: Limpeza de JSON mais robusta
                cleaned_text = response.text.strip()
                # Remove ```json e ```
                cleaned_text = re.sub(r'^```(json)?\s*', '', cleaned_text, flags=re.MULTILINE)
                cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE)
                cleaned_text = cleaned_text.strip() # Remove qualquer espaço em branco extra
                
                response_data = json.loads(cleaned_text)
                
                return {
                    'content': response_data.get('response', "Sorry, I couldn't generate a response."),
                    'suggestions': response_data.get('suggestions', [])
                }
            
            except json.JSONDecodeError as json_err:
                print(f"Gemini JSON parse error: {json_err}. Raw text: {response.text}")
                # Se falhar o parse, retorna o texto bruto (sem o lixo do markdown)
                cleaned_text = re.sub(r'^```(json)?\n', '', response.text.strip(), flags=re.MULTILINE)
                cleaned_text = re.sub(r'\n```$', '', cleaned_text.strip(), flags=re.MULTILINE)
                return {
                    'content': cleaned_text,
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


def transcribe_audio_gemini(audio_file: UploadedFile) -> dict:
    """
    Transcreve áudio usando Google Gemini API
    
    Args:
        audio_file: Arquivo de áudio enviado pelo usuário
    
    Returns:
        dict: {'success': bool, 'transcription': str, 'error': str (opcional)}
    """
    try:
        print(f"[Gemini Transcription] Iniciando transcrição de: {audio_file.name}")
        print(f"[Gemini Transcription] Tamanho do arquivo: {audio_file.size} bytes")
        
        client = get_ai_client()
        
        # Lê os bytes do arquivo de áudio
        audio_bytes = audio_file.read()
        
        # Determina o MIME type do áudio
        mime_type, _ = mimetypes.guess_type(audio_file.name)
        
        # Se não conseguir detectar, tenta pela extensão
        if not mime_type:
            _, ext = os.path.splitext(audio_file.name)
            ext_lower = ext.lower()
            
            # Mapeamento de extensões de áudio comuns
            audio_mime_map = {
                '.m4a': 'audio/m4a',
                '.mp3': 'audio/mp3',
                '.wav': 'audio/wav',
                '.aac': 'audio/aac',
                '.ogg': 'audio/ogg',
                '.flac': 'audio/flac',
                '.3gp': 'audio/3gpp',
                '.webm': 'audio/webm'
            }
            
            mime_type = audio_mime_map.get(ext_lower, 'audio/m4a')
        
        print(f"[Gemini Transcription] MIME type detectado: {mime_type}")
        
        # Cria o part do áudio inline
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type
        )
        
        # Prompt para transcrição
        prompt = "Generate a transcript of the speech in Portuguese. Return only the transcribed text, without any additional formatting or explanation."
        
        print("[Gemini Transcription] Enviando para API Gemini...")
        
        # Chama a API Gemini com o áudio inline
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                prompt,
                audio_part
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,  # Baixa temperatura para transcrições mais precisas
            )
        )
        
        if response.text:
            transcription = response.text.strip()
            print(f"[Gemini Transcription] Transcrição bem-sucedida: {transcription[:100]}...")
            
            return {
                'success': True,
                'transcription': transcription
            }
        else:
            print("[Gemini Transcription] Nenhum texto retornado na resposta")
            return {
                'success': False,
                'error': 'No transcription returned from Gemini'
            }
    
    except Exception as e:
        print(f"[Gemini Transcription] Erro ao transcrever áudio: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def generate_tts_audio(message_text: str, output_path: str) -> dict:
    """
    Gera áudio TTS usando Gemini 2.5 Flash TTS.
    Retorna dict com informações do áudio gerado.
    """
    try:
        import wave
        
        client = get_ai_client()
        print(f"[TTS Service] Gerando áudio para mensagem...")
        
        # ✅ CORREÇÃO: Usar modelo TTS correto
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-tts',  # Modelo com suporte a TTS
            contents=message_text,  # Texto direto, sem prefixo "TTS:"
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],  # Solicita saída de áudio
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore"  # Voz feminina natural
                        )
                    )
                )
            )
        )
        
        # Extrai os dados de áudio da resposta
        if not response.candidates or not response.candidates[0].content.parts:
            raise Exception("Nenhum áudio gerado na resposta")
        
        audio_part = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                audio_part = part.inline_data
                break
        
        if not audio_part:
            raise Exception("Nenhum áudio encontrado na resposta")
        
        # Os dados de áudio vêm como PCM bruto
        pcm_data = audio_part.data
        
        # Configurações de áudio padrão do Gemini TTS
        sample_rate = 24000  # 24kHz
        channels = 1  # Mono
        sample_width = 2  # 16-bit PCM
        
        # Salva como arquivo WAV
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        
        print(f"[TTS Service] Áudio salvo em: {output_path}")
        
        return {
            'success': True,
            'file_path': output_path,
            'duration_seconds': len(pcm_data) / (sample_rate * channels * sample_width)
        }
    
    except Exception as e:
        print(f"[TTS Service] Erro ao gerar áudio: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }