from google import genai
from google.genai import types
import os
import json
import mimetypes
from django.conf import settings
from .models import ChatMessage, Chat
from bots.models import Bot
import re
from django.core.files.uploadedfile import UploadedFile
from django.core.files import File
import tempfile
from pathlib import Path
from django.utils import timezone
from datetime import datetime, timedelta
from django.db import transaction
import uuid
import wave
from .vector_service import VectorService
import threading
import logging

# Logger
logger = logging.getLogger(__name__)

# Instância global do serviço de vetor
vector_service = VectorService()


def get_ai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada")
    client = genai.Client(api_key=api_key)
    return client


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def _build_conversation_history(chat_id: int, limit: int = 15, exclude_message_id: int = None):
    """
    Constrói histórico LINEAR das últimas N mensagens.
    Evita gaps e mantém contexto contínuo.
    
    Returns:
        tuple: (gemini_history, recent_texts)
            - gemini_history: Lista formatada para o Gemini
            - recent_texts: Lista de textos recentes (para filtrar memória)
    """
    queryset = ChatMessage.objects.filter(chat_id=chat_id)
    
    if exclude_message_id:
        queryset = queryset.exclude(id=exclude_message_id)
    
    messages = queryset.order_by('-created_at')[:limit]
    messages = list(reversed(messages))  # Ordem cronológica
    
    gemini_history = []
    recent_texts = []
    
    for msg in messages:
        if not msg.content:
            continue
        if "unexpected error" in msg.content.lower():
            continue
        
        role = 'user' if msg.role == 'user' else 'model'
        gemini_history.append({
            "role": role,
            "parts": [{"text": msg.content}]
        })
        recent_texts.append(msg.content)
    
    return gemini_history, recent_texts


def _build_system_instruction(bot_prompt: str, user_name: str, memories: list, current_time: str):
    """
    Constrói o system instruction otimizado para roleplay natural.
    Prioridade clara: Personagem > Contexto > Memória > Regras
    """
    
    # Seção de memória (apenas se houver memórias relevantes)
    memory_section = ""
    if memories:
        memory_items = "\n".join([f"• {mem}" for mem in memories])
        memory_section = f"""
## MEMÓRIA DE LONGO PRAZO
Fatos relevantes sobre {user_name} (use naturalmente, sem listar):
{memory_items}
"""
    
    return f"""# PERSONAGEM
{bot_prompt}

## CONTEXTO ATUAL
- Você está conversando com: {user_name}
- Data/Hora: {current_time}
{memory_section}
## DIRETRIZES IMPORTANTES

1. **MANTENHA O PERSONAGEM** - Você É o personagem definido acima, não uma IA. Nunca quebre o personagem.

2. **RESPONDA APENAS À MENSAGEM ATUAL** - Não repita informações que você já disse em mensagens anteriores. O usuário pode ver o histórico.

3. **USE MEMÓRIAS SUTILMENTE** - Se souber algo sobre o usuário (como o nome do pet), mencione naturalmente em contexto apropriado, não force.

4. **SEJA CONCISO E NATURAL** - Responda como uma pessoa real responderia em uma conversa. Evite respostas muito longas a menos que seja necessário.

5. **EVITE RECAPITULAR** - Não resuma conversas anteriores. Assuma que o usuário lembra do que foi dito. Avance a conversa.

6. **FORMATAÇÃO** - Use Markdown (negrito, listas) apenas quando realmente ajudar na clareza. Personagens casuais não usam formatação excessiva."""


def _parse_ai_response(response_text: str) -> dict:
    """
    Faz parse da resposta da IA, extraindo conteúdo e sugestões.
    Suporta tanto formato JSON quanto formato natural com marcador.
    """
    result = {
        'content': "",
        'suggestions': [],
        'audio_path': None,
        'duration_ms': 0
    }
    
    if not response_text:
        result['content'] = "Desculpe, não consegui gerar uma resposta."
        return result
    
    text = response_text.strip()
    
    # Tenta primeiro como JSON (compatibilidade)
    try:
        # Remove markdown code blocks se existirem (CORRIGIDO AQUI)
        cleaned = re.sub(r'^```\w*\s*', '', text) 
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
        
        data = json.loads(cleaned)
        if isinstance(data, dict):
            result['content'] = data.get('response', data.get('content', ''))
            result['suggestions'] = data.get('suggestions', [])[:2]
            if result['content']:
                return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Formato natural com marcador ---SUGESTÕES---
    if "---SUGESTÕES---" in text or "---SUGGESTIONS---" in text:
        separator = "---SUGESTÕES---" if "---SUGESTÕES---" in text else "---SUGGESTIONS---"
        parts = text.split(separator)
        result['content'] = parts[0].strip()
        
        if len(parts) > 1:
            suggestions_text = parts[1]
            # Extrai sugestões numeradas (1. xxx ou - xxx)
            suggestions = re.findall(r'(?:^|\n)\s*(?:\d+\.|-)\s*(.+)', suggestions_text)
            result['suggestions'] = [s.strip() for s in suggestions[:2] if s.strip()]
    else:
        # Resposta pura sem sugestões
        result['content'] = text
    
    # Limpa formatação residual
    result['content'] = result['content'].strip()
    
    return result


# =============================================================================
# FUNÇÕES DE MEMÓRIA (BACKGROUND)
# =============================================================================

def _summarize_fact(text: str, role: str = 'user') -> str:
    """
    Usa a IA para extrair um fato conciso e duradouro do texto.
    Retorna string vazia se não houver fato relevante.
    """
    if not text or len(text) < 15:
        return ""
    
    try:
        client = get_ai_client()
        
        prompt = f"""Analise o texto abaixo e extraia APENAS fatos concretos e duradouros que valem a pena lembrar.

Texto ({role}): "{text}"

REGRAS:
- Retorne "NO_FACT" para: saudações, agradecimentos, perguntas genéricas, conversa casual
- Retorne "NO_FACT" se for apenas uma pergunta sem informação nova
- Fatos devem ser em terceira pessoa: "O usuário tem um cachorro chamado Rex"
- Máximo 1 frase concisa (menos de 20 palavras)
- Foque em: preferências, informações pessoais, contextos importantes, planos

Responda APENAS com o fato extraído ou "NO_FACT"."""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        
        summary = response.text.strip()
        
        if "NO_FACT" in summary.upper() or len(summary) < 10:
            return ""
        
        # Remove aspas e formatação extra
        summary = summary.strip('"\'')
        
        return summary
        
    except Exception as e:
        logger.warning(f"[Memory Summary Error] {e}")
        return ""


def _process_memory_background(user_id, bot_id, user_text, ai_text):
    """
    Função executada em thread separada para processar e salvar memórias.
    """
    try:
        # 1. Processar mensagem do Usuário (prioridade)
        if user_text and len(user_text) > 25:
            fact = _summarize_fact(user_text, 'user')
            if fact:
                vector_service.add_memory(user_id, bot_id, fact, 'user')
                logger.debug(f"[Memory] Fato do usuário salvo: {fact[:50]}...")

        # 2. Processar mensagem da IA (apenas se contiver informação nova significativa)
        if ai_text and len(ai_text) > 80:
            fact = _summarize_fact(ai_text, 'assistant')
            if fact:
                vector_service.add_memory(user_id, bot_id, fact, 'assistant')
                logger.debug(f"[Memory] Fato da IA salvo: {fact[:50]}...")

    except Exception as e:
        logger.error(f"[Background Memory Error] {e}")


# =============================================================================
# FUNÇÕES PRINCIPAIS
# =============================================================================

def generate_suggestions_for_bot(prompt: str):
    """Gera sugestões iniciais para um bot baseado no prompt."""
    try:
        client = get_ai_client()
        
        instruction = f"""Based on the following bot's instructions, generate exactly three short, engaging, and distinct conversation starters (under 10 words each).
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
        
        # CORRIGIDO AQUI: Regex estava quebrado
        cleaned_response = re.sub(r'^```\w*\s*', '', response.text)
        cleaned_response = re.sub(r'\n```$', '', cleaned_response.strip(), flags=re.MULTILINE)
        
        suggestions = json.loads(cleaned_response)
        
        if isinstance(suggestions, list) and len(suggestions) > 0 and all(isinstance(s, str) for s in suggestions):
            return suggestions[:3]
            
    except Exception as e:
        logger.warning(f"Could not generate suggestions: {e}")
    
    return ["Tell me more.", "What can you do?", "Give me an example."]


def get_ai_response(chat_id: int, user_message_text: str, user_message_obj: ChatMessage = None, reply_with_audio: bool = False):
    """
    Obtém resposta da IA com contexto histórico e memória de longo prazo.
    
    Args:
        chat_id: ID do chat
        user_message_text: Texto da mensagem do usuário
        user_message_obj: Objeto da mensagem (opcional, para anexos)
        reply_with_audio: Se True, gera áudio TTS junto com a resposta
    
    Returns:
        dict com 'content', 'suggestions', 'audio_path', 'duration_ms'
    """
    try:
        client = get_ai_client()
        chat = Chat.objects.select_related('bot', 'user').get(id=chat_id)
        bot = chat.bot
        
        # 1. Preparação dos Dados Básicos
        user_defined_prompt = bot.prompt.strip() if bot.prompt else "Você é um assistente de IA útil e amigável."
        user_name = chat.user.first_name if chat.user.first_name else "Usuário"
        current_time_str = datetime.now().strftime('%d/%m/%Y %H:%M')
        
        # 2. Construir Histórico (LINEAR - sem gaps)
        exclude_id = user_message_obj.id if user_message_obj else None
        gemini_history, recent_texts = _build_conversation_history(
            chat_id, 
            limit=12,
            exclude_message_id=exclude_id
        )
        
        # 3. Buscar Memórias (excluindo textos do histórico recente)
        retrieved_memories = []
        try:
            retrieved_memories = vector_service.search_memory(
                user_id=chat.user_id,
                bot_id=bot.id,
                query_text=user_message_text,
                limit=4,
                exclude_texts=recent_texts
            )
            if retrieved_memories:
                logger.info(f"[AI Service] Memória injetada: {len(retrieved_memories)} itens")
        except Exception as mem_error:
            logger.warning(f"[AI Service] Erro ao recuperar memória: {mem_error}")
        
        # 4. Construir System Instruction
        system_instruction = _build_system_instruction(
            bot_prompt=user_defined_prompt,
            user_name=user_name,
            memories=retrieved_memories,
            current_time=current_time_str
        )
        
        # 5. Configuração do Modelo
        generation_config = types.GenerateContentConfig(
            temperature=0.8,
            max_output_tokens=2500,
            top_p=0.95,
            system_instruction=system_instruction,
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
            ]
        )
        
        # 6. Preparar Input (mensagem atual + anexos)
        input_parts = []
        
        # Processar anexos da mensagem atual
        if user_message_obj and user_message_obj.attachment:
            try:
                if hasattr(user_message_obj.attachment, 'path') and user_message_obj.attachment.path:
                    file_path = user_message_obj.attachment.path
                    if os.path.exists(file_path):
                        mime_type, _ = mimetypes.guess_type(user_message_obj.original_filename or "file")
                        if not mime_type and user_message_obj.attachment_type == 'image':
                            mime_type = 'image/jpeg'
                        
                        if mime_type and (mime_type.startswith('image/') or mime_type == 'application/pdf'):
                            with open(file_path, 'rb') as f:
                                input_parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type))
            except Exception as e:
                logger.warning(f"[AI Service] Erro ao processar anexo: {e}")
        
        # Adicionar texto da mensagem com instrução de sugestões
        final_user_prompt = f"""{user_message_text}

---
Após sua resposta, forneça 2 sugestões curtas (máx 8 palavras cada) do que eu poderia perguntar/dizer em seguida.
Use este formato exato:

[Sua resposta aqui]

---SUGESTÕES---
1. [sugestão 1]
2. [sugestão 2]"""
        
        input_parts.append({"text": final_user_prompt})
        
        # 7. Montar Contents
        contents = gemini_history + [{"role": "user", "parts": input_parts}]
        
        # 8. Chamada à API
        logger.info(f"[AI Service] Gerando resposta... Audio: {reply_with_audio}")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=generation_config
        )
        
        # 9. Parse da Resposta
        response_text = response.text if response.text else ""
        result_data = _parse_ai_response(response_text)
        
        if not result_data['content']:
            result_data['content'] = "Desculpe, minha resposta foi bloqueada."
        
        # 10. Salvar Memória em Background
        if result_data['content'] and len(user_message_text) > 10:
            threading.Thread(
                target=_process_memory_background,
                args=(chat.user_id, bot.id, user_message_text, result_data['content'])
            ).start()
        
        # 11. Gerar TTS se solicitado
        if reply_with_audio and result_data['content']:
            logger.info("[AI Service] Gerando áudio TTS...")
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    tts_result = generate_tts_audio(result_data['content'], temp_audio_file.name)
                    if tts_result['success']:
                        result_data['audio_path'] = tts_result['file_path']
                        result_data['duration_ms'] = tts_result.get('duration_ms', 0)
                        logger.info(f"[AI Service] TTS gerado. Duração: {result_data['duration_ms']}ms")
                    else:
                        logger.warning(f"[AI Service] TTS falhou: {tts_result.get('error')}")
            except Exception as e:
                logger.error(f"[AI Service] Exceção no TTS: {e}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Erro no serviço de IA: {e}")
        import traceback
        traceback.print_exc()
        return {
            'content': "Ocorreu um erro inesperado. Por favor, tente novamente.",
            'suggestions': [],
            'audio_path': None,
            'duration_ms': 0
        }


def transcribe_audio_gemini(audio_file: UploadedFile) -> dict:
    """Transcreve áudio usando Gemini."""
    try:
        client = get_ai_client()
        audio_bytes = audio_file.read()
        
        mime_type, _ = mimetypes.guess_type(audio_file.name)
        if not mime_type:
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
        logger.error(f"[Transcription Error] {e}")
        return {'success': False, 'error': str(e)}


def generate_tts_audio(message_text: str, output_path: str) -> dict:
    """
    Gera áudio TTS usando Gemini e calcula a duração do arquivo WAV.
    """
    try:
        client = get_ai_client()
        
        # Limita texto para TTS
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
            raise Exception("Nenhum áudio gerado.")
        
        audio_part = None
        for p in response.candidates[0].content.parts:
            if p.inline_data:
                audio_part = p.inline_data
                break
        
        if not audio_part:
            raise Exception("Nenhum dado de áudio encontrado.")
        
        # Salva como WAV
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_part.data)
        
        # Calcula duração
        with wave.open(output_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_ms = int((frames / float(rate)) * 1000)
        
        return {'success': True, 'file_path': output_path, 'duration_ms': duration_ms}
        
    except Exception as e:
        logger.error(f"[TTS Error] {e}")
        return {'success': False, 'error': str(e)}


def handle_voice_interaction(chat_id: int, audio_file: UploadedFile, user) -> dict:
    """Handler para interação de voz (sem resposta em áudio)."""
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
        
        # 2. Salvar mensagem do usuário
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
        
        # 3. Gerar resposta IA
        ai_response_data = get_ai_response(
            chat_id,
            transcription,
            user_message_obj=user_message,
            reply_with_audio=reply_with_audio
        )
        
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
            duration=duration_ms
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
                logger.error(f"[Handle Voice] Erro ao anexar áudio: {e}")
                ai_message.attachment_type = None
        
        ai_message.save()
        
        chat.last_message_at = timezone.now()
        chat.save()
        
        return {"user_message": user_message, "ai_message": ai_message}