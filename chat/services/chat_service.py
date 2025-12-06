"""
Serviço principal de chat com IA.
Orquestra busca de contexto multi-doc, geração de resposta e salvamento de memória.
"""

import os
import json
import re
import mimetypes
import threading
import logging
import tempfile
import uuid

from datetime import datetime
from django.utils import timezone
from django.db import transaction
from django.core.files import File

from google.genai import types

from ..models import ChatMessage, Chat
from ..vector_service import VectorService
from .ai_client import get_ai_client, detect_intent
from .image_service import ImageGenerationService  # Importação do novo serviço
from .context_builder import (
    build_conversation_history, 
    build_system_instruction,
    get_recent_attachment_context  # NOVO
)
from .memory_service import process_memory_background
from .tts_service import generate_tts_audio
from .transcription_service import transcribe_audio_gemini

logger = logging.getLogger(__name__)

# Instância global do serviço vetorial
vector_service = VectorService()
# Instância global do serviço de imagem
image_service = ImageGenerationService()


def _parse_ai_response(response_text: str) -> dict:
    """
    Faz parse da resposta da IA, extraindo conteúdo e sugestões.
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

    # Tenta primeiro como JSON
    try:
        # CORREÇÃO: A regex estava cortada aqui.
        # Remove ```json ou ``` no início
        cleaned = re.sub(r'^```\w*', '', text, flags=re.MULTILINE)
        # Remove ``` no final
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
            suggestions = re.findall(r'(?:^|\n)\s*(?:\d+\.|-)\s*(.+)', suggestions_text)
            result['suggestions'] = [s.strip() for s in suggestions[:2] if s.strip()]
    else:
        result['content'] = text

    result['content'] = result['content'].strip()
    return result


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

        # CORREÇÃO: A regex também estava cortada aqui
        text_content = response.text if response.text else "[]"
        cleaned_response = re.sub(r'^```\w*', '', text_content, flags=re.MULTILINE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response, flags=re.MULTILINE).strip()

        suggestions = json.loads(cleaned_response)

        if (
            isinstance(suggestions, list)
            and len(suggestions) > 0
            and all(isinstance(s, str) for s in suggestions)
        ):
            return suggestions[:3]

    except Exception as e:
        logger.warning(f"Could not generate suggestions: {e}")

    return ["Tell me more.", "What can you do?", "Give me an example."]


def get_ai_response(
    chat_id: int,
    user_message_text: str,
    user_message_obj: ChatMessage = None,
    reply_with_audio: bool = False
) -> dict:
    """
    Obtém resposta da IA com contexto histórico, memória e RAG Multi-Documento.
    Agora inclui detecção de intenção para geração de imagens.
    """
    try:
        # 0. DETECÇÃO DE INTENÇÃO (NOVO)
        # Só detecta se não tiver anexo de arquivo (se tiver anexo, assume que é para analisar o arquivo)
        intent = 'TEXT'
        if not (user_message_obj and user_message_obj.attachment):
            try:
                intent = detect_intent(user_message_text)
                logger.info(f"[ChatService] Intenção detectada: {intent}")
            except Exception as e:
                logger.warning(f"[ChatService] Falha na detecção de intenção: {e}")
                intent = 'TEXT'

        # --- FLUXO DE GERAÇÃO DE IMAGEM ---
        if intent == 'IMAGE':
            try:
                logger.info(f"[ChatService] Iniciando fluxo de geração de imagem para: {user_message_text[:30]}...")
                image_rel_path = image_service.generate_and_save_image(user_message_text)
                
                # Retorna estrutura similar ao parse_ai_response, mas com metadados de imagem
                # A View (ChatMessageListView) precisará ser adaptada se quiser usar 'image_path'
                # ou podemos salvar a mensagem da IA aqui mesmo?
                # O padrão atual do projeto é retornar dict e a View salva.
                # Mas para imagem, precisamos passar o path do anexo.
                
                # Vamos retornar um dict especial que a View pode entender, ou instruir a View
                # O método atual na View espera 'content', 'suggestions', 'audio_path'.
                # Vamos adicionar 'generated_image_path' ao retorno.
                
                return {
                    'content': f"Aqui está a imagem que criei para você com base em \"{user_message_text}\".",
                    'suggestions': ["Gere outra variação", "Mude o estilo", "Obrigado!"],
                    'audio_path': None,
                    'duration_ms': 0,
                    'generated_image_path': image_rel_path # Campo novo para a View tratar
                }

            except Exception as img_err:
                logger.error(f"[ChatService] Erro ao gerar imagem: {img_err}")
                # Fallback para texto se falhar a imagem
                return {
                    'content': f"Tentei gerar a imagem, mas encontrei um erro: {str(img_err)}. Posso tentar novamente ou ajudar com outra coisa?",
                    'suggestions': [],
                    'audio_path': None
                }

        # --- FLUXO DE TEXTO (EXISTENTE) ---
        
        client = get_ai_client()
        chat = Chat.objects.select_related('bot', 'user').get(id=chat_id)
        bot = chat.bot

        # 1. Preparação Básica
        user_defined_prompt = bot.prompt.strip() if bot.prompt else "Você é um assistente útil."
        user_name = chat.user.first_name if chat.user.first_name else "Usuário"
        current_time_str = datetime.now().strftime('%d/%m/%Y %H:%M')

        # 2. Construir Histórico Recente
        exclude_id = user_message_obj.id if user_message_obj else None
        gemini_history, recent_texts = build_conversation_history(
            chat_id, limit=12, exclude_message_id=exclude_id
        )

        # 3. BUSCA INTELIGENTE MULTI-DOC (RAG + MEMÓRIA)
        doc_contexts, memory_contexts, available_doc_names = _get_smart_context(
            query=user_message_text,
            user_id=chat.user_id,
            bot_id=bot.id,
            chat_id=chat_id
        )

        # 4. System Instruction COM LISTA DE DOCS
        system_instruction = build_system_instruction(
            bot_prompt=user_defined_prompt,
            user_name=user_name,
            doc_contexts=doc_contexts,
            memory_contexts=memory_contexts,
            current_time=current_time_str,
            available_docs=available_doc_names  # NOVO: passa lista de docs
        )

        # 5. Config
        generation_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2500,
            system_instruction=system_instruction
        )

        # 6. Preparar Input (Texto + Anexos atuais)
        input_parts = []

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
                                input_parts.append(
                                    types.Part.from_bytes(data=f.read(), mime_type=mime_type)
                                )
            except Exception as e:
                logger.warning(f"[AI Service] Erro anexo: {e}")

        final_user_prompt = f"""{user_message_text}

---

Após responder, forneça 2 sugestões curtas de continuação no formato:

[Resposta]

---SUGESTÕES---
1. [Sugestão 1]
2. [Sugestão 2]"""

        input_parts.append({"text": final_user_prompt})

        contents = gemini_history + [{"role": "user", "parts": input_parts}]

        # 7. Chamada à API
        logger.info(f"[AI Service] Gerando resposta para chat {chat_id}...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=generation_config
        )

        # 8. Processamento da Resposta
        result_data = _parse_ai_response(response.text if response.text else "")

        # 9. Salvar Memória em Background
        if result_data['content'] and len(user_message_text) > 10:
            threading.Thread(
                target=process_memory_background,
                args=(chat.user_id, bot.id, user_message_text, result_data['content'])
            ).start()

        # 10. TTS (Opcional)
        if reply_with_audio and result_data['content']:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    tts = generate_tts_audio(result_data['content'], temp_audio.name)
                    if tts['success']:
                        result_data['audio_path'] = tts['file_path']
                        result_data['duration_ms'] = tts.get('duration_ms', 0)
            except Exception as e:
                logger.error(f"[TTS Error] {e}")

        return result_data

    except Exception as e:
        logger.error(f"Erro AI Service: {e}")
        return {'content': "Erro ao processar resposta.", 'suggestions': [], 'audio_path': None}


def _get_smart_context(
    query: str,
    user_id: int,
    bot_id: int,
    chat_id: int
) -> tuple:
    """
    Busca contexto de forma inteligente usando o VectorService multi-doc.
    
    Returns:
        Tuple: (doc_contexts, memory_contexts, available_doc_names)
    """
    try:
        # Pega documento mais recente do chat (para referências pronominais)
        recent_source = get_recent_attachment_context(chat_id)
        
        if recent_source:
            logger.info(f"[RAG] Documento mais recente no chat: {recent_source}")
        
        # Busca com classificação automática de query (multi-doc)
        doc_contexts, memory_contexts = vector_service.search_context(
            query_text=query,
            user_id=user_id,
            bot_id=bot_id,
            limit=6,
            recent_doc_source=recent_source
        )
        
        # Lista de documentos disponíveis (para o system instruction)
        available_docs = vector_service.get_available_documents(user_id, bot_id)
        available_names = [d['source'] for d in available_docs]
        
        logger.info(f"[RAG] Docs encontrados: {len(doc_contexts)}, Memórias: {len(memory_contexts)}, Disponíveis: {available_names}")
        
        return doc_contexts, memory_contexts, available_names
        
    except Exception as e:
        logger.warning(f"[RAG] Erro na busca de contexto: {e}")
        return [], [], []


def handle_voice_interaction(chat_id: int, audio_file, user) -> dict:
    """Handler para interação de voz (sem resposta em áudio)."""
    result = handle_voice_message(chat_id, audio_file, reply_with_audio=False, user=user)
    return {
        "transcription": result['user_message'].content,
        "ai_response_text": result['ai_message'].content,
        "user_message": result['user_message'],
        "ai_messages": [result['ai_message']]
    }


def handle_voice_message(chat_id: int, user_audio_file, reply_with_audio: bool, user) -> dict:
    """Processa mensagem de voz do usuário e gera resposta."""
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
        generated_image_path = ai_response_data.get('generated_image_path') # NOVO: Suporte a imagem

        # 4. Criar mensagem IA
        # Se tiver imagem gerada, a mensagem deve ser configurada com ela
        ai_message = ChatMessage(
            chat=chat,
            role=ChatMessage.Role.ASSISTANT,
            content=ai_text,
            suggestion1=ai_suggestions[0] if len(ai_suggestions) > 0 else None,
            suggestion2=ai_suggestions[1] if len(ai_suggestions) > 1 else None,
            duration=duration_ms
        )

        if generated_image_path:
             # Se for imagem gerada
             # O caminho gerado é relativo (ex: chat_attachments/xyz.png)
             # Precisamos anexar isso ao FieldFile do Django.
             # Como o arquivo já está no disco (media/chat_attachments/...), 
             # podemos atribuir diretamente o nome relativo ao campo FileField.
             ai_message.attachment.name = generated_image_path
             ai_message.attachment_type = 'image'
             ai_message.original_filename = "generated_image.png"

        elif audio_path and os.path.exists(audio_path):
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