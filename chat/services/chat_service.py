# chat/services/chat_service.py
"""
Serviço principal de chat com IA.
Orquestra busca de contexto multi-doc, geração de resposta e salvamento de memória.
Atualizado para conectar a flag allow_web_search do Bot ao fluxo de Prompt e Tools.
"""

import os
import json
import re
import mimetypes
import threading
import logging
import tempfile
import uuid

import time

from datetime import datetime
from django.utils import timezone
from django.db import transaction
from django.core.files import File

from google.genai import types

from ..models import ChatMessage, Chat, ChatResponseMetric
from ..vector_service import VectorService
from .ai_client import get_ai_client, detect_intent, generate_content_stream
from .image_service import ImageGenerationService
from .context_builder import (
    build_conversation_history,
    build_system_instruction,
    get_recent_attachment_context
)
from .memory_service import process_memory_background
from .tts_service import generate_tts_audio
from .transcription_service import transcribe_audio_gemini

logger = logging.getLogger(__name__)

# Instância global do serviço vetorial
vector_service = VectorService()
# Instância global do serviço de imagem
image_service = ImageGenerationService()


def _detect_lang(text: str) -> str:
    """
    Detecta idioma (pt, es, en) baseado em heurística simples.
    """
    text = text.lower()

    # Spanish heuristics
    es_markers = ["¿", "¡", "qué", "cómo", "por qué", "dónde", "fuente", "fuentes"]
    if any(m in text for m in es_markers):
        return "es"

    # Portuguese heuristics
    pt_markers = ["você", "não", "por que", "onde", "fonte", "fontes", "documento", "documentos", "tutor", "quais", "quem", "qual"]
    # Check whole words for some markers to avoid false positives (e.g. 'none' containing 'on')
    # Simple check is usually enough given the distinct markers
    if any(m in text for m in pt_markers):
        return "pt"

    # Default to English
    return "en"

def _safe_excerpt(text: str, max_len: int = 120) -> str:
    """
    Gera um trecho seguro do texto (sem quebras de linha, truncado).
    """
    if not text:
        return ""

    # Remove newlines and extra spaces
    cleaned = " ".join(text.split())

    if len(cleaned) > max_len:
        return cleaned[:max_len] + "…"
    return cleaned

def _build_strict_refusal(bot_name: str, question: str, lang: str = None, has_any_sources: bool = False) -> str:
    """
    Constrói a mensagem de recusa strict determinística.
    """
    if not lang:
        lang = _detect_lang(question)

    q_excerpt = _safe_excerpt(question)

    prefix = f"{bot_name}: " if bot_name else ""

    # Templates
    templates = {
        "pt": {
            True: "{prefix}Não encontrei essa informação nas suas fontes para responder em modo restrito.\n\nPergunta: “{q}”\n\nPara eu ajudar com base nas fontes, você pode:\n- adicionar uma fonte relevante,\n- indicar onde isso aparece (arquivo/página/trecho),\n- ou reformular a pergunta usando termos presentes nos documentos.",
            False: "{prefix}No modo restrito, eu só posso responder usando fontes.\n\nPergunta: “{q}”\n\nPara eu ajudar, adicione uma fonte (PDF, imagem, link, etc.) e tente novamente."
        },
        "en": {
            True: "{prefix}I couldn’t find this information in your sources to answer in strict mode.\n\nQuestion: “{q}”\n\nTo help based on your sources, you can:\n- add a relevant source,\n- point to where this appears (file/page/section),\n- or rephrase using terms present in the documents.",
            False: "{prefix}In strict mode, I can only answer using sources.\n\nQuestion: “{q}”\n\nTo help, add a source (PDF, image, link, etc.) and try again."
        },
        "es": {
            True: "{prefix}No encontré esta información en tus fuentes para responder en modo estricto.\n\nPregunta: “{q}”\n\nPara ayudar basándome en tus fuentes, puedes:\n- agregar una fuente relevante,\n- indicar dónde aparece (archivo/página/sección),\n- o reformular usando términos presentes en los documentos.",
            False: "{prefix}En modo estricto, solo puedo responder usando fuentes.\n\nPregunta: “{q}”\n\nPara ayudar, agrega una fuente (PDF, imagen, enlace, etc.) e inténtalo de nuevo."
        }
    }

    template = templates.get(lang, templates["en"])[has_any_sources]
    return template.format(prefix=prefix, q=q_excerpt)


def _calculate_metrics(response_text: str, context_sources: list) -> dict:
    """Calcula métricas de cobertura de fontes na resposta."""
    if not response_text:
        return {'sources_count': len(context_sources), 'cited_count': 0, 'has_citation': False}

    cited_count = 0
    has_citation = False

    # 1. Detectar nomes de arquivos presentes no texto
    # (Simplificado: busca exata ou parcial do nome)
    unique_sources = set(context_sources)
    for src in unique_sources:
        # Remove extensão para flexibilidade (ex: "relatorio.pdf" -> "relatorio")
        base_name = os.path.splitext(src)[0]
        if src in response_text or base_name in response_text:
            cited_count += 1
            has_citation = True

    # 2. Detectar padrões explícitos de citação se nenhum nome encontrado
    if not has_citation:
        # Strict validation: Only accept [n] format.
        # Legacy patterns like [Fonte n] or text references are NOT valid citations for metrics.
        if re.search(r'\[\d+\]', response_text):
            has_citation = True

    return {
        'sources_count': len(unique_sources),
        'cited_count': cited_count,
        'has_citation': has_citation
    }

def _save_metrics(message: ChatMessage, metrics: dict):
    """Salva as métricas no banco de dados."""
    try:
        ChatResponseMetric.objects.create(
            message=message,
            sources_count=metrics['sources_count'],
            cited_count=metrics['cited_count'],
            has_citation=metrics['has_citation']
        )
    except Exception as e:
        logger.error(f"Erro ao salvar métricas: {e}")

def _parse_ai_response(response_text: str) -> dict:
    """
    Faz parse da resposta da IA para endpoints NÃO-STREAMING.
    Para streaming, a lógica agora está embutida em process_message_stream.
    """
    result = {'content': "", 'suggestions': [], 'audio_path': None, 'duration_ms': 0}
    if not response_text:
        result['content'] = "Desculpe, não consegui gerar uma resposta."
        return result

    text = response_text.strip()

    if "|||SUGGESTIONS|||" in text:
        parts = text.split("|||SUGGESTIONS|||")
        result['content'] = parts[0].strip()
        try:
            json_text = parts[1].strip()
            json_text = re.sub(r'^```\w*', '', json_text, flags=re.MULTILINE)
            json_text = re.sub(r'\s*```$', '', json_text, flags=re.MULTILINE).strip()
            suggestions = json.loads(json_text)
            if isinstance(suggestions, list):
                result['suggestions'] = [str(s) for s in suggestions][:3]
        except Exception as e:
            logger.warning(f"Erro ao parsear sugestões JSON: {e}")

    elif text.startswith('{') or text.startswith('```json'):
        try:
            cleaned = re.sub(r'^```\w*', '', text, flags=re.MULTILINE)
            cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
            data = json.loads(cleaned)
            if isinstance(data, dict):
                result['content'] = data.get('response', data.get('content', ''))
                result['suggestions'] = data.get('suggestions', [])[:3]
                if result['content']: return result
        except: pass
        if not result['content']: result['content'] = text

    elif "---SUGESTÕES---" in text or "---SUGGESTIONS---" in text:
        sep = "---SUGESTÕES---" if "---SUGESTÕES---" in text else "---SUGGESTIONS---"
        parts = text.split(sep)
        result['content'] = parts[0].strip()
        if len(parts) > 1:
            sugs = re.findall(r'(?:^|\n)\s*(?:\d+\.|-)\s*(.+)', parts[1])
            result['suggestions'] = [s.strip() for s in sugs[:3] if s.strip()]
    else:
        result['content'] = text

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
        text_content = response.text if response.text else "[]"
        cleaned_response = re.sub(r'^```\w*', '', text_content, flags=re.MULTILINE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response, flags=re.MULTILINE).strip()
        suggestions = json.loads(cleaned_response)
        if (isinstance(suggestions, list) and len(suggestions) > 0):
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
    """Obtém resposta da IA (Modo Síncrono/Não-Stream)."""
    try:
        # DETECÇÃO DE INTENÇÃO (IMAGEM vs TEXTO)
        intent = 'TEXT'
        if not (user_message_obj and user_message_obj.attachment):
            try:
                intent = detect_intent(user_message_text)
            except Exception as e:
                intent = 'TEXT'

        if intent == 'IMAGE':
            try:
                image_rel_path = image_service.generate_and_save_image(user_message_text)
                return {
                    'content': f"Aqui está a imagem que criei para você com base em \"{user_message_text}\".",
                    'suggestions': ["Gere outra variação", "Mude o estilo", "Obrigado!"],
                    'audio_path': None,
                    'duration_ms': 0,
                    'generated_image_path': image_rel_path
                }
            except Exception as img_err:
                return {
                    'content': f"Erro ao gerar imagem: {str(img_err)}.",
                    'suggestions': [],
                    'audio_path': None
                }

        # FLUXO DE TEXTO
        client = get_ai_client()
        chat = Chat.objects.select_related('bot', 'user').get(id=chat_id)
        bot = chat.bot

        # --- Recupera flag de Web Search e Strict Context ---
        allow_web_search = getattr(bot, 'allow_web_search', False)
        strict_context = getattr(bot, 'strict_context', False)

        user_defined_prompt = bot.prompt.strip() if bot.prompt else "Você é um assistente útil."
        user_name = chat.user.first_name if chat.user.first_name else "Usuário"
        current_time_str = datetime.now().strftime('%d/%m/%Y %H:%M')

        exclude_id = user_message_obj.id if user_message_obj else None
        gemini_history, _ = build_conversation_history(chat_id, limit=12, exclude_message_id=exclude_id)

        # Obter IDs dos espaços de estudo vinculados
        study_space_ids = list(bot.study_spaces.values_list('id', flat=True))

        doc_contexts, memory_contexts, available_doc_names = _get_smart_context(
            query=user_message_text,
            user_id=chat.user_id,
            bot_id=bot.id,
            chat_id=chat_id,
            study_space_ids=study_space_ids
        )

        # Observability Log
        logger.info(f"[Context] Chat {chat_id} | Bot {bot.id} | Strict: {strict_context} | Web: {allow_web_search}")
        logger.info(f"[Context] Available Docs: {available_doc_names}")
        
        # --- Format Contexts with Citations ---
        formatted_doc_contexts = []
        source_map = {} # source_id -> {index: 1, title: 'Title'}
        used_source_indices = []

        if doc_contexts:
            for chunk in doc_contexts:
                # chunk is now a Dict: {content, source, source_id, ...}
                s_id = chunk.get('source_id') or chunk.get('source') # Fallback to title if ID missing
                s_title = chunk.get('source', 'Documento')
                
                if s_id not in source_map:
                    source_map[s_id] = {'index': len(source_map) + 1, 'title': s_title}
                
                s_idx = source_map[s_id]['index']
                used_source_indices.append(s_idx)
                
                # Format: [1] Title\nContent
                formatted_doc_contexts.append(f"[{s_idx}] {s_title}\n{chunk['content']}")

            logger.info(f"[Context] Sources Mapped: {source_map}")

        # --- Strict Mode Fallback Logic (NotebookLM Style) ---
        if strict_context and not doc_contexts:
            if available_doc_names:
                logger.info("[Sync] Strict Mode + No Context Found -> Generating Refusal Template")
                refusal_text = _build_strict_refusal(bot.name, user_message_text, has_any_sources=True)
                return _parse_ai_response(refusal_text)
            else:
                logger.info("[Sync] Strict Mode + No Docs at All -> Refusal")
                refusal_text = _build_strict_refusal(bot.name, user_message_text, has_any_sources=False)
                return _parse_ai_response(refusal_text)

        # --- Mixed Mode Fallback (Sync) ---
        elif not strict_context and not doc_contexts and allow_web_search:
             logger.info("[Sync] Mixed Mode + No Context -> Forcing Two-Block Answer")
             mixed_prompt = (
                 f"User Question: '{user_message_text}'\n\n"
                 f"Your Personality: '{user_defined_prompt}'\n"
                 "CONTEXT CHECK: You searched the user's documents but found NO matches.\n"
                 "INSTRUCTION: You must answer using general knowledge/web search, but you MUST format it in two distinct blocks.\n\n"
                 "TEMPLATE:\n"
                 f"Nas suas fontes, não encontrei informações sobre {user_message_text}.\n\n"
                 "Fora do contexto dos documentos, de forma geral:\n"
                 "<Insert your helpful answer here based on general knowledge or web search>"
             )
             
             system_instruction = build_system_instruction(
                bot_prompt=user_defined_prompt,
                user_name=user_name,
                doc_contexts=[],
                memory_contexts=memory_contexts,
                current_time=current_time_str,
                available_docs=available_doc_names,
                allow_web_search=True,
                strict_context=False
            )
             
             generation_config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2500,
                system_instruction=system_instruction,
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
             
             # Override input
             input_parts = [{"text": mixed_prompt}]

        else:
            # --- Standard Flow ---
            system_instruction = build_system_instruction(
                bot_prompt=user_defined_prompt,
                user_name=user_name,
            doc_contexts=formatted_doc_contexts,
                memory_contexts=memory_contexts,
                current_time=current_time_str,
                available_docs=available_doc_names,
                allow_web_search=allow_web_search, # Passa a flag para o construtor de prompt
                strict_context=strict_context
            )

            # Adjust temperature based on RAG context presence
            temperature = 0.3 if formatted_doc_contexts else 0.7

            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2500,
                system_instruction=system_instruction
            )

            # Adiciona ferramenta Google Search na configuração síncrona (Apenas se Strict Context estiver OFF)
            if allow_web_search and not strict_context:
                # Se config.tools já existe, adiciona. Se não, cria.
                if hasattr(generation_config, 'tools') and generation_config.tools:
                    generation_config.tools.append(types.Tool(google_search=types.GoogleSearch()))
                else:
                    generation_config.tools = [types.Tool(google_search=types.GoogleSearch())]

            input_parts = []
        if user_message_obj and user_message_obj.attachment:
            try:
                if hasattr(user_message_obj.attachment, 'path') and user_message_obj.attachment.path:
                    file_path = user_message_obj.attachment.path
                    mime_type, _ = mimetypes.guess_type(user_message_obj.original_filename or "file")
                    if not mime_type and user_message_obj.attachment_type == 'image': mime_type = 'image/jpeg'
                    if mime_type and (mime_type.startswith('image/') or mime_type == 'application/pdf'):
                        with open(file_path, 'rb') as f:
                            input_parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type))
            except Exception: pass

        final_user_prompt = f"""{user_message_text}\n\n---\nSe possível, forneça sugestões de continuação usando o formato |||SUGGESTIONS||| definido no system prompt."""
        input_parts.append({"text": final_user_prompt})
        contents = gemini_history + [{"role": "user", "parts": input_parts}]

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=generation_config
        )

        result_data = _parse_ai_response(response.text if response.text else "")

        # --- POST-GENERATION GUARDRAIL (STRICT MODE) ---
        # If strict_context is ON, but response has NO citations, assume hallucination/failure.
        if strict_context and result_data['content']:
            has_citation = bool(re.search(r'\[\d+\]', result_data['content']))
            if not has_citation:
                logger.warning(f"[Guardrail] Chat {chat_id}: Strict Mode enabled but NO citations found. Triggering refusal.")
                refusal_text = _build_strict_refusal(bot.name, user_message_text, has_any_sources=bool(available_doc_names))
                result_data = _parse_ai_response(refusal_text)
                # Clear citations legend logic triggers below since content changed
                source_map = {} 

        # Build Sources list for frontend
        sources_list = []
        if source_map:
            # Extract citations actually used in the FINAL text
            used_indices = set(re.findall(r'\[(\d+)\]', result_data['content']))
            
            # Map back to source details
            unique_sources = {}
            for s_id, s_info in source_map.items():
                if str(s_info['index']) in used_indices:
                    if s_id not in unique_sources:
                        unique_sources[s_id] = {
                            'id': s_id,
                            'title': s_info['title'],
                            'type': 'file', # Default, could be refined if source_map had type
                            'index': s_info['index']
                        }
            
            # Safely create the list
            try:
                sources_list = sorted(unique_sources.values(), key=lambda x: x['index'])
            except Exception as e:
                logger.error(f"[Sources Error] Failed to process sources list: {e}")
                sources_list = []

            result_data['sources'] = sources_list

        # Metrics Logic
        metrics = _calculate_metrics(result_data['content'], available_doc_names)
        logger.info(f"[Metrics] Msg Response: {metrics}")

        # Save metrics requires a Message object.
        # Since get_ai_response returns dict (and caller creates message later or earlier?),
        # we can't easily link to message ID here unless passed.
        # But handle_voice_message DOES create AI message.
        # Wait, get_ai_response is usually called by a view which then saves the message.
        # Ideally, we should return metrics in the result_data so the caller can save them.

        result_data['metrics'] = metrics # Pass metrics up

        if result_data['content'] and len(user_message_text) > 10:
            threading.Thread(
                target=process_memory_background,
                args=(chat.user_id, bot.id, user_message_text, result_data['content'])
            ).start()

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


def process_message_stream(user_id: int, chat_id: int, user_message_text: str):
    """
    Generator que processa a mensagem e envia chunks via SSE.
    """
    SEPARATOR = '|||SUGGESTIONS|||'
    SEPARATOR_LEN = len(SEPARATOR)
    CHUNK_DELAY = 0.03

    chat = Chat.objects.select_related('bot', 'user').get(id=chat_id, user_id=user_id)
    bot = chat.bot

    allow_web_search = getattr(bot, 'allow_web_search', False)
    strict_context = getattr(bot, 'strict_context', False)

    # 1. Yield Start
    yield f"data: {json.dumps({'type': 'start', 'status': 'processing'})}\n\n"

    try:
        # --- Context Retrieval ---
        user_defined_prompt = bot.prompt.strip() if bot.prompt else "Você é um assistente útil."
        user_name = chat.user.first_name if chat.user.first_name else "Usuário"
        current_time_str = datetime.now().strftime('%d/%m/%Y %H:%M')

        gemini_history, _ = build_conversation_history(chat_id, limit=10)
        study_space_ids = list(bot.study_spaces.values_list('id', flat=True))

        doc_contexts, memory_contexts, available_docs = _get_smart_context(
            query=user_message_text,
            user_id=chat.user_id,
            bot_id=bot.id,
            chat_id=chat_id,
            study_space_ids=study_space_ids
        )

        logger.info(f"[Stream] Context: {len(doc_contexts)} chunks, Strict: {strict_context}")

        # Format Contexts
        formatted_doc_contexts = []
        source_map = {}
        if doc_contexts:
            for chunk in doc_contexts:
                s_id = chunk.get('source_id') or chunk.get('source')
                s_title = chunk.get('source', 'Documento')
                if s_id not in source_map:
                    source_map[s_id] = {'index': len(source_map) + 1, 'title': s_title}
                s_idx = source_map[s_id]['index']
                formatted_doc_contexts.append(f"[{s_idx}] {s_title}\n{chunk['content']}")

        # --- DECISION BLOCK ---
        
        # CASE 1: Strict ON + NO Context
        if strict_context and not doc_contexts:
            logger.info("[Stream] Strict Mode + No Context -> Immediate Refusal")

            # Use deterministic strict refusal
            refusal_text = _build_strict_refusal(bot.name, user_message_text, has_any_sources=bool(available_docs))

            # Create final message immediately (No stream)
            ai_message = ChatMessage.objects.create(
                chat=chat,
                role=ChatMessage.Role.ASSISTANT,
                content=refusal_text,
                sources=[]
            )
            chat.last_message_at = timezone.now()
            chat.save()

            # Send End Event
            end_payload = {
                'type': 'end',
                'message_id': ai_message.id,
                'clean_content': refusal_text,
                'suggestions': [],
                'sources': []
            }
            yield f"data: {json.dumps(end_payload)}\n\n"
            return

        # CASE 2: Strict ON + Context (Sync Generation + Pseudo-Stream)
        elif strict_context and doc_contexts:
            logger.info("[Stream] Strict Mode + Context -> Sync Generation + Citation Check")

            system_instruction = build_system_instruction(
                bot_prompt=user_defined_prompt,
                user_name=user_name,
                doc_contexts=formatted_doc_contexts,
                memory_contexts=memory_contexts,
                current_time=current_time_str,
                available_docs=available_docs,
                allow_web_search=False, # Web Search disabled in Strict Mode always
                strict_context=True
            )
            
            config = types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=3000,
                system_instruction=system_instruction
            )
            
            prompt_text = f"""{user_message_text}\n\n---\nSe possível, forneça sugestões de continuação usando o formato |||SUGGESTIONS||| definido no system prompt."""
            contents = gemini_history + [{"role": "user", "parts": [{"text": prompt_text}]}]

            # SYNC CALL
            client = get_ai_client()
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=config
            )

            full_text = response.text if response.text else ""

            # VALIDATE CITATION
            has_citation = bool(re.search(r'\[\d+\]', full_text))

            if not has_citation:
                logger.warning("[Stream] Strict Mode Guardrail: No citation found. Generating refusal.")
                full_text = _build_strict_refusal(bot.name, user_message_text, has_any_sources=True)
                source_map = {} # Clear sources

            # Parse Suggestions (Sync)
            final_suggestions = []
            clean_content = full_text
            if SEPARATOR in full_text:
                parts = full_text.split(SEPARATOR)
                clean_content = parts[0].strip()
                try:
                    s_json = "".join(parts[1:])
                    s_json = re.sub(r'^```\w*', '', s_json, flags=re.MULTILINE)
                    s_json = re.sub(r'\s*```$', '', s_json, flags=re.MULTILINE).strip()
                    parsed = json.loads(s_json)
                    if isinstance(parsed, list):
                        final_suggestions = [str(s) for s in parsed][:3]
                except: pass

            # PSEUDO-STREAM CHUNKS
            chunk_size = 300
            for i in range(0, len(clean_content), chunk_size):
                chunk = clean_content[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                time.sleep(CHUNK_DELAY)

            # Extract Sources
            final_sources_list = []
            if source_map:
                used_indices = set(re.findall(r'\[(\d+)\]', clean_content))
                unique_sources = {}
                for s_id, s_info in source_map.items():
                    if str(s_info['index']) in used_indices:
                        if s_id not in unique_sources:
                            unique_sources[s_id] = {
                                'id': s_id,
                                'title': s_info['title'],
                                'type': 'file',
                                'index': s_info['index']
                            }
                # Safe processing of sources list
                try:
                    final_sources_list = sorted(unique_sources.values(), key=lambda x: x['index'])
                except Exception as e:
                    logger.error(f"[Stream Sources Error] Failed to process sources list: {e}")
                    final_sources_list = []

            # Save Message
            ai_message = ChatMessage.objects.create(
                chat=chat,
                role=ChatMessage.Role.ASSISTANT,
                content=clean_content,
                suggestion1=final_suggestions[0] if len(final_suggestions) > 0 else None,
                suggestion2=final_suggestions[1] if len(final_suggestions) > 1 else None,
                sources=final_sources_list
            )
            chat.last_message_at = timezone.now()
            chat.save()

            # Metrics
            metrics = _calculate_metrics(clean_content, available_docs)
            _save_metrics(ai_message, metrics)

            # End Event
            end_payload = {
                'type': 'end',
                'message_id': ai_message.id,
                'clean_content': clean_content,
                'suggestions': final_suggestions,
                'sources': final_sources_list
            }
            yield f"data: {json.dumps(end_payload)}\n\n"

            # Background Memory
            if len(clean_content) > 10:
                threading.Thread(
                    target=process_memory_background,
                    args=(chat.user_id, bot.id, user_message_text, clean_content)
                ).start()
            return

        # CASE 3: Strict OFF (Normal / Mixed) -> Real Stream
        else:
            logger.info("[Stream] Strict OFF -> Real Stream")

            # Handle Mixed Mode Prompt Logic
            if not doc_contexts and allow_web_search:
                 mixed_prompt = (
                     f"User Question: '{user_message_text}'\n\n"
                     f"Your Personality: '{user_defined_prompt}'\n"
                     "CONTEXT CHECK: You searched the user's documents but found NO matches.\n"
                     "INSTRUCTION: You must answer using general knowledge/web search, but you MUST format it in two distinct blocks.\n\n"
                     "TEMPLATE:\n"
                     f"Nas suas fontes, não encontrei informações sobre {user_message_text}.\n\n"
                     "Fora do contexto dos documentos, de forma geral:\n"
                     "<Insert your helpful answer here based on general knowledge or web search>"
                 )
                 prompt_text = mixed_prompt
                 # Pass empty doc_contexts to builder but allow web search
                 system_instruction = build_system_instruction(
                    bot_prompt=user_defined_prompt,
                    user_name=user_name,
                    doc_contexts=[],
                    memory_contexts=memory_contexts,
                    current_time=current_time_str,
                    available_docs=available_docs,
                    allow_web_search=True,
                    strict_context=False
                )
            else:
                 prompt_text = f"""{user_message_text}\n\n---\nSe possível, forneça sugestões de continuação usando o formato |||SUGGESTIONS||| definido no system prompt."""
                 system_instruction = build_system_instruction(
                    bot_prompt=user_defined_prompt,
                    user_name=user_name,
                    doc_contexts=formatted_doc_contexts,
                    memory_contexts=memory_contexts,
                    current_time=current_time_str,
                    available_docs=available_docs,
                    allow_web_search=allow_web_search,
                    strict_context=strict_context
                )

            contents = gemini_history + [{"role": "user", "parts": [{"text": prompt_text}]}]

            config = types.GenerateContentConfig(
                temperature=0.3 if doc_contexts else 0.7,
                max_output_tokens=3000,
                system_instruction=system_instruction
            )

            # Unified Tool Rule
            use_search = allow_web_search and not strict_context
            if use_search:
                 config.tools = [types.Tool(google_search=types.GoogleSearch())]

            # STREAM CALL
            stream = generate_content_stream(contents, config, use_google_search=use_search)

            buffer = ""
            full_clean_content = ""
            suggestions_json_str = ""
            is_collecting_suggestions = False

            for text_chunk in stream:
                if not isinstance(text_chunk, str) or not text_chunk: continue

                buffer += text_chunk

                # Check for suggestions separator
                # "Só tratar como sugestões se: estiver nas últimas 800–1200 chars OU após \n\n---\n"

                if not is_collecting_suggestions and SEPARATOR in buffer:
                    parts = buffer.split(SEPARATOR)
                    text_part = parts[0]
                    suggestion_part = "".join(parts[1:])

                    total_so_far = full_clean_content + text_part

                    # Logic: We trust the unique separator token.
                    # Checking for \n\n---\n is risky if model doesn't echo it.
                    is_valid_position = True

                    if is_valid_position:
                         # Valid Separator
                        if text_part:
                            full_clean_content += text_part
                            yield f"data: {json.dumps({'type': 'chunk', 'text': text_part})}\n\n"
                            time.sleep(CHUNK_DELAY)

                        is_collecting_suggestions = True
                        suggestions_json_str = suggestion_part
                        buffer = ""

                    else:
                        # Invalid Separator (Middle of text without proper context) -> Treat as text
                        # Don't switch mode, just flush buffer partially or normally
                        # Since `buffer` contains `SEPARATOR`, we can't just keep it in buffer forever.
                        # We should flush what we have as text.
                        # However, we must be careful not to double flush if SEPARATOR is partial.
                        # But `SEPARATOR in buffer` means we have the full separator.
                        # So we flush the whole buffer as text.
                        full_clean_content += buffer
                        yield f"data: {json.dumps({'type': 'chunk', 'text': buffer})}\n\n"
                        time.sleep(CHUNK_DELAY)
                        buffer = ""

                elif is_collecting_suggestions:
                    suggestions_json_str += buffer
                    buffer = ""
                else:
                    # Safe buffer flush
                    if len(buffer) > SEPARATOR_LEN:
                        safe_chunk = buffer[:-SEPARATOR_LEN]
                        buffer = buffer[-SEPARATOR_LEN:]
                        full_clean_content += safe_chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'text': safe_chunk})}\n\n"
                        time.sleep(CHUNK_DELAY)

            # Finalize Stream
            if buffer and not is_collecting_suggestions:
                full_clean_content += buffer
                yield f"data: {json.dumps({'type': 'chunk', 'text': buffer})}\n\n"

            # Parse Suggestions
            final_suggestions = []
            if suggestions_json_str:
                try:
                    s_json = re.sub(r'^```\w*', '', suggestions_json_str, flags=re.MULTILINE)
                    s_json = re.sub(r'\s*```$', '', s_json, flags=re.MULTILINE).strip()
                    parsed = json.loads(s_json)
                    if isinstance(parsed, list):
                        final_suggestions = [str(s) for s in parsed][:3]
                except: pass

            # Extract Sources (Standard Flow)
            final_sources_list = []
            if source_map:
                used_indices = set(re.findall(r'\[(\d+)\]', full_clean_content))
                unique_sources = {}
                for s_id, s_info in source_map.items():
                    if str(s_info['index']) in used_indices:
                        if s_id not in unique_sources:
                            unique_sources[s_id] = {
                                'id': s_id,
                                'title': s_info['title'],
                                'type': 'file',
                                'index': s_info['index']
                            }
                # Safe processing of sources list
                try:
                    final_sources_list = sorted(unique_sources.values(), key=lambda x: x['index'])
                except Exception as e:
                    logger.error(f"[Stream Sources Error] Failed to process sources list: {e}")
                    final_sources_list = []

            # Save
            ai_message = ChatMessage.objects.create(
                chat=chat,
                role=ChatMessage.Role.ASSISTANT,
                content=full_clean_content,
                suggestion1=final_suggestions[0] if len(final_suggestions) > 0 else None,
                suggestion2=final_suggestions[1] if len(final_suggestions) > 1 else None,
                sources=final_sources_list
            )
            chat.last_message_at = timezone.now()
            chat.save()

            # Metrics
            metrics = _calculate_metrics(full_clean_content, available_docs)
            _save_metrics(ai_message, metrics)

            # End Event
            end_payload = {
                'type': 'end',
                'message_id': ai_message.id,
                'clean_content': full_clean_content,
                'suggestions': final_suggestions,
                'sources': final_sources_list
            }
            yield f"data: {json.dumps(end_payload)}\n\n"

            # Background Memory
            if len(full_clean_content) > 10:
                threading.Thread(
                    target=process_memory_background,
                    args=(chat.user_id, bot.id, user_message_text, full_clean_content)
                ).start()

    except Exception as e:
        logger.error(f"[Stream Error] {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"


def _get_smart_context(
    query: str,
    user_id: int,
    bot_id: int,
    chat_id: int,
    study_space_ids: list = None,
    allowed_source_ids: list = None
) -> tuple:
    """Busca contexto de forma inteligente usando o VectorService multi-doc."""
    try:
        recent_source = get_recent_attachment_context(chat_id)
        doc_contexts, memory_contexts = vector_service.search_context(
            query_text=query,
            user_id=user_id,
            bot_id=bot_id,
            study_space_ids=study_space_ids,
            allowed_source_ids=allowed_source_ids,
            limit=6,
            recent_doc_source=recent_source
        )
        available_docs = vector_service.get_available_documents(
            user_id,
            bot_id,
            study_space_ids=study_space_ids
        )
        available_names = [d['source'] for d in available_docs]
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

        user_audio_file.seek(0)
        trans_result = transcribe_audio_gemini(user_audio_file)
        transcription = trans_result['transcription'] if trans_result['success'] else "[Áudio - Transcrição indisponível]"
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

        ai_response_data = get_ai_response(
            chat_id,
            transcription,
            user_message_obj=user_message,
            reply_with_audio=reply_with_audio
        )

        ai_text = ai_response_data.get('content', '')
        ai_suggestions = ai_response_data.get('suggestions', [])
        ai_sources = ai_response_data.get('sources', [])
        audio_path = ai_response_data.get('audio_path')
        duration_ms = ai_response_data.get('duration_ms', 0)
        generated_image_path = ai_response_data.get('generated_image_path')

        ai_message = ChatMessage(
            chat=chat,
            role=ChatMessage.Role.ASSISTANT,
            content=ai_text,
            suggestion1=ai_suggestions[0] if len(ai_suggestions) > 0 else None,
            suggestion2=ai_suggestions[1] if len(ai_suggestions) > 1 else None,
            duration=duration_ms,
            sources=ai_sources
        )

        ai_message.save() # Save first to get ID

        # Save metrics if present
        if 'metrics' in ai_response_data:
            _save_metrics(ai_message, ai_response_data['metrics'])

        if generated_image_path:
             ai_message.attachment.name = generated_image_path
             ai_message.attachment_type = 'image'
             ai_message.original_filename = "generated_image.png"

        elif audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, 'rb') as f:
                    filename = f"reply_tts_{uuid.uuid4().hex[:10]}.wav"
                    ai_message.attachment.save(filename, File(f), save=False) # save=False? Attachment needs ID usually or instance?
                    # If instance already saved, save=True updates it.
                    # Django FileField save() saves the file and updates the instance.
                    # Since we called ai_message.save() above, it has an ID.
                    ai_message.attachment_type = 'audio'
                    ai_message.original_filename = "voice_reply.wav"
                    ai_message.save() # Update metadata first? No, attachment.save handles it.
                os.remove(audio_path)
            except Exception as e:
                logger.error(f"[Handle Voice] Erro ao anexar áudio: {e}")
                ai_message.attachment_type = None
                ai_message.save()

        chat.last_message_at = timezone.now()
        chat.save()

        return {"user_message": user_message, "ai_message": ai_message}