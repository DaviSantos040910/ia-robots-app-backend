import logging
from chat.models import ChatMessage
from chat.file_processor import FileProcessor
from chat.services.token_service import TokenService

logger = logging.getLogger(__name__)

class SourceAssemblyService:
    # Safety limit for context (Gemini 1.5 Pro is 1M+, Flash is 1M. We set a safe margin)
    MAX_CONTEXT_TOKENS = 800_000

    @staticmethod
    def get_context_from_config(chat_id: int, config: dict) -> str:
        """
        Reúne o conteúdo bruto dos arquivos selecionados.

        Args:
            chat_id: ID do chat.
            config: Dicionário contendo:
                - selectedSourceIds (list[int]): IDs das mensagens contendo os arquivos.

        Returns:
            str: O contexto montado pronto para o LLM.
        """
        context_parts = []
        current_tokens = 0

        # 1. Processar Arquivos Selecionados
        source_ids = config.get('selectedSourceIds', [])
        if source_ids:
            messages = ChatMessage.objects.filter(id__in=source_ids, chat_id=chat_id)

            for message in messages:
                # Check if we already hit the limit
                if current_tokens >= SourceAssemblyService.MAX_CONTEXT_TOKENS:
                    logger.warning(f"Context limit reached ({SourceAssemblyService.MAX_CONTEXT_TOKENS}). Skipping remaining files.")
                    context_parts.append("\n[SYSTEM: Maximum context size reached. Remaining files omitted.]")
                    break

                if message.attachment:
                    try:
                        file_path = message.attachment.path
                        file_name = message.original_filename or f"file_{message.id}"

                        # Cache Read-Through
                        content = message.extracted_text

                        if not content:
                            logger.info(f"Extracting text for message {message.id} (File: {file_name})")
                            # Extrai conteúdo usando o processador
                            content = FileProcessor.extract_text(file_path)

                            # Save to cache if successful
                            if content:
                                message.extracted_text = content
                                message.save(update_fields=['extracted_text'])
                        else:
                            logger.info(f"Using cached text for message {message.id}")

                        if content:
                            # Estimate tokens for this file
                            file_tokens = TokenService.estimate_tokens(content)

                            # Check if adding this file exceeds limit
                            if current_tokens + file_tokens > SourceAssemblyService.MAX_CONTEXT_TOKENS:
                                # Calculate how much space is left
                                space_left = SourceAssemblyService.MAX_CONTEXT_TOKENS - current_tokens
                                if space_left > 100: # Only include if reasonable chunk remains
                                    truncated_content = TokenService.truncate_to_token_limit(content, space_left)
                                    context_parts.append(f"\n--- FILE: {file_name} (Partial) ---\n{truncated_content}")
                                    current_tokens += space_left
                                else:
                                    logger.warning(f"Skipping file {file_name} due to full context.")
                            else:
                                context_parts.append(f"\n--- FILE: {file_name} ---\n{content}")
                                current_tokens += file_tokens
                        else:
                             logger.warning(f"Conteúdo vazio para arquivo {file_name} (ID: {message.id})")

                    except Exception as e:
                        logger.error(f"Erro ao processar arquivo da mensagem {message.id}: {e}")
                        context_parts.append(f"\n--- FILE: {message.original_filename} ---\n[Erro ao ler arquivo]")

        return "\n".join(context_parts)
