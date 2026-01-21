import logging
from chat.models import ChatMessage
from chat.file_processor import FileProcessor

logger = logging.getLogger(__name__)

class SourceAssemblyService:
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

        # 1. Processar Arquivos Selecionados
        source_ids = config.get('selectedSourceIds', [])
        if source_ids:
            messages = ChatMessage.objects.filter(id__in=source_ids, chat_id=chat_id)

            for message in messages:
                if message.attachment:
                    try:
                        file_path = message.attachment.path
                        file_name = message.original_filename or f"file_{message.id}"

                        # Extrai conteúdo usando o processador da Fase 1
                        content = FileProcessor.extract_text(file_path)

                        if content:
                            context_parts.append(f"\n--- FILE: {file_name} ---\n{content}")
                        else:
                             logger.warning(f"Conteúdo vazio para arquivo {file_name} (ID: {message.id})")

                    except Exception as e:
                        logger.error(f"Erro ao processar arquivo da mensagem {message.id}: {e}")
                        context_parts.append(f"\n--- FILE: {message.original_filename} ---\n[Erro ao ler arquivo]")

        return "\n".join(context_parts)
