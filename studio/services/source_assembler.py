import logging
from chat.models import ChatMessage
from chat.file_processor import FileProcessor
from chat.services.context_builder import build_conversation_history

logger = logging.getLogger(__name__)

class SourceAssemblyService:
    @staticmethod
    def get_context_from_config(chat_id: int, config: dict) -> str:
        """
        Reúne o conteúdo bruto dos arquivos selecionados e opcionalmente o histórico do chat.

        Args:
            chat_id: ID do chat.
            config: Dicionário contendo:
                - selectedSourceIds (list[int]): IDs das mensagens contendo os arquivos.
                - includeChatContext (bool): Se deve incluir o histórico recente do chat.

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

        # 2. Processar Contexto do Chat (Opcional)
        if config.get('includeChatContext', False):
            try:
                # Reutiliza a função existente para buscar histórico
                history, _ = build_conversation_history(chat_id, limit=20)

                chat_text = "\n--- CHAT HISTORY ---\n"
                for turn in history:
                    role = turn.get('role', 'user').upper()
                    # O formato de parts varia (str ou list), precisamos extrair o texto
                    parts = turn.get('parts', [])
                    text = ""
                    if isinstance(parts, str):
                        text = parts
                    elif isinstance(parts, list):
                        # Pega o primeiro elemento que tenha texto
                        for p in parts:
                            if isinstance(p, dict) and 'text' in p:
                                text += p['text'] + " "
                            elif isinstance(p, str):
                                text += p + " "

                    if text.strip():
                        chat_text += f"{role}: {text.strip()}\n"

                context_parts.append(chat_text)

            except Exception as e:
                logger.error(f"Erro ao recuperar histórico do chat {chat_id}: {e}")

        return "\n".join(context_parts)
