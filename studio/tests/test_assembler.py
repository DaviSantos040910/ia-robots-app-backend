from django.test import TestCase
from unittest.mock import patch, MagicMock
from chat.models import Chat, ChatMessage
from django.contrib.auth import get_user_model
from bots.models import Bot
from studio.services.source_assembler import SourceAssemblyService
import json

User = get_user_model()

class SourceAssemblerTest(TestCase):
    def setUp(self):
        # Setup básico de usuário, bot e chat
        self.user = User.objects.create(username="testuser")
        # Bot tem 'owner' em vez de 'user'
        self.bot = Bot.objects.create(name="TestBot", owner=self.user)
        self.chat = Chat.objects.create(user=self.user, bot=self.bot)

    @patch('chat.file_processor.FileProcessor.extract_text')
    def test_get_context_caching(self, mock_extract):
        """Testa o cache read-through do SourceAssemblyService."""

        msg1 = ChatMessage.objects.create(
            chat=self.chat,
            role='user',
            content="File 1",
            attachment="path/to/file1.pdf",
            original_filename="file1.pdf",
            extracted_text=None # Initially empty
        )

        mock_extract.return_value = "Texto Extraído"

        config = {"selectedSourceIds": [msg1.id]}

        # 1. Primeira chamada: Deve extrair e salvar
        result1 = SourceAssemblyService.get_context_from_config(self.chat.id, config)

        self.assertIn("Texto Extraído", result1)
        mock_extract.assert_called_once()

        # Verifica se salvou no banco
        msg1.refresh_from_db()
        self.assertEqual(msg1.extracted_text, "Texto Extraído")

        # 2. Segunda chamada: Deve usar o cache (não chamar extrator)
        mock_extract.reset_mock()
        result2 = SourceAssemblyService.get_context_from_config(self.chat.id, config)

        self.assertIn("Texto Extraído", result2)
        mock_extract.assert_not_called()

    @patch('chat.file_processor.FileProcessor.extract_text')
    def test_get_context_files_only(self, mock_extract):
        """Testa a montagem de contexto apenas com arquivos."""

        # Cria mensagens com anexos dummy
        msg1 = ChatMessage.objects.create(
            chat=self.chat,
            role='user',
            content="File 1",
            attachment="path/to/file1.pdf",
            original_filename="file1.pdf"
        )
        msg2 = ChatMessage.objects.create(
            chat=self.chat,
            role='user',
            content="File 2",
            attachment="path/to/file2.docx",
            original_filename="file2.docx"
        )

        # Mock do retorno do extrator
        def side_effect(path, mime_type=None):
            if "file1" in path: return "Conteúdo do PDF 1"
            if "file2" in path: return "Conteúdo do DOCX 2"
            return ""
        mock_extract.side_effect = side_effect

        config = {
            "selectedSourceIds": [msg1.id, msg2.id],
            "includeChatContext": False
        }

        result = SourceAssemblyService.get_context_from_config(self.chat.id, config)

        self.assertIn("--- FILE: file1.pdf ---", result)
        self.assertIn("Conteúdo do PDF 1", result)
        self.assertIn("--- FILE: file2.docx ---", result)
        self.assertIn("Conteúdo do DOCX 2", result)

    @patch('chat.file_processor.FileProcessor.extract_text')
    def test_get_context_no_chat_history(self, mock_extract):
        """Testa que histórico de chat NÃO é incluído, mesmo se solicitado (feature removida)."""
        msg1 = ChatMessage.objects.create(
            chat=self.chat,
            role='user',
            attachment="dummy.txt",
            original_filename="dummy.txt"
        )
        mock_extract.return_value = "Texto do Arquivo"

        config = {
            "selectedSourceIds": [msg1.id],
            "includeChatContext": True # Deve ser ignorado
        }

        result = SourceAssemblyService.get_context_from_config(self.chat.id, config)

        self.assertIn("--- FILE: dummy.txt ---", result)
        self.assertIn("Texto do Arquivo", result)
        self.assertNotIn("--- CHAT HISTORY ---", result)
