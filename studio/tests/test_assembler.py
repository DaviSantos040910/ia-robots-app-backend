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

    @patch('studio.services.source_assembler.build_conversation_history')
    def test_get_context_with_chat_history(self, mock_build_history):
        """Testa a inclusão do histórico do chat."""

        # Mock do histórico
        mock_build_history.return_value = ([
            {"role": "user", "parts": [{"text": "Olá bot"}]},
            {"role": "model", "parts": [{"text": "Olá humano"}]}
        ], None)

        config = {
            "selectedSourceIds": [],
            "includeChatContext": True
        }

        result = SourceAssemblyService.get_context_from_config(self.chat.id, config)

        self.assertIn("--- CHAT HISTORY ---", result)
        self.assertIn("USER: Olá bot", result)
        self.assertIn("MODEL: Olá humano", result)

    @patch('chat.file_processor.FileProcessor.extract_text')
    def test_get_context_mixed(self, mock_extract):
        """Testa arquivos + histórico."""
        msg1 = ChatMessage.objects.create(
            chat=self.chat,
            role='user',
            attachment="dummy.txt",
            original_filename="dummy.txt"
        )
        mock_extract.return_value = "Texto do Arquivo"

        # Mock do histórico via patch interno da função seria complexo,
        # então vamos confiar que ele chama a função mockada no teste anterior
        # ou vamos mockar build_conversation_history novamente aqui se quisermos testar integração

        with patch('studio.services.source_assembler.build_conversation_history') as mock_hist:
            mock_hist.return_value = ([{"role": "user", "parts": [{"text": "Histórico"}]}], None)

            config = {
                "selectedSourceIds": [msg1.id],
                "includeChatContext": True
            }

            result = SourceAssemblyService.get_context_from_config(self.chat.id, config)

            self.assertIn("--- FILE: dummy.txt ---", result)
            self.assertIn("Texto do Arquivo", result)
            self.assertIn("--- CHAT HISTORY ---", result)
            self.assertIn("USER: Histórico", result)
