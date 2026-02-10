from django.test import TestCase
from unittest.mock import patch, MagicMock
from studio.models import KnowledgeArtifact
from studio.jobs.artifact_jobs import generate_artifact_job
from chat.models import Chat, Bot
from django.contrib.auth import get_user_model
from django.utils import timezone
import uuid

User = get_user_model()

class ArtifactJobTest(TestCase):
    def setUp(self):
        # Create user
        self.user = User.objects.create(email="test@example.com", username="testuser")
        # Create bot (requires name and owner)
        self.bot = Bot.objects.create(name="Test Bot", owner=self.user)
        # Create chat
        self.chat = Chat.objects.create(user=self.user, bot=self.bot)
        # Create artifact
        self.artifact = KnowledgeArtifact.objects.create(
            chat=self.chat,
            type=KnowledgeArtifact.ArtifactType.QUIZ,
            title="Test Quiz",
            status=KnowledgeArtifact.Status.PROCESSING,
            stage=KnowledgeArtifact.Stage.QUEUED,
            correlation_id=uuid.uuid4()
        )
        self.options = {
            'quantity': 5,
            'difficulty': 'Easy'
        }

    @patch('studio.jobs.artifact_jobs.SourceAssemblyService.get_context_from_config')
    @patch('studio.jobs.artifact_jobs.get_ai_client')
    @patch('studio.jobs.artifact_jobs.get_model')
    def test_generate_standard_artifact_success(self, mock_get_model, mock_get_client, mock_get_context):
        # Setup Mocks
        mock_get_context.return_value = "Test Context"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_model.return_value = "gemini-pro"

        mock_response = MagicMock()
        mock_response.parsed = {"questions": []}
        mock_client.models.generate_content.return_value = mock_response

        # Execute Job
        generate_artifact_job(self.artifact.id, self.options)

        # Assertions
        self.artifact.refresh_from_db()
        self.assertEqual(self.artifact.status, KnowledgeArtifact.Status.READY)
        self.assertEqual(self.artifact.stage, KnowledgeArtifact.Stage.READY)
        self.assertIsNotNone(self.artifact.finished_at)
        self.assertEqual(self.artifact.content, {"questions": []})
        self.assertEqual(self.artifact.attempts, 1)

    @patch('studio.jobs.artifact_jobs.SourceAssemblyService.get_context_from_config')
    def test_generate_artifact_failure(self, mock_get_context):
        # Setup Failure
        mock_get_context.side_effect = Exception("Context Error")

        # Execute Job
        with self.assertRaises(Exception):
            generate_artifact_job(self.artifact.id, self.options)

        # Assertions
        self.artifact.refresh_from_db()
        self.assertEqual(self.artifact.status, KnowledgeArtifact.Status.ERROR)
        self.assertEqual(self.artifact.stage, KnowledgeArtifact.Stage.ERROR)
        self.assertIn("Context Error", self.artifact.error_message)
        self.assertIsNotNone(self.artifact.finished_at)

    @patch('studio.jobs.artifact_jobs.SourceAssemblyService.get_context_from_config')
    @patch('studio.jobs.artifact_jobs.PodcastScriptingService.generate_script')
    @patch('studio.jobs.artifact_jobs.AudioMixerService.mix_podcast')
    def test_generate_podcast_success(self, mock_mix, mock_script, mock_get_context):
        # Setup Podcast Artifact
        self.artifact.type = KnowledgeArtifact.ArtifactType.PODCAST
        self.artifact.save()

        mock_get_context.return_value = "Context"
        mock_script.return_value = [{"speaker": "Host", "text": "Hello"}]
        mock_mix.return_value = "podcast.mp3"

        # Execute
        generate_artifact_job(self.artifact.id, self.options)

        # Assertions
        self.artifact.refresh_from_db()
        self.assertEqual(self.artifact.status, KnowledgeArtifact.Status.READY)
        self.assertEqual(self.artifact.media_url, "/media/podcast.mp3")
        self.assertEqual(self.artifact.content, [{"speaker": "Host", "text": "Hello"}])
