from django.test import TestCase
from unittest.mock import patch, MagicMock
from studio.services.podcast_scripting import PodcastScriptingService
from studio.services.audio_mixer import AudioMixerService

class PodcastPersonalizationTest(TestCase):

    @patch('studio.services.podcast_scripting.get_ai_client')
    def test_script_generation_includes_bot_persona(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = [{"speaker": "Host (Tutor)", "text": "Hello"}]
        mock_client.models.generate_content.return_value = mock_response

        # Execute
        script = PodcastScriptingService.generate_script(
            title="Lesson 1",
            context="Content",
            bot_name="Tutor",
            bot_prompt="Be funny."
        )

        # Assertions
        self.assertEqual(script[0]['speaker'], "Host (Tutor)")

        # Check Prompt Construction
        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs['contents']

        self.assertIn("Host (Tutor)", prompt)
        self.assertIn("Be funny", prompt)
        self.assertIn("Co-host", prompt)

    @patch('studio.services.audio_mixer.generate_tts_audio')
    @patch('studio.services.audio_mixer.tempfile.NamedTemporaryFile')
    def test_audio_mixer_voice_selection(self, mock_temp, mock_tts):
        # Setup Mocks
        mock_tts.return_value = {'success': True}
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"

        script = [
            {"speaker": "Host (Tutor)", "text": "Hi"},
            {"speaker": "Co-host", "text": "Hello"}
        ]

        # We mock Pydub AudioSegment to avoid real file ops failing
        with patch('studio.services.audio_mixer.AudioSegment'):
            AudioMixerService.mix_podcast(script)

        # Verify Voice Selection
        # Call 1: Host -> Kore
        # Call 2: Co-host -> Fenrir

        # We can inspect calls to generate_tts_audio
        self.assertEqual(mock_tts.call_count, 2)

        # Since calls happen in threads, order isn't guaranteed, so we check existence
        calls = [c.kwargs['voice_name'] for c in mock_tts.call_args_list]
        self.assertIn("Kore", calls)
        self.assertIn("Fenrir", calls)
