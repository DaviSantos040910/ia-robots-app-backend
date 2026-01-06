from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, MagicMock
from chat.file_processor import FileProcessor
import io
import zipfile

class FileProcessorTests(TestCase):
    def test_extract_text_from_txt(self):
        file_content = b"Hello, this is a test text file."
        file = SimpleUploadedFile("test.txt", file_content, content_type="text/plain")

        result = FileProcessor.process_file(file)
        self.assertEqual(result['extracted_text'], "Hello, this is a test text file.")

    def test_process_zip_file(self):
        # Create a mock zip file in memory
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('hello.txt', 'Hello world')
            z.writestr('code.py', 'print("Python")')
            z.writestr('ignore.exe', 'binary data')

        buffer.seek(0)
        file = SimpleUploadedFile("archive.zip", buffer.read(), content_type="application/zip")

        result = FileProcessor.process_file(file)

        extracted = result['extracted_text']
        self.assertIn("--- FILE: hello.txt ---", extracted)
        self.assertIn("Hello world", extracted)
        self.assertIn("--- FILE: code.py ---", extracted)
        self.assertIn('print("Python")', extracted)
        self.assertNotIn("binary data", extracted)

        # self.assertIn('archive.zip', result['metadata']['type'])
        self.assertEqual(result['metadata']['type'], 'zip_content')
        self.assertIn('hello.txt', result['metadata']['structure'])

    @patch('chat.file_processor.YouTubeTranscriptApi')
    def test_process_youtube_link(self, mock_api_class):
        # Mock the chain: YouTubeTranscriptApi().list().find_transcript().fetch()
        mock_instance = mock_api_class.return_value
        mock_list = mock_instance.list.return_value
        mock_transcript = mock_list.find_transcript.return_value

        # fetch returns a list of dicts
        mock_transcript.fetch.return_value = [{'text': 'Hello youtube', 'start': 0.0, 'duration': 1.0}]

        file_content = b"https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        file = SimpleUploadedFile("youtube_link.txt", file_content, content_type="text/plain")

        result = FileProcessor.process_file(file)

        # Verify calls
        mock_api_class.assert_called()
        mock_instance.list.assert_called_with('dQw4w9WgXcQ')
        mock_list.find_transcript.assert_called_with(['pt', 'en'])
        mock_transcript.fetch.assert_called()

        self.assertIn("Hello youtube", result['extracted_text'])
        self.assertEqual(result['metadata']['source'], 'youtube_transcript')
        self.assertEqual(result['metadata']['video_id'], 'dQw4w9WgXcQ')

    @patch('chat.file_processor.transcribe_audio_gemini')
    def test_process_audio(self, mock_transcribe):
        mock_response = {'success': True, 'transcription': 'This is audio text.'}
        mock_transcribe.return_value = mock_response

        file_content = b"audio data"
        file = SimpleUploadedFile("audio.mp3", file_content, content_type="audio/mpeg")

        result = FileProcessor.process_file(file, file_type="audio")

        mock_transcribe.assert_called_once()
        self.assertEqual(result['extracted_text'], 'This is audio text.')
        self.assertEqual(result['metadata']['type'], 'audio_transcription')
