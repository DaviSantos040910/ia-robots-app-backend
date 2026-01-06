# chat/services/transcription_service.py

import mimetypes
import logging
from google.genai import types

from .ai_client import get_ai_client

logger = logging.getLogger(__name__)


def transcribe_audio_gemini(audio_file) -> dict:
    """
    Transcreve áudio usando Gemini.
    """
    try:
        client = get_ai_client()

        # Se audio_file for UploadedFile, use .read(). Se for path, open.
        if hasattr(audio_file, 'read'):
            audio_bytes = audio_file.read()
            # Reset pointer se necessário, mas .read() consome.
            # Se for usado novamente, precisaria de seek(0).
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)

            filename = getattr(audio_file, 'name', 'audio.m4a')
        else:
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            filename = str(audio_file)

        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = 'audio/m4a'

        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        prompt = (
            "Generate a transcript of the speech in Portuguese. "
            "Return only the transcribed text, strictly without timestamps or speaker labels."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, audio_part],
            config=types.GenerateContentConfig(temperature=0.2)
        )

        if response.text:
            return {'success': True, 'transcription': response.text.strip()}
        else:
            return {'success': False, 'error': 'No transcription returned'}

    except Exception as e:
        logger.error(f"[Transcription Error] {e}")
        return {'success': False, 'error': str(e)}
