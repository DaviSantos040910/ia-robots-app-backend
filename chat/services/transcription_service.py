# chat/services/transcription_service.py

import mimetypes
import logging
from google.genai import types

from .ai_client import get_ai_client

logger = logging.getLogger(__name__)


def transcribe_audio_gemini(audio_file) -> dict:
    """
    Transcreve áudio usando Gemini.
    Lógica idêntica à transcribe_audio_gemini em ai_service.py.
    """
    try:
        client = get_ai_client()

        audio_bytes = audio_file.read()
        mime_type, _ = mimetypes.guess_type(audio_file.name)
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
