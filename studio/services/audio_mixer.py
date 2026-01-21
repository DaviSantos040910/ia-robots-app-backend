import os
import logging
import uuid
import tempfile
from django.conf import settings
from pydub import AudioSegment
from chat.services.tts_service import generate_tts_audio

logger = logging.getLogger(__name__)

class AudioMixerService:
    @staticmethod
    def mix_podcast(script: list) -> str:
        """
        Mixes a podcast script into a single audio file.
        script: List of dicts [{"speaker": "...", "text": "..."}]
        Returns: Relative path to the generated audio file in MEDIA_ROOT.
        """
        if not script:
            raise ValueError("Script is empty.")

        # Voice Mapping
        # Gemini Voices: 'Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'
        # Alex (Host) -> 'Kore' (Balanced)
        # Jamie (Guest) -> 'Fenrir' (Deeper/Distinct) or 'Puck'
        voice_map = {
            "Host (Alex)": "Kore",
            "Guest (Jamie)": "Fenrir"
        }

        full_audio = AudioSegment.empty()
        temp_files = []

        try:
            for turn in script:
                speaker = turn.get("speaker", "Host (Alex)")
                text = turn.get("text", "")
                if not text: continue

                voice = voice_map.get(speaker, "Kore")

                # Generate segment
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    temp_path = tf.name

                result = generate_tts_audio(text, temp_path, voice_name=voice)

                if result.get('success'):
                    temp_files.append(temp_path)
                    try:
                        segment = AudioSegment.from_wav(temp_path)
                        full_audio += segment
                        # Add a small pause between speakers
                        full_audio += AudioSegment.silent(duration=300)
                    except Exception as e:
                        logger.error(f"Error processing audio segment for text '{text[:20]}...': {e}")
                else:
                    logger.warning(f"Failed to generate audio for turn: {result.get('error')}")

            # Export Final Mix
            filename = f"podcast_mix_{uuid.uuid4().hex[:10]}.mp3"
            # Ensure media directory exists
            output_dir = os.path.join(settings.MEDIA_ROOT, 'podcasts')
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, filename)

            # Export
            # Note: bitrate 128k is standard for speech
            full_audio.export(output_path, format="mp3", bitrate="128k")

            return f"podcasts/{filename}"

        except Exception as e:
            logger.error(f"Error mixing podcast: {e}")
            raise e
        finally:
            # Cleanup temp files
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
