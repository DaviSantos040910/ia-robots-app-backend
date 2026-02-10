import json
import logging
from google.genai import types
from chat.services.ai_client import get_ai_client, get_model

logger = logging.getLogger(__name__)

class PodcastScriptingService:
    @staticmethod
    def generate_script(title: str, context: str, duration_constraint: str = "Medium", bot_name: str = "Alex", bot_prompt: str = "") -> list:
        """
        Generates a podcast script dialogue between the Bot (Host) and a Co-host.
        Returns a list of dicts: [{"speaker": "Host (BotName)", "text": "..."}, ...]
        """
        client = get_ai_client()
        model_name = get_model('chat')

        host_name = f"Host ({bot_name})"
        cohost_name = "Co-host"

        # Define Schema with Dynamic Host Name
        script_schema = types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "speaker": types.Schema(type=types.Type.STRING, enum=[host_name, cohost_name]),
                    "text": types.Schema(type=types.Type.STRING)
                },
                required=["speaker", "text"]
            )
        )

        persona_instruction = ""
        if bot_prompt:
            persona_instruction = f"HOST PERSONA ({host_name}):\n{bot_prompt}\nAdopt this persona for the Host's tone and style.\n"

        prompt = (
            f"Create an engaging, educational podcast script titled '{title}'.\n"
            f"ROLES:\n"
            f"- {host_name}: The expert/guide. {persona_instruction}\n"
            f"- {cohost_name}: The curious, insightful interviewer who asks questions to clarity content.\n\n"
            f"Style: Conversational, deep dive, slightly informal but professional. NotebookLM style.\n"
            f"Duration Context: {duration_constraint}.\n"
            f"Language: Portuguese (unless context strongly implies English).\n\n"
            f"SOURCE MATERIAL:\n{context}\n\n"
            f"Generate a dialogue script."
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=script_schema,
                    temperature=0.7
                )
            )

            if response.parsed:
                return response.parsed
            else:
                # Fallback
                return json.loads(response.text)

        except Exception as e:
            logger.error(f"Error generating podcast script: {e}")
            raise
