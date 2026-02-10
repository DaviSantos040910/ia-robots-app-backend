import json
import logging
from google.genai import types
from chat.services.ai_client import get_ai_client
from core.genai_models import GENAI_MODEL_TEXT

logger = logging.getLogger(__name__)

class PodcastScriptingService:
    @staticmethod
    def generate_script(title: str, context: str, duration_constraint: str = "Medium", bot_name: str = "Alex", bot_prompt: str = "") -> list:
        """
        Generates a podcast script dialogue between the Bot (Host) and a Co-host.
        Returns a list of dicts: [{"speaker": "Host (BotName)", "text": "..."}, ...]
        """
        client = get_ai_client()
        model_name = GENAI_MODEL_TEXT

        host_display = f"Host ({bot_name})"

        # Define Final Schema (NotebookLM-like)
        script_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "episode_title": types.Schema(type=types.Type.STRING, description="Catchy title (4-80 chars)"),
                "episode_summary": types.Schema(type=types.Type.STRING, description="Brief summary (2-4 lines, no markdown)"),
                "chapters": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "title": types.Schema(type=types.Type.STRING),
                            "start_turn_index": types.Schema(type=types.Type.INTEGER)
                        }
                    )
                ),
                "dialogue": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "speaker": types.Schema(type=types.Type.STRING, enum=["HOST", "COHOST"]),
                            "display_name": types.Schema(type=types.Type.STRING),
                            "text": types.Schema(type=types.Type.STRING)
                        },
                        required=["speaker", "text"]
                    )
                )
            },
            required=["episode_title", "episode_summary", "chapters", "dialogue"]
        )

        # 1. STYLE / PERSONA
        persona_instruction = ""
        if bot_prompt:
            persona_instruction = f"HOST PERSONA:\n{bot_prompt}\nAdopt this persona for the Host's tone and style.\n"

        style_block = (
            f"STYLE & ROLES:\n"
            f"- HOST ({host_display}): The expert/guide. {persona_instruction}\n"
            f"- COHOST: The curious, insightful interviewer who asks questions to clarify content.\n"
            f"- Tone: Conversational, deep dive, slightly informal but professional. NotebookLM style.\n"
            f"- Language: Portuguese (unless context strongly implies English).\n"
            f"- Duration Context: {duration_constraint}.\n"
        )

        # 2. FACT POLICY (HARD)
        fact_policy = (
            "FACT POLICY (STRICT RULES):\n"
            "- USE ONLY THE PROVIDED SOURCE MATERIAL FOR FACTS.\n"
            "- DO NOT INVENT DETAILS, NUMBERS, OR EXAMPLES NOT PRESENT IN THE SOURCE.\n"
            "- IF THE CONTEXT IS INSUFFICIENT, THE HOST MUST CLEARLY STATE THAT THE TOPIC CANNOT BE FULLY DISCUSSED.\n"
            "- AVOID HALLUCINATIONS AT ALL COSTS.\n"
        )

        # 3. TASK & CONTEXT
        prompt = (
            f"Create an engaging, educational podcast script titled '{title}'.\n\n"
            f"{style_block}\n"
            f"{fact_policy}\n"
            f"SOURCE MATERIAL:\n{context}\n\n"
            f"Generate a full script with title, summary, chapters, and dialogue."
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=script_schema,
                    temperature=0.4
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
