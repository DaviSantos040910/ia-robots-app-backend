# chat/ai_service.py
import os
import json
import google.generativeai as genai
from django.conf import settings
from .models import ChatMessage, Chat
from bots.models import Bot

# Configura a API key a partir do ficheiro .env
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada no ficheiro .env")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

def generate_suggestions_for_bot(prompt: str):
    """
    Calls the AI model to generate three initial conversation starters
    based on the bot's main prompt.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        instruction = f"""
        Based on the following bot's instructions, generate exactly three short, engaging, and distinct conversation starters (under 10 words each).
        The user will see these as suggestion chips to start the conversation.
        Return the result as a valid JSON array of strings. For example: ["Suggestion 1", "Suggestion 2", "Suggestion 3"].

        Bot Instructions: "{prompt}"
        """

        response = model.generate_content(instruction)

        cleaned_response = response.text.strip().replace('`', '').replace('json', '')
        suggestions = json.loads(cleaned_response)

        if isinstance(suggestions, list) and len(suggestions) == 3 and all(isinstance(s, str) for s in suggestions):
            return suggestions

    except Exception as e:
        print(f"Could not generate suggestions: {e}")

    # Fallback in case of an error
    return ["Tell me more.", "What can you do?", "Give me an example."]


def get_ai_response(chat_id: int, user_message: str) -> str:
    """
    Obtém uma resposta do modelo Gemini usando o método de chat,
    que é o ideal para conversas com histórico e configurações avançadas.
    """
    try:
        chat = Chat.objects.select_related('bot').get(id=chat_id)
        bot = chat.bot

        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=500
        )

        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite',
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=bot.prompt
        )

        history = ChatMessage.objects.filter(chat_id=chat_id).order_by('created_at')[:10]

        gemini_history = []
        for msg in history:
            if "An unexpected error occurred" in msg.content:
                continue
            role = 'user' if msg.role == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [{'text': msg.content}]})

        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_message)

        if response.parts:
            return response.text.strip()
        else:
            reason = "UNKNOWN"
            if response.candidates and response.candidates[0].finish_reason:
                reason = response.candidates[0].finish_reason.name
            print(f"Gemini response was blocked. Finish Reason: {reason}")
            return "My response was blocked. Please try a different message."

    except Exception as e:
        print(f"An error occurred in Gemini AI service: {e}")
        return "An unexpected error occurred while generating a response. Please try again."
