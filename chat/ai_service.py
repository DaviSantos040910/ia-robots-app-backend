# chat/ai_service.py
import google.generativeai as genai
from django.conf import settings
from .models import ChatMessage
from bots.models import Bot

# Configure the Gemini client with the API key from settings
genai.configure(api_key=settings.GEMINI_API_KEY)

def get_ai_response(chat_id: int, user_message: str) -> str:
    """
    Gets a response from the Gemini model based on the chat history.

    Args:
        chat_id: The ID of the current chat session.
        user_message: The latest message from the user.

    Returns:
        The content of the AI's response as a string.
    """
    try:
        # Retrieve the chat and its associated bot
        chat = ChatMessage.objects.select_related('chat__bot').filter(chat_id=chat_id).first().chat
        bot = chat.bot

        # Initialize the generative model
        model = genai.GenerativeModel('gemini-pro')

        # Get the last 10 messages to build the conversation history
        history = ChatMessage.objects.filter(chat_id=chat_id).order_by('created_at').select_related('chat__bot')[:10]

        # Format the history for the Gemini API.
        # The system prompt is handled differently.
        # Gemini expects a list of `content` objects with `role` and `parts`.
        gemini_history = []
        for msg in history:
            # The Gemini API uses 'model' for the assistant's role
            role = 'user' if msg.role == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [{'text': msg.content}]})

        # Start a chat session with the bot's prompt as the system instruction
        # and the existing message history.
        chat_session = model.start_chat(
            history=gemini_history,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=150
            )
        )
        
        # The system prompt in Gemini is best added as the first part of the user's message
        # if the API doesn't have a dedicated system prompt field in the chat session.
        # A common pattern is to frame the context at the start.
        # Note: The bot's prompt (system instruction) is prepended here for context.
        prompt_with_context = f"{bot.prompt}\n\nUser: {user_message}\nAI:"

        # Send the user's message to the chat session
        response = chat_session.send_message(prompt_with_context)

        return response.text.strip()

    except Exception as e:
        print(f"An error occurred in Gemini AI service: {e}")
        return "An unexpected error occurred while generating a response. Please try again."