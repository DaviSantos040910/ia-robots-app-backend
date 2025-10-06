# chat/ai_service.py
import google.generativeai as genai
import os
from django.conf import settings
from .models import ChatMessage, Chat
from bots.models import Bot

# Configura a API key a partir do ficheiro .env
try:
    # Esta biblioteca usa a variável GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada no ficheiro .env")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

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

        # Usamos o padrão GenerativeModel, que é o correto para chat
        # As configurações são passadas aqui, na criação do modelo.
        model = genai.GenerativeModel(
            # Usamos o nome do modelo com '-latest' que é mais estável
            model_name='gemini-2.5-flash-lite',
            generation_config=generation_config,
            safety_settings=safety_settings,
            # Adicionamos o prompt do sistema aqui, que é a forma mais moderna
            system_instruction=bot.prompt
        )
        
        history = ChatMessage.objects.filter(chat_id=chat_id).order_by('created_at')[:10]

        gemini_history = []
        for msg in history:
            # Evita enviar o erro genérico de volta para a IA como histórico
            if "An unexpected error occurred" in msg.content:
                continue
            role = 'user' if msg.role == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [{'text': msg.content}]})

        # Iniciamos a sessão de chat, que é o método que lida com o histórico
        chat_session = model.start_chat(history=gemini_history)
        
        # Enviamos apenas a mensagem do utilizador, pois o prompt do sistema já foi definido no modelo
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