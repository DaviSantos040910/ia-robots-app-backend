# chat/ai_service.py
import google.generativeai as genai
import os
import json # <-- CORREÇÃO: Importação do JSON adicionada
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

# --- CORREÇÃO: Função com indentação corrigida ---
def generate_suggestions_for_bot(prompt: str):
    """
    Calls the AI model to generate three initial conversation starters
    based on the bot's main prompt.
    """
    try:
        # Usar 'gemini-1.5-flash-latest' é mais estável
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
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
    
    # --- CORREÇÃO: Indentação do return de fallback corrigida ---
    return ["Tell me more.", "What can you do?", "Give me an example."]


# --- FUNÇÃO ATUALIZADA ---
def get_ai_response(chat_id: int, user_message: str):
    """
    Obtém uma resposta ESTRUTURADA (resposta + sugestões) do modelo Gemini.
    Retorna um dicionário: {'content': '...', 'suggestions': [...]}
    """
    try:
        chat = Chat.objects.select_related('bot').get(id=chat_id)
        bot = chat.bot

        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=500,
            # --- NOVO: Forçar a saída em formato JSON ---
            response_mime_type="application/json"
        )

        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash-latest',
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=bot.prompt
        )
        
        history = ChatMessage.objects.filter(chat_id=chat_id).order_by('created_at').values('role', 'content')[:10]

        gemini_history = []
        for msg in history:
            if "An unexpected error occurred" in msg.content:
                continue
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [{'text': msg['content']}]})

        chat_session = model.start_chat(history=gemini_history)
        
        # --- NOVO: Instrução para a IA retornar um JSON ---
        ai_prompt = f"""
        User's message: "{user_message}"

        Based on the user's message and the conversation history, provide a helpful response and generate exactly two distinct, short, and relevant follow-up suggestions (under 10 words each) that the user might ask next.

        Respond with a valid JSON object with the following structure:
        {{
          "response": "Your main answer to the user's message goes here.",
          "suggestions": ["First suggestion", "Second suggestion"]
        }}
        """
        
        response = chat_session.send_message(ai_prompt)

        if response.parts:
            # Parse the JSON response
            response_data = json.loads(response.text)
            return {
                'content': response_data.get('response', "Sorry, I couldn't generate a response."),
                'suggestions': response_data.get('suggestions', [])
            }
        else:
            reason = "UNKNOWN"
            if response.candidates and response.candidates[0].finish_reason:
                 reason = response.candidates[0].finish_reason.name
            print(f"Gemini response was blocked. Finish Reason: {reason}")
            return {
                'content': "My response was blocked. Please try a different message.",
                'suggestions': []
            }

    except Exception as e:
        print(f"An error occurred in Gemini AI service: {e}")
        return {
            'content': "An unexpected error occurred while generating a response. Please try again.",
            'suggestions': []
        }