# chat/services/ai_client.py

from google import genai
import os


def get_ai_client():
    """
    Retorna o client do Gemini usando a variável de ambiente GEMINI_API_KEY.
    Lógica idêntica à função original em ai_service.py.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada")
    client = genai.Client(api_key=api_key)
    return client
