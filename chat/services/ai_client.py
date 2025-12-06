# chat/services/ai_client.py
"""
Cliente de IA centralizado com suporte a Gemini API e Vertex AI.
Alterne entre os dois mudando USE_VERTEX_AI nas settings.
"""

from google import genai
from google.genai import types
from django.conf import settings
import os
import logging

logger = logging.getLogger(__name__)

# Flag para alternar entre Gemini API e Vertex AI
# Adicione USE_VERTEX_AI = False no settings.py (mude para True quando migrar)
USE_VERTEX_AI = getattr(settings, 'USE_VERTEX_AI', False)

# Configurações Vertex AI (preencha quando for migrar)
VERTEX_PROJECT_ID = getattr(settings, 'VERTEX_PROJECT_ID', '')
VERTEX_LOCATION = getattr(settings, 'VERTEX_LOCATION', 'us-central1')


def get_ai_client():
    """
    Retorna o client apropriado baseado na configuração.
    - Gemini API: usa GEMINI_API_KEY
    - Vertex AI: usa credenciais do Google Cloud
    """
    if USE_VERTEX_AI:
        return _get_vertex_client()
    return _get_gemini_client()


def _get_gemini_client():
    """Cliente para Gemini API (gratuito/pago com API Key)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não encontrada nas variáveis de ambiente")
    return genai.Client(api_key=api_key)


def _get_vertex_client():
    """
    Cliente para Vertex AI (Google Cloud).
    Requer: pip install google-cloud-aiplatform
    E configurar GOOGLE_APPLICATION_CREDENTIALS ou gcloud auth
    """
    if not VERTEX_PROJECT_ID:
        raise ValueError("VERTEX_PROJECT_ID não configurado no settings.py")
    
    # Vertex AI usa autenticação do Google Cloud (não API key)
    return genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION
    )


def detect_intent(user_message: str) -> str:
    """
    Classifica a intenção do usuário: TEXT ou IMAGE.
    Usa modelo leve para economia de tokens.
    """
    if not user_message or len(user_message.strip()) < 3:
        return "TEXT"
    
    try:
        client = get_ai_client()
        
        prompt = f"""Analise a mensagem abaixo e determine se o usuário está pedindo para CRIAR/GERAR uma imagem visual.

Mensagem: "{user_message}"

Regras:
- IMAGE: Se pedir explicitamente para criar, gerar, desenhar, fazer uma imagem/foto/ilustração/arte
- TEXT: Para qualquer outro tipo de pergunta, pedido de texto, explicação, análise

Responda APENAS: TEXT ou IMAGE"""

        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',  # Modelo mais leve e rápido
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=10
            )
        )
        
        if response.text:
            result = response.text.strip().upper()
            if result in ['TEXT', 'IMAGE']:
                return result
        
        return "TEXT"
        
    except Exception as e:
        logger.warning(f"[detect_intent] Erro: {e}")
        return "TEXT"


# Modelos disponíveis por tipo de API
MODELS = {
    'gemini_api': {
        'chat': 'gemini-2.5-flash',
        'chat_pro': 'gemini-2.5-pro',
        'image': 'gemini-2.5-flash-image',  # Geração de imagem
        'embedding': 'text-embedding-004',
        'tts': 'gemini-2.5-flash-preview-tts',
        'lite': 'gemini-2.5-flash-lite',
    },
    'vertex_ai': {
        'chat': 'gemini-2.5-flash',
        'chat_pro': 'gemini-2.5-pro',
        'image': 'imagen-3.0-generate-002',  # Imagen 3 (melhor qualidade)
        'embedding': 'text-embedding-004',
        'tts': 'gemini-2.5-flash-preview-tts',
        'lite': 'gemini-2.5-flash-lite',
    }
}


def get_model(model_type: str) -> str:
    """
    Retorna o modelo apropriado baseado no tipo e na configuração atual.
    
    Args:
        model_type: 'chat', 'chat_pro', 'image', 'embedding', 'tts', 'lite'
    
    Returns:
        Nome do modelo para usar na API
    """
    api_type = 'vertex_ai' if USE_VERTEX_AI else 'gemini_api'
    return MODELS[api_type].get(model_type, MODELS[api_type]['chat'])
