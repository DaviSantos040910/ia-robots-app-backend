import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do ficheiro .env
load_dotenv()

# Configurar a API com a sua chave
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERRO: A variável GEMINI_API_KEY não foi encontrada no ficheiro .env")
else:
    genai.configure(api_key=api_key)
    
    print("A procurar por modelos disponíveis para a sua chave de API...")
    print("-" * 30)
    
    try:
        # Listar todos os modelos que suportam o método 'generateContent'
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Ocorreu um erro ao contactar a API da Google: {e}")

    print("-" * 30)
    print("Copie um dos nomes da lista acima (ex: models/gemini-1.5-flash-latest) e use-o no seu código.")