# chat/vector_service.py

import chromadb
import google.generativeai as genai
from django.conf import settings
import logging
import uuid
from datetime import datetime
import os

# Configuração de Logger
logger = logging.getLogger(__name__)


class VectorService:

    def __init__(self):
        """
        Inicializa o cliente ChromaDB e configura a API do Google Gemini.
        """
        self.client = None
        self.collection = None
        try:
            # 1. Configuração da API do Google Gemini
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.error("GEMINI_API_KEY não encontrada nas configurações.")
                return

            genai.configure(api_key=api_key)

            # 2. Inicialização do ChromaDB
            db_path = str(settings.CHROMA_DB_PATH)

            if not os.path.exists(db_path):
                os.makedirs(db_path, exist_ok=True)

            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name="chat_memory")
            logger.info(f"VectorService inicializado com sucesso. Path: {db_path}")

        except Exception as e:
            logger.critical(f"Falha crítica ao inicializar VectorService: {str(e)}")

    def _get_embedding(self, text, task_type="retrieval_document"):
        """
        Gera o embedding do texto usando o modelo 'models/text-embedding-004'.
        """
        if not text:
            return None
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type=task_type,
                title="Chat Memory" if task_type == "retrieval_document" else None
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Erro na API Google Embeddings: {str(e)}")
            return None

    def add_memory(self, user_id, bot_id, text, role):
        """
        Adiciona uma interação à memória de longo prazo.
        """
        if not self.collection:
            logger.warning("Tentativa de adicionar memória sem ChromaDB inicializado.")
            return

        try:
            embedding = self._get_embedding(text, task_type="retrieval_document")
            if not embedding:
                logger.warning("Embedding falhou. Memória ignorada.")
                return

            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            metadatas = {
                'user_id': str(user_id),
                'bot_id': str(bot_id),
                'role': role,
                'timestamp': timestamp
            }

            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadatas],
                ids=[doc_id]
            )
            logger.debug(f"Memória adicionada: {doc_id} (User: {user_id}, Bot: {bot_id})")

        except Exception as e:
            logger.error(f"Erro ao salvar memória no ChromaDB: {str(e)}")

    def search_memory(self, user_id, bot_id, query_text, limit=5, exclude_texts=None):
        """
        Busca contextos relevantes na memória baseados na query atual.
        Filtra textos que já estão no histórico recente para evitar repetição.
        
        Args:
            user_id: ID do usuário
            bot_id: ID do bot
            query_text: Texto da busca
            limit: Número máximo de resultados
            exclude_texts: Lista de textos a excluir (histórico recente)
        """
        if not self.collection:
            logger.warning("Tentativa de buscar memória sem ChromaDB inicializado.")
            return []

        exclude_texts = exclude_texts or []
        # Normaliza textos para comparação (primeiros 100 chars, lowercase)
        exclude_set = set()
        for text in exclude_texts:
            if text:
                normalized = text.lower().strip()[:100]
                exclude_set.add(normalized)

        try:
            embedding = self._get_embedding(query_text, task_type="retrieval_query")
            if not embedding:
                return []

            # Busca mais resultados para compensar os que serão filtrados
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit * 3,  # Busca 3x mais para compensar filtros
                where={
                    "$and": [
                        {"user_id": str(user_id)},
                        {"bot_id": str(bot_id)}
                    ]
                }
            )

            if results and results.get('documents'):
                found_docs = results['documents'][0]
                
                # Filtra documentos que já estão no histórico recente
                filtered_docs = []
                for doc in found_docs:
                    doc_normalized = doc.lower().strip()[:100]
                    
                    # Verifica se o documento é similar a algum texto excluído
                    is_duplicate = False
                    for excluded in exclude_set:
                        # Se houver sobreposição significativa, considera duplicata
                        if doc_normalized in excluded or excluded in doc_normalized:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        filtered_docs.append(doc)
                    
                    if len(filtered_docs) >= limit:
                        break
                
                logger.debug(f"Memória: {len(found_docs)} encontrados, {len(filtered_docs)} após filtro.")
                return filtered_docs

            return []

        except Exception as e:
            logger.error(f"Erro ao buscar memória no ChromaDB: {str(e)}")
            return []
