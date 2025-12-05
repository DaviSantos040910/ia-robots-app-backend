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
        if not self.collection: return

        try:
            embedding = self._get_embedding(text, task_type="retrieval_document")
            if not embedding: return

            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            metadatas = {
                'user_id': str(user_id),
                'bot_id': str(bot_id),
                'role': role,
                'timestamp': timestamp,
                'type': 'memory' # Diferencia memória de conversação de documentos
            }

            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadatas],
                ids=[doc_id]
            )
            logger.debug(f"Memória adicionada: {doc_id}")

        except Exception as e:
            logger.error(f"Erro ao salvar memória: {str(e)}")

    def add_document_chunks(self, user_id, bot_id, chunks, source_name):
        """
        Adiciona chunks de um documento processado ao ChromaDB.
        Método síncrono e otimizado para inserção em lote.
        """
        if not self.collection or not chunks:
            logger.warning("VectorService: Coleção não iniciada ou chunks vazios.")
            return

        logger.info(f"Iniciando indexação de {len(chunks)} chunks para {source_name}...")
        
        docs, embeds, metas, ids = [], [], [], []
        timestamp = datetime.now().isoformat()
        
        for i, chunk in enumerate(chunks):
            # Gera embedding para cada chunk
            embedding = self._get_embedding(chunk, task_type="retrieval_document")
            
            if embedding:
                doc_id = str(uuid.uuid4())
                docs.append(chunk)
                embeds.append(embedding)
                ids.append(doc_id)
                
                # Metadados específicos para documentos RAG
                metas.append({
                    'user_id': str(user_id),
                    'bot_id': str(bot_id),
                    'type': 'document',
                    'source': source_name,
                    'chunk_index': i,
                    'timestamp': timestamp
                })
        
        # Salva em lote no ChromaDB
        if docs:
            try:
                self.collection.add(
                    documents=docs, 
                    embeddings=embeds, 
                    metadatas=metas, 
                    ids=ids
                )
                logger.info(f"RAG SUCESSO: {len(docs)} chunks indexados de '{source_name}'.")
            except Exception as e:
                logger.error(f"RAG ERRO ao salvar no banco vetorial: {e}")

    def search_context(self, query_text, user_id, bot_id, limit=5):
        """
        Busca híbrida: Retorna tanto memória quanto trechos de documentos.
        """
        if not self.collection: return []
        try:
            embedding = self._get_embedding(query_text, task_type="retrieval_query")
            if not embedding: return []

            # Busca vetorial filtrada por usuário e bot
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                where={"$and": [{"user_id": str(user_id)}, {"bot_id": str(bot_id)}]}
            )

            formatted = []
            if results and results['documents']:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    doc_type = meta.get('type', 'memory')
                    
                    if doc_type == 'document':
                        source = meta.get('source', 'Anexo desconhecido')
                        formatted.append(f"[CONTEXTO DO ARQUIVO: {source}]\n{doc}")
                    else:
                        formatted.append(f"[MEMÓRIA DE CONVERSA]\n{doc}")
            
            return formatted
        except Exception as e:
            logger.error(f"Erro search_context: {e}")
            return []