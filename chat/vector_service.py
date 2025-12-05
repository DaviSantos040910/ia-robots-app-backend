# chat/vector_service.py
"""
Serviço vetorial com suporte inteligente a múltiplos documentos.
"""

import chromadb
import google.generativeai as genai
from django.conf import settings
import logging
import uuid
from datetime import datetime
import os
import re
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Tipos de query para determinar estratégia de busca."""
    REFERENCE = "reference"      # "o que é isso?", "esse documento"
    SPECIFIC = "specific"        # "no arquivo X.pdf", "segundo o documento Y"
    COMPARATIVE = "comparative"  # "compare os dois", "diferença entre os documentos"
    GENERAL = "general"          # Perguntas gerais que podem estar em qualquer doc


class VectorService:
    """
    Gerencia busca vetorial com suporte inteligente a múltiplos documentos.
    """
    
    def __init__(self):
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Inicializa ChromaDB e API Gemini."""
        try:
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.error("GEMINI_API_KEY não encontrada.")
                return
            
            genai.configure(api_key=api_key)
            
            db_path = str(settings.CHROMA_DB_PATH)
            os.makedirs(db_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="chat_memory",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"VectorService inicializado: {db_path}")
            
        except Exception as e:
            logger.critical(f"Falha ao inicializar VectorService: {e}")
    
    def _get_embedding(self, text: str, task_type: str = "retrieval_document") -> Optional[List[float]]:
        """Gera embedding usando Gemini."""
        if not text or len(text.strip()) < 3:
            return None
        
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text[:8000],
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            return None

    # =========================================================================
    # MÉTODOS DE ADIÇÃO
    # =========================================================================
    
    def add_memory(self, user_id: int, bot_id: int, text: str, role: str) -> None:
        """Adiciona memória de conversação."""
        if not self.collection or not text:
            return
        
        try:
            embedding = self._get_embedding(text)
            if not embedding:
                return
            
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{
                    'user_id': str(user_id),
                    'bot_id': str(bot_id),
                    'role': role,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'memory'
                }],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            logger.error(f"Erro ao salvar memória: {e}")

    def add_document_chunks(
        self, 
        user_id: int, 
        bot_id: int, 
        chunks: List[str], 
        source_name: str,
        message_id: Optional[int] = None
    ) -> None:
        """Adiciona chunks de documento com metadados completos."""
        if not self.collection or not chunks:
            return
        
        logger.info(f"Indexando {len(chunks)} chunks de '{source_name}'")
        
        docs, embeds, metas, ids = [], [], [], []
        timestamp = datetime.now().isoformat()
        
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            if not embedding:
                continue
            
            docs.append(chunk)
            embeds.append(embedding)
            ids.append(str(uuid.uuid4()))
            metas.append({
                'user_id': str(user_id),
                'bot_id': str(bot_id),
                'type': 'document',
                'source': source_name,
                'source_lower': source_name.lower(),  # Para busca case-insensitive
                'chunk_index': i,
                'total_chunks': len(chunks),
                'timestamp': timestamp,
                'message_id': str(message_id) if message_id else ''
            })
        
        if docs:
            try:
                self.collection.add(
                    documents=docs, embeddings=embeds, metadatas=metas, ids=ids
                )
                logger.info(f"RAG: {len(docs)} chunks indexados de '{source_name}'")
            except Exception as e:
                logger.error(f"Erro ao indexar documento: {e}")

    # =========================================================================
    # MÉTODOS DE ANÁLISE DE QUERY
    # =========================================================================
    
    def classify_query(self, query: str, available_sources: List[str]) -> Tuple[QueryType, Optional[str]]:
        """
        Classifica o tipo de query e extrai documento específico se mencionado.
        
        Returns:
            Tuple: (QueryType, nome_do_documento_se_especificado)
        """
        query_lower = query.lower()
        
        # 1. Verifica se menciona documento específico pelo nome
        for source in available_sources:
            source_lower = source.lower()
            # Remove extensão para comparação mais flexível
            source_base = os.path.splitext(source_lower)[0]
            
            if source_lower in query_lower or source_base in query_lower:
                return QueryType.SPECIFIC, source
        
        # 2. Detecta queries comparativas
        comparative_patterns = [
            r'\b(compare|comparar|diferença|diferente|versus|vs\.?|entre os)\b',
            r'\b(os dois|ambos|os documentos|os arquivos)\b',
            r'\b(primeiro|segundo|terceiro)\s+(documento|arquivo)\b'
        ]
        if any(re.search(p, query_lower) for p in comparative_patterns):
            return QueryType.COMPARATIVE, None
        
        # 3. Detecta referências pronominais (documento mais recente)
        reference_patterns = [
            r'\b(isso|isto|esse|este|essa|esta)\b',
            r'\b(esse|este|o)\s+(documento|arquivo|pdf|texto)\b',
            r'\bresuma\s*(isso|isto|esse|este)?\b',
            r'\bexplique\s*(isso|isto|esse|este)?\b',
            r'\bo que (é|são|diz|fala)\s*(isso|isto|esse|este)?\b'
        ]
        if any(re.search(p, query_lower) for p in reference_patterns):
            return QueryType.REFERENCE, None
        
        # 4. Query geral - busca em todos os documentos
        return QueryType.GENERAL, None

    def get_available_documents(self, user_id: int, bot_id: int) -> List[Dict]:
        """
        Lista todos os documentos disponíveis para o usuário/bot.
        
        Returns:
            Lista de dicts com 'source' e 'timestamp', ordenados por recência.
        """
        if not self.collection:
            return []
        
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"user_id": str(user_id)},
                        {"bot_id": str(bot_id)},
                        {"type": "document"}
                    ]
                },
                include=["metadatas"]
            )
            
            if not results or not results['metadatas']:
                return []
            
            # Agrupa por source e pega o timestamp mais recente de cada
            docs_map = {}
            for meta in results['metadatas']:
                source = meta.get('source', '')
                timestamp = meta.get('timestamp', '')
                
                if source and (source not in docs_map or timestamp > docs_map[source]):
                    docs_map[source] = timestamp
            
            # Ordena por timestamp (mais recente primeiro)
            sorted_docs = sorted(
                [{'source': s, 'timestamp': t} for s, t in docs_map.items()],
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            return sorted_docs
            
        except Exception as e:
            logger.error(f"Erro ao listar documentos: {e}")
            return []

    # =========================================================================
    # MÉTODO PRINCIPAL DE BUSCA
    # =========================================================================
    
    def search_context(
        self, 
        query_text: str, 
        user_id: int, 
        bot_id: int, 
        limit: int = 6,
        recent_doc_source: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Busca inteligente com suporte a múltiplos documentos.
        
        Estratégias por tipo de query:
        - REFERENCE: Prioriza documento mais recente
        - SPECIFIC: Busca apenas no documento mencionado
        - COMPARATIVE: Busca em todos os documentos, agrupa por fonte
        - GENERAL: Busca híbrida em todos, rankeado por relevância
        
        Returns:
            Tuple: (doc_contexts, memory_contexts)
        """
        if not self.collection:
            return [], []
        
        try:
            # 1. Lista documentos disponíveis
            available_docs = self.get_available_documents(user_id, bot_id)
            available_sources = [d['source'] for d in available_docs]
            
            logger.info(f"[RAG] Documentos disponíveis: {available_sources}")
            
            # 2. Classifica a query
            query_type, specific_doc = self.classify_query(query_text, available_sources)
            logger.info(f"[RAG] Tipo de query: {query_type.value}, Doc específico: {specific_doc}")
            
            # 3. Executa estratégia de busca apropriada
            if query_type == QueryType.SPECIFIC and specific_doc:
                doc_contexts = self._search_specific_document(
                    query_text, user_id, bot_id, specific_doc, limit
                )
            elif query_type == QueryType.REFERENCE:
                # Usa documento mais recente (do contexto ou da lista)
                target_source = recent_doc_source or (available_docs[0]['source'] if available_docs else None)
                doc_contexts = self._search_specific_document(
                    query_text, user_id, bot_id, target_source, limit
                ) if target_source else []
            elif query_type == QueryType.COMPARATIVE:
                doc_contexts = self._search_comparative(
                    query_text, user_id, bot_id, available_sources, limit
                )
            else:  # GENERAL
                doc_contexts = self._search_general(
                    query_text, user_id, bot_id, limit
                )
            
            # 4. Busca memórias (sempre complementar)
            memory_contexts = self._search_memories(query_text, user_id, bot_id, limit=3)
            
            return doc_contexts, memory_contexts
            
        except Exception as e:
            logger.error(f"Erro em search_context: {e}")
            return [], []

    def _search_specific_document(
        self, query: str, user_id: int, bot_id: int, source: str, limit: int
    ) -> List[str]:
        """Busca em um documento específico."""
        embedding = self._get_embedding(query, "retrieval_query")
        if not embedding:
            return []
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"bot_id": str(bot_id)},
                    {"type": "document"},
                    {"source": source}
                ]
            }
        )
        
        return self._format_doc_results(results)

    def _search_comparative(
        self, query: str, user_id: int, bot_id: int, sources: List[str], limit: int
    ) -> List[str]:
        """
        Busca comparativa - garante resultados de múltiplos documentos.
        Distribui o limite entre os documentos disponíveis.
        """
        embedding = self._get_embedding(query, "retrieval_query")
        if not embedding:
            return []
        
        all_results = []
        per_doc_limit = max(2, limit // len(sources)) if sources else limit
        
        for source in sources[:4]:  # Máximo 4 documentos para comparação
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=per_doc_limit,
                where={
                    "$and": [
                        {"user_id": str(user_id)},
                        {"bot_id": str(bot_id)},
                        {"type": "document"},
                        {"source": source}
                    ]
                }
            )
            all_results.extend(self._format_doc_results(results))
        
        return all_results[:limit]

    def _search_general(
        self, query: str, user_id: int, bot_id: int, limit: int
    ) -> List[str]:
        """Busca geral em todos os documentos, rankeado por relevância."""
        embedding = self._get_embedding(query, "retrieval_query")
        if not embedding:
            return []
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"bot_id": str(bot_id)},
                    {"type": "document"}
                ]
            }
        )
        
        return self._format_doc_results(results)

    def _search_memories(
        self, query: str, user_id: int, bot_id: int, limit: int
    ) -> List[str]:
        """Busca apenas memórias de conversação."""
        embedding = self._get_embedding(query, "retrieval_query")
        if not embedding:
            return []
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"bot_id": str(bot_id)},
                    {"type": "memory"}
                ]
            }
        )
        
        contexts = []
        if results and results['documents'] and results['documents'][0]:
            for doc in results['documents'][0]:
                contexts.append(f"[MEMÓRIA]\n{doc}")
        
        return contexts

    def _format_doc_results(self, results: dict) -> List[str]:
        """Formata resultados de documentos com fonte clara."""
        contexts = []
        
        if not results or not results['documents'] or not results['documents'][0]:
            return contexts
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            source = meta.get('source', 'Documento')
            chunk_idx = meta.get('chunk_index', 0)
            total = meta.get('total_chunks', 1)
            
            # Formato claro para a IA saber de onde vem cada trecho
            header = f"[DOCUMENTO: {source} (trecho {chunk_idx + 1}/{total})]"
            contexts.append(f"{header}\n{doc}")
        
        return contexts

# Instância global singleton exportada
vector_service = VectorService()