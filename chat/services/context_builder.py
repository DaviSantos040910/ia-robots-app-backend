# chat/services/context_builder.py
"""
Construtor de contexto e system instructions para a IA.
Suporta múltiplos documentos com citação de fonte.
"""

from typing import List, Tuple, Optional
from datetime import datetime
from ..models import ChatMessage
import logging

logger = logging.getLogger(__name__)


def build_conversation_history(
    chat_id: int,
    limit: int = 15,
    exclude_message_id: Optional[int] = None
) -> Tuple[List[dict], List[str]]:
    """
    Constrói histórico LINEAR das últimas N mensagens.
    """
    queryset = ChatMessage.objects.filter(chat_id=chat_id)

    if exclude_message_id:
        queryset = queryset.exclude(id=exclude_message_id)

    messages = queryset.order_by('-created_at')[:limit]
    messages = list(reversed(messages))  # Ordem cronológica

    gemini_history: List[dict] = []
    recent_texts: List[str] = []

    for msg in messages:
        if not msg.content:
            continue
        if "unexpected error" in msg.content.lower():
            continue

        role = 'user' if msg.role == 'user' else 'model'

        gemini_history.append({
            "role": role,
            "parts": [{"text": msg.content}]
        })
        recent_texts.append(msg.content)

    return gemini_history, recent_texts


def get_recent_attachment_context(chat_id: int) -> Optional[str]:
    """
    Retorna o nome do arquivo anexado mais recentemente no chat.
    Útil para resolver "esse documento", "isso", etc.
    """
    recent_attachment = (
        ChatMessage.objects
        .filter(
            chat_id=chat_id,
            attachment__isnull=False,
            attachment_type='file'
        )
        .order_by('-created_at')
        .first()
    )
    
    return recent_attachment.original_filename if recent_attachment else None


def build_system_instruction(
    bot_prompt: str,
    user_name: str,
    doc_contexts: List[str],
    memory_contexts: List[str],
    current_time: str,
    available_docs: Optional[List[str]] = None
) -> str:
    """
    Constrói system instruction otimizado para RAG multi-documento.
    
    Args:
        bot_prompt: Prompt do personagem/bot
        user_name: Nome do usuário
        doc_contexts: Lista de trechos de documentos formatados
        memory_contexts: Lista de memórias formatadas
        current_time: Data/hora atual
        available_docs: Lista de nomes de documentos disponíveis (ordenados por recência)
    """
    
    # Lista de documentos disponíveis
    docs_list_section = ""
    if available_docs:
        docs_list = "\n".join(f"  {i+1}. {doc}" for i, doc in enumerate(available_docs))
        docs_list_section = f"""
## DOCUMENTOS DO USUÁRIO
Arquivos enviados (do mais recente ao mais antigo):
{docs_list}
"""

    # Seção de conteúdo dos documentos
    knowledge_section = ""
    if doc_contexts:
        knowledge_section = f"""
## TRECHOS RELEVANTES DOS DOCUMENTOS
{chr(10).join(doc_contexts)}
"""

    # Seção de memória pessoal
    memory_section = ""
    if memory_contexts:
        memory_section = f"""
## MEMÓRIA PESSOAL
Contexto sobre {user_name} e conversas anteriores:
{chr(10).join(memory_contexts)}
"""

    return f"""# PERSONAGEM
{bot_prompt}

## CONTEXTO ATUAL
- Conversando com: {user_name}
- Data/Hora: {current_time}
{docs_list_section}
{knowledge_section}
{memory_section}
## DIRETRIZES PARA DOCUMENTOS
1. **SEMPRE CITE A FONTE** - Ao usar informação de um documento, diga: "De acordo com [nome_do_arquivo]..." ou "No documento [nome]..."
2. **REFERÊNCIAS PRONOMINAIS** - Se o usuário perguntar "o que é isso?", "resuma isso", etc. sem especificar, refira-se ao documento MAIS RECENTE da lista (item 1).
3. **COMPARAÇÕES** - Se pedirem para comparar documentos, analise cada um separadamente e depois compare.
4. **MÚLTIPLOS DOCUMENTOS** - Se a resposta envolver mais de um documento, organize por fonte.
5. **DOCUMENTO ESPECÍFICO** - Se o usuário mencionar um arquivo pelo nome, foque nele.
6. **SEM DOCUMENTO** - Se não houver documentos ou a pergunta não for sobre eles, responda normalmente.

## DIRETRIZES GERAIS
1. **MANTENHA O PERSONAGEM** - Você É o personagem definido acima.
2. **SEJA CONCISO** - Responda de forma natural e direta.
3. **NÃO REPITA** - Evite repetir informações já ditas.
4. **FORMATAÇÃO** - Use Markdown apenas quando ajudar na clareza."""
