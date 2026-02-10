from enum import Enum
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class EvidenceDecision(Enum):
    ANSWER = "ANSWER"
    REFUSE = "REFUSE"
    UNCERTAIN = "UNCERTAIN"

class EvidenceGate:
    """
    Decide deterministicamente se há evidência suficiente para responder em modo estrito.
    Evita alucinações e inconsistências.
    """

    @staticmethod
    def evaluate(doc_contexts: List[Dict], threshold_strict: float = 0.45) -> Tuple[EvidenceDecision, str]:
        """
        Avalia a qualidade do contexto retornado.

        Args:
            doc_contexts: Lista de chunks com 'score' (distância cosine, menor é melhor).
            threshold_strict: Threshold base (ex: 0.45).

        Returns:
            (Decision, Reason)
        """
        if not doc_contexts:
            return EvidenceDecision.REFUSE, "no_context"

        # 1. Extrair métricas
        scores = [d.get('score', 1.0) for d in doc_contexts]
        best_score = min(scores)
        unique_sources = len(set(d.get('source_id') for d in doc_contexts))

        # Média dos top 4 (ou menos)
        top_k = scores[:4]
        avg_score = sum(top_k) / len(top_k) if top_k else 1.0

        # Gap entre melhor e segundo melhor
        gap = 0.0
        if len(scores) > 1:
            sorted_scores = sorted(scores)
            gap = sorted_scores[1] - sorted_scores[0]

        # 2. Definir Limites Dinâmicos
        # T_answer: Se for melhor que isso, responde seguro.
        T_answer = threshold_strict * 0.92

        # T_refuse: Se for pior que isso, recusa direto.
        T_refuse = threshold_strict * 1.08

        reason_debug = f"best={best_score:.3f}, avg={avg_score:.3f}, gap={gap:.3f}, uniq={unique_sources}, T_ans={T_answer:.3f}, T_ref={T_refuse:.3f}"

        # 3. Árvore de Decisão

        # A) Zona Clara de Resposta
        if best_score <= T_answer:
            return EvidenceDecision.ANSWER, f"strong_match ({reason_debug})"

        # B) Zona Clara de Recusa
        if best_score >= T_refuse:
            return EvidenceDecision.REFUSE, f"weak_match ({reason_debug})"

        # C) Zona Cinzenta (Gray Zone)
        # Tenta salvar se houver robustez (várias fontes ou consistência)

        # C.1) Várias fontes concordam (heurística: se trouxe chunks de 2+ docs, provavelmente é um tema recorrente)
        if unique_sources >= 2:
            return EvidenceDecision.ANSWER, f"gray_zone_multi_source ({reason_debug})"

        # C.2) Texto muito curto (falso positivo comum em headers/footers)
        first_chunk_len = len(doc_contexts[0].get('content', ''))
        if first_chunk_len < 200:
            return EvidenceDecision.REFUSE, f"gray_zone_short_text ({reason_debug})"

        # C.3) Consistência (gap pequeno e média aceitável)
        # Se avg está dentro do threshold original e os chunks são parecidos (gap baixo)
        if avg_score <= threshold_strict and gap <= 0.03:
            return EvidenceDecision.ANSWER, f"gray_zone_consistent ({reason_debug})"

        # C.4) Incerteza (Default Refuse ou Uncertain para tratamento posterior)
        # Por segurança no strict mode, default é REFUSE se não caiu nas exceções acima.
        return EvidenceDecision.REFUSE, f"gray_zone_fallback ({reason_debug})"

evidence_gate = EvidenceGate()
