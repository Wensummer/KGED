from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import re

from .dense_retrieval import DenseRetriever
from .evidence_stance import NLIStanceClassifier


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


_DEFAULT_DENSE: DenseRetriever | None = None
_DEFAULT_NLI: NLIStanceClassifier | None = None


@dataclass
class RankedDoc:
    doc_id: str
    entity_id: str
    entity_name: str
    title: str
    text: str
    source: str
    url: str
    time: str
    authority: float
    relevance: float
    timeliness: float
    quality: float
    dense_score: float = 0.0
    stance_label: str = "neutral"
    stance_confidence: float = 0.0
    stance_support_prob: float = 0.0
    stance_refute_prob: float = 0.0
    stance_neutral_prob: float = 1.0


def authority_score(source: str) -> float:
    src = (source or "").lower()
    if src == "wordnet":
        return 1.0
    if src == "wikidata":
        return 0.95
    if src == "wikipedia":
        return 0.85
    if src.startswith("newsapi:"):
        return 0.65
    return 0.5


def timeliness_score(time_value: str) -> float:
    if not time_value:
        return 0.5

    raw = time_value.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return 0.5

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    if age_days <= 365:
        return 1.0
    if age_days <= 365 * 3:
        return 0.85
    if age_days <= 365 * 5:
        return 0.7
    if age_days <= 365 * 10:
        return 0.55
    return 0.35


def relevance_score(text: str, query_text: str, h_name: str, t_name: str) -> float:
    query_tokens = _tokenize(query_text)
    doc_tokens = _tokenize(text)
    if not query_tokens:
        overlap_score = 0.0
    else:
        overlap_score = len(set(query_tokens).intersection(doc_tokens)) / len(set(query_tokens))

    lower_text = text.lower()
    h_hit = 1.0 if h_name and h_name.lower() in lower_text else 0.0
    t_hit = 1.0 if t_name and t_name.lower() in lower_text else 0.0
    return min(1.0, 0.55 * overlap_score + 0.225 * h_hit + 0.225 * t_hit)


def _resolve_weights(weights: Sequence[float]) -> tuple[float, float, float, float]:
    if len(weights) >= 4:
        wa, wr, wt, wd = [max(0.0, float(v)) for v in weights[:4]]
    elif len(weights) == 3:
        wa, wr, wt = [max(0.0, float(v)) for v in weights]
        wd = 0.0
    else:
        wa, wr, wt, wd = 0.3, 0.35, 0.15, 0.2
    z = wa + wr + wt + wd
    if z <= 0:
        return 0.3, 0.35, 0.15, 0.2
    return wa / z, wr / z, wt / z, wd / z


def _get_default_dense_retriever() -> DenseRetriever:
    global _DEFAULT_DENSE
    if _DEFAULT_DENSE is None:
        _DEFAULT_DENSE = DenseRetriever()
    return _DEFAULT_DENSE


def _get_default_nli_classifier() -> NLIStanceClassifier:
    global _DEFAULT_NLI
    if _DEFAULT_NLI is None:
        _DEFAULT_NLI = NLIStanceClassifier()
    return _DEFAULT_NLI


def rank_external_docs_from_candidates(
    candidates,
    kb_internal,
    triple,
    query_text: str,
    top_k: int = 4,
    weights: Sequence[float] = (0.3, 0.35, 0.15, 0.2),
    dense_scores: Optional[dict[str, float]] = None,
) -> List[RankedDoc]:
    h_node = kb_internal.entities.get(triple.h_id)
    t_node = kb_internal.entities.get(triple.t_id)
    h_name = h_node.name if h_node and h_node.name else triple.h_id
    t_name = t_node.name if t_node and t_node.name else triple.t_id

    seen_doc_ids = set()
    wa, wr, wt, wd = _resolve_weights(weights)

    ranked: List[RankedDoc] = []
    for doc in candidates:
        if doc.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc.doc_id)

        authority = authority_score(doc.source)
        relevance = relevance_score(doc.text, query_text, h_name, t_name)
        timeliness = timeliness_score(doc.time)
        dense_score = _clip(float((dense_scores or {}).get(doc.doc_id, 0.0)))

        quality = _clip(wa * authority + wr * relevance + wt * timeliness + wd * dense_score)
        ranked.append(
            RankedDoc(
                doc_id=doc.doc_id,
                entity_id=doc.entity_id,
                entity_name=doc.entity_name,
                title=doc.title,
                text=doc.text,
                source=doc.source,
                url=doc.url,
                time=doc.time,
                authority=authority,
                relevance=relevance,
                timeliness=timeliness,
                dense_score=dense_score,
                quality=quality,
            )
        )

    ranked.sort(key=lambda x: x.quality, reverse=True)
    return ranked[:top_k]


def attach_stance(
    docs: Iterable[RankedDoc],
    claim_text: str,
    nli_classifier: NLIStanceClassifier,
    max_chars: int = 1200,
) -> List[RankedDoc]:
    output: List[RankedDoc] = []
    claim = " ".join((claim_text or "").strip().split())
    for doc in docs:
        evidence = " ".join((doc.text or "").strip().split())
        if len(evidence) > max_chars:
            evidence = evidence[:max_chars]
        result = nli_classifier.predict(claim, evidence)
        output.append(
            replace(
                doc,
                stance_label=result.label,
                stance_confidence=_clip(result.confidence),
                stance_support_prob=_clip(result.support_prob),
                stance_refute_prob=_clip(result.refute_prob),
                stance_neutral_prob=_clip(result.neutral_prob),
            )
        )
    return output


def rank_external_docs(
    kb_external,
    kb_internal,
    triple,
    query_text: str,
    top_k: int = 4,
    max_candidates: int = 20,
    weights: Sequence[float] = (0.3, 0.35, 0.15, 0.2),
    retrieval_mode: str = "hybrid",
    dense_retriever: Optional[DenseRetriever] = None,
    claim_text: str = "",
    nli_classifier: Optional[NLIStanceClassifier] = None,
) -> List[RankedDoc]:
    candidates = kb_external.rank_docs_for_entities(
        [triple.h_id, triple.t_id],
        query_text=query_text,
        top_k=max_candidates,
    )

    mode = (retrieval_mode or "hybrid").lower()
    dense_scores: dict[str, float] = {}
    if mode in {"dense", "hybrid"} and candidates:
        retriever = dense_retriever if dense_retriever is not None else _get_default_dense_retriever()
        ranked_dense = retriever.rank(query_text=query_text, docs=candidates, top_k=max_candidates)
        dense_scores = {doc.doc_id: score for doc, score in ranked_dense}
        if mode == "dense":
            candidates = [doc for doc, _ in ranked_dense]
        else:
            candidates = sorted(candidates, key=lambda d: dense_scores.get(d.doc_id, 0.0), reverse=True)

    ranked = rank_external_docs_from_candidates(
        candidates=candidates,
        kb_internal=kb_internal,
        triple=triple,
        query_text=query_text,
        top_k=top_k,
        weights=weights,
        dense_scores=dense_scores,
    )

    if claim_text.strip():
        classifier = nli_classifier if nli_classifier is not None else _get_default_nli_classifier()
        ranked = attach_stance(ranked, claim_text=claim_text, nli_classifier=classifier)

    return ranked


def aggregate_semantic_signals(docs: Iterable[RankedDoc]) -> dict:
    support_mass = 0.0
    refute_mass = 0.0
    neutral_mass = 0.0
    support_count = 0
    refute_count = 0
    neutral_count = 0
    top_quality = 0.0
    top_dense = 0.0

    doc_list = list(docs)
    for doc in doc_list:
        q = _clip(doc.quality)
        top_quality = max(top_quality, q)
        top_dense = max(top_dense, _clip(doc.dense_score))

        label = (doc.stance_label or "neutral").lower()
        conf = _clip(doc.stance_confidence)
        mass = q * max(0.2, conf)
        if label == "support":
            support_mass += mass
            support_count += 1
        elif label == "refute":
            refute_mass += mass
            refute_count += 1
        else:
            neutral_mass += mass
            neutral_count += 1

    total = support_mass + refute_mass + neutral_mass
    directional = support_mass + refute_mass

    support_ratio = support_mass / total if total > 0 else 0.0
    refute_ratio = refute_mass / total if total > 0 else 0.0
    neutral_ratio = neutral_mass / total if total > 0 else 0.0

    if directional <= 1e-9:
        evidence_consistency = 0.5
        support_refute_gap = 0.0
    else:
        evidence_consistency = max(support_mass, refute_mass) / directional
        support_refute_gap = abs(support_mass - refute_mass) / directional

    return {
        "doc_count": len(doc_list),
        "support_count": support_count,
        "refute_count": refute_count,
        "neutral_count": neutral_count,
        "support_mass": round(support_mass, 4),
        "refute_mass": round(refute_mass, 4),
        "neutral_mass": round(neutral_mass, 4),
        "support_ratio": round(support_ratio, 4),
        "refute_ratio": round(refute_ratio, 4),
        "neutral_ratio": round(neutral_ratio, 4),
        "evidence_consistency": round(_clip(evidence_consistency), 4),
        "support_refute_gap": round(_clip(support_refute_gap), 4),
        "top_quality": round(top_quality, 4),
        "top_dense": round(top_dense, 4),
    }


def render_ranked_docs(docs: Iterable[RankedDoc], max_chars: int = 1800) -> str:
    lines: List[str] = []
    total = 0

    for idx, doc in enumerate(docs, 1):
        header = (
            f"{idx}. [Q={doc.quality:.2f} A={doc.authority:.2f} "
            f"R={doc.relevance:.2f} T={doc.timeliness:.2f} D={doc.dense_score:.2f}] "
            f"[stance={doc.stance_label}:{doc.stance_confidence:.2f}] "
            f"[{doc.source}] {doc.title} (entity={doc.entity_name})"
        )
        body = doc.text.strip().replace("\n", " ")
        piece = header + "\n" + body
        if lines and total + len(piece) > max_chars:
            break
        lines.append(piece)
        total += len(piece) + 1
    return "\n\n".join(lines)


def docs_as_records(docs: Iterable[RankedDoc]) -> List[dict]:
    return [
        {
            "doc_id": doc.doc_id,
            "entity_id": doc.entity_id,
            "entity_name": doc.entity_name,
            "title": doc.title,
            "source": doc.source,
            "url": doc.url,
            "time": doc.time,
            "authority": round(doc.authority, 4),
            "relevance": round(doc.relevance, 4),
            "timeliness": round(doc.timeliness, 4),
            "dense_score": round(doc.dense_score, 4),
            "quality": round(doc.quality, 4),
            "stance_label": doc.stance_label,
            "stance_confidence": round(doc.stance_confidence, 4),
            "stance_support_prob": round(doc.stance_support_prob, 4),
            "stance_refute_prob": round(doc.stance_refute_prob, 4),
            "stance_neutral_prob": round(doc.stance_neutral_prob, 4),
        }
        for doc in docs
    ]
