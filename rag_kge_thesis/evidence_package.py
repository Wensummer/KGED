from __future__ import annotations

from typing import Dict, Iterable, List

from .early_stop import StructuralSignals
from .evidence_quality import RankedDoc, aggregate_semantic_signals


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_structural_vector(signals: StructuralSignals) -> Dict[str, float]:
    return {
        "conflict_score": round(_clip(signals.conflict_score), 4),
        "support_score": round(_clip(signals.support_score), 4),
        "self_loop": 1.0 if signals.self_loop else 0.0,
        "reverse_same_relation": 1.0 if signals.reverse_same_relation else 0.0,
        "same_rel_head_norm": round(min(1.0, signals.same_rel_head_count / 5.0), 4),
        "same_rel_tail_norm": round(min(1.0, signals.same_rel_tail_count / 5.0), 4),
        "two_hop_norm": round(min(1.0, signals.two_hop_path_count / 8.0), 4),
        "kge_prior": round(_clip(signals.kge_prior if signals.kge_prior is not None else 0.5), 4),
    }


def _pick_key_semantic_evidence(docs: List[RankedDoc], max_items: int) -> List[Dict]:
    picked: List[Dict] = []
    by_label = {
        "support": [doc for doc in docs if doc.stance_label == "support"],
        "refute": [doc for doc in docs if doc.stance_label == "refute"],
        "neutral": [doc for doc in docs if doc.stance_label == "neutral"],
    }

    for label in ("support", "refute", "neutral"):
        if not by_label[label]:
            continue
        top = by_label[label][0]
        picked.append(
            {
                "type": "external_doc",
                "stance": label,
                "doc_id": top.doc_id,
                "title": top.title,
                "source": top.source,
                "quality": round(top.quality, 4),
                "dense_score": round(top.dense_score, 4),
                "stance_confidence": round(top.stance_confidence, 4),
            }
        )

    if len(picked) < max_items:
        used = {item["doc_id"] for item in picked}
        for doc in docs:
            if doc.doc_id in used:
                continue
            picked.append(
                {
                    "type": "external_doc",
                    "stance": doc.stance_label,
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "quality": round(doc.quality, 4),
                    "dense_score": round(doc.dense_score, 4),
                    "stance_confidence": round(doc.stance_confidence, 4),
                }
            )
            used.add(doc.doc_id)
            if len(picked) >= max_items:
                break

    return picked[:max_items]


def build_unified_evidence_package(
    structural_signals: StructuralSignals,
    ranked_docs: Iterable[RankedDoc],
    max_key_docs: int = 4,
) -> Dict:
    docs = list(ranked_docs)
    structural_vector = build_structural_vector(structural_signals)
    semantic_vector = aggregate_semantic_signals(docs)

    unified_names = list(structural_vector.keys()) + [
        "support_ratio",
        "refute_ratio",
        "neutral_ratio",
        "evidence_consistency",
        "top_quality",
        "top_dense",
        "doc_count_norm",
    ]
    unified_values = [
        float(structural_vector[name])
        for name in structural_vector.keys()
    ]
    unified_values.extend(
        [
            _clip(float(semantic_vector.get("support_ratio", 0.0))),
            _clip(float(semantic_vector.get("refute_ratio", 0.0))),
            _clip(float(semantic_vector.get("neutral_ratio", 0.0))),
            _clip(float(semantic_vector.get("evidence_consistency", 0.0))),
            _clip(float(semantic_vector.get("top_quality", 0.0))),
            _clip(float(semantic_vector.get("top_dense", 0.0))),
            _clip(min(1.0, float(semantic_vector.get("doc_count", 0)) / 8.0)),
        ]
    )

    structural_evidence = [
        {
            "type": "structural_signal",
            "name": "self_loop",
            "value": 1.0 if structural_signals.self_loop else 0.0,
        },
        {
            "type": "structural_signal",
            "name": "reverse_same_relation",
            "value": 1.0 if structural_signals.reverse_same_relation else 0.0,
        },
        {
            "type": "structural_signal",
            "name": "conflict_score",
            "value": round(structural_signals.conflict_score, 4),
        },
        {
            "type": "structural_signal",
            "name": "support_score",
            "value": round(structural_signals.support_score, 4),
        },
    ]

    semantic_evidence = _pick_key_semantic_evidence(docs, max_items=max_key_docs)

    return {
        "structural_vector": structural_vector,
        "semantic_vector": semantic_vector,
        "unified_vector": {
            "names": unified_names,
            "values": [round(v, 4) for v in unified_values],
        },
        "evidence_consistency": round(float(semantic_vector.get("evidence_consistency", 0.0)), 4),
        "key_structural_evidence": structural_evidence,
        "key_semantic_evidence": semantic_evidence,
    }


def render_evidence_package(evidence_package: Dict, max_chars: int = 900) -> str:
    if not isinstance(evidence_package, dict):
        return ""

    structural = evidence_package.get("structural_vector") or {}
    semantic = evidence_package.get("semantic_vector") or {}

    lines: List[str] = [
        "[统一证据向量]",
        (
            "- structural: "
            f"conflict={float(structural.get('conflict_score', 0.0)):.2f}, "
            f"support={float(structural.get('support_score', 0.0)):.2f}, "
            f"self_loop={int(float(structural.get('self_loop', 0.0)) >= 0.5)}, "
            f"reverse={int(float(structural.get('reverse_same_relation', 0.0)) >= 0.5)}"
        ),
        (
            "- semantic: "
            f"support_ratio={float(semantic.get('support_ratio', 0.0)):.2f}, "
            f"refute_ratio={float(semantic.get('refute_ratio', 0.0)):.2f}, "
            f"neutral_ratio={float(semantic.get('neutral_ratio', 0.0)):.2f}, "
            f"consistency={float(semantic.get('evidence_consistency', 0.0)):.2f}"
        ),
    ]

    key_semantic = evidence_package.get("key_semantic_evidence") or []
    if key_semantic:
        lines.append("- key_external_evidence:")
        for item in key_semantic[:3]:
            lines.append(
                f"  * [{item.get('stance', 'neutral')}] "
                f"Q={float(item.get('quality', 0.0)):.2f} "
                f"D={float(item.get('dense_score', 0.0)):.2f} "
                f"{item.get('title', '')} ({item.get('source', '')})"
            )

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
