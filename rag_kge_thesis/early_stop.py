from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class StructuralSignals:
    self_loop: bool
    reverse_same_relation: bool
    same_rel_head_count: int
    same_rel_tail_count: int
    two_hop_path_count: int
    kge_prior: Optional[float]
    conflict_score: float
    support_score: float
    relation_tags: List[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.kge_prior is not None:
            data["kge_prior"] = round(self.kge_prior, 4)
        data["conflict_score"] = round(self.conflict_score, 4)
        data["support_score"] = round(self.support_score, 4)
        return data


@dataclass
class EarlyStopDecision:
    stage: str
    label: str
    confidence: float
    reason: str
    risk_flags: List[str]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _count_two_hop_paths(kb_internal, h_id: str, t_id: str) -> int:
    out_mid = {e.t_id for e in kb_internal.out_neighbors.get(h_id, [])}
    in_mid = {e.h_id for e in kb_internal.in_neighbors.get(t_id, [])}
    return len(out_mid.intersection(in_mid))


def build_structural_signals(
    kb_internal,
    triple,
    relation_tags: List[str],
    kge_prior: Optional[float] = None,
) -> StructuralSignals:
    self_loop = triple.h_id == triple.t_id
    reverse_same_relation = any(
        e.r_id == triple.r_id and e.t_id == triple.h_id
        for e in kb_internal.out_neighbors.get(triple.t_id, [])
    )
    same_rel_head_count = sum(
        1
        for e in kb_internal.out_neighbors.get(triple.h_id, [])
        if e.r_id == triple.r_id and e.t_id != triple.t_id
    )
    same_rel_tail_count = sum(
        1
        for e in kb_internal.in_neighbors.get(triple.t_id, [])
        if e.r_id == triple.r_id and e.h_id != triple.h_id
    )
    two_hop_path_count = _count_two_hop_paths(kb_internal, triple.h_id, triple.t_id)

    antisymmetric = any(
        tag in relation_tags for tag in ("antisymmetric", "hierarchical", "compositional")
    )

    conflict_score = 0.0
    if self_loop:
        conflict_score += 0.7
    if reverse_same_relation:
        conflict_score += 0.7 if antisymmetric else 0.25
    if kge_prior is not None:
        conflict_score += 0.45 * _clip(kge_prior)

    support_score = 0.0
    support_score += min(0.3, 0.1 * same_rel_head_count)
    support_score += min(0.25, 0.1 * same_rel_tail_count)
    support_score += min(0.2, 0.05 * two_hop_path_count)
    if kge_prior is not None:
        support_score += 0.35 * (1.0 - _clip(kge_prior))
    else:
        support_score += 0.175

    if self_loop:
        support_score -= 0.25
    if reverse_same_relation and antisymmetric:
        support_score -= 0.2

    return StructuralSignals(
        self_loop=self_loop,
        reverse_same_relation=reverse_same_relation,
        same_rel_head_count=same_rel_head_count,
        same_rel_tail_count=same_rel_tail_count,
        two_hop_path_count=two_hop_path_count,
        kge_prior=kge_prior,
        conflict_score=_clip(conflict_score),
        support_score=_clip(support_score),
        relation_tags=list(relation_tags),
    )


def heuristic_structural_stop(
    signals: StructuralSignals,
    incorrect_threshold: float = 0.92,
    correct_threshold: float = 0.9,
    allow_correct: bool = False,
) -> Optional[EarlyStopDecision]:
    flags: List[str] = []
    if signals.self_loop:
        flags.append("self_loop")
    if signals.reverse_same_relation:
        flags.append("reverse_same_relation")
    if signals.kge_prior is not None and signals.kge_prior >= 0.85:
        flags.append("high_kge_prior")
    if signals.self_loop and any(
        tag in signals.relation_tags for tag in ("antisymmetric", "hierarchical", "compositional")
    ):
        flags.append("antisymmetric_self_loop")
        return EarlyStopDecision(
            stage="structural_heuristic",
            label="incorrect",
            confidence=max(0.93, signals.conflict_score),
            reason="候选三元组在具有方向性或层级约束的关系上形成自环，属于高风险结构冲突，直接早停。",
            risk_flags=flags,
        )

    if signals.conflict_score >= incorrect_threshold:
        return EarlyStopDecision(
            stage="structural_heuristic",
            label="incorrect",
            confidence=signals.conflict_score,
            reason=(
                "结构证据已出现高风险冲突，包含自环/反向同关系或较高的结构异常先验，"
                "因此直接判为可疑错误三元组。"
            ),
            risk_flags=flags,
        )

    if allow_correct and signals.support_score >= correct_threshold and signals.conflict_score <= 0.15:
        return EarlyStopDecision(
            stage="structural_heuristic",
            label="correct",
            confidence=signals.support_score,
            reason="结构支持明显且无高风险冲突，因此直接接受该三元组。",
            risk_flags=flags,
        )

    return None


def heuristic_external_confirmation_stop(
    signals: StructuralSignals,
    semantic_signals: Optional[Dict],
    consistency_threshold: float = 0.85,
    support_ratio_threshold: float = 0.60,
    top_quality_threshold: float = 0.70,
    max_conflict_for_confirm: float = 0.35,
    max_refute_ratio: float = 0.20,
) -> Optional[EarlyStopDecision]:
    if not semantic_signals:
        return None

    support_ratio = _safe_float(semantic_signals.get("support_ratio"), 0.0)
    refute_ratio = _safe_float(semantic_signals.get("refute_ratio"), 0.0)
    consistency = _safe_float(semantic_signals.get("evidence_consistency"), 0.0)
    top_quality = _safe_float(semantic_signals.get("top_quality"), 0.0)
    support_count = int(_safe_float(semantic_signals.get("support_count"), 0.0))

    if signals.conflict_score > max_conflict_for_confirm:
        return None
    if refute_ratio > max_refute_ratio:
        return None
    if support_ratio < support_ratio_threshold:
        return None
    if consistency < consistency_threshold:
        return None
    if top_quality < top_quality_threshold:
        return None
    if support_count <= 0:
        return None

    confidence = _clip(0.55 * consistency + 0.30 * support_ratio + 0.15 * top_quality)
    flags = [
        "external_high_consistency",
        "external_support_dominant",
        "internal_low_conflict",
    ]
    return EarlyStopDecision(
        stage="external_confirmation",
        label="correct",
        confidence=max(confidence, consistency_threshold),
        reason="外部高质量证据在支持方向上形成稳定一致，且内部未发现显著冲突，触发外部确认早停。",
        risk_flags=flags,
    )


def render_structural_signals(signals: StructuralSignals) -> str:
    lines = [
        f"- relation_tags: {', '.join(signals.relation_tags) if signals.relation_tags else 'none'}",
        f"- self_loop: {signals.self_loop}",
        f"- reverse_same_relation: {signals.reverse_same_relation}",
        f"- same_rel_head_count: {signals.same_rel_head_count}",
        f"- same_rel_tail_count: {signals.same_rel_tail_count}",
        f"- two_hop_path_count: {signals.two_hop_path_count}",
        f"- kge_prior: {signals.kge_prior if signals.kge_prior is not None else 'n/a'}",
        f"- conflict_score: {signals.conflict_score:.3f}",
        f"- support_score: {signals.support_score:.3f}",
    ]
    return "\n".join(lines)


def should_stop_after_llm(
    parsed: Optional[dict],
    confidence_threshold: float,
    consistency_threshold: Optional[float] = None,
) -> bool:
    if not isinstance(parsed, dict):
        return False

    label = str(parsed.get("label", "")).lower()
    if label not in {"correct", "incorrect"}:
        return False

    confidence = _safe_float(parsed.get("confidence"), 0.0)
    if confidence < confidence_threshold:
        return False

    if consistency_threshold is None:
        return True

    consistency = _safe_float(parsed.get("evidence_consistency"), confidence)
    return consistency >= consistency_threshold
