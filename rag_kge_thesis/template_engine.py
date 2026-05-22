from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


TEMPLATE_FAMILIES = ("T_rel", "T_path", "T_ext", "T_conf")


def load_templates(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_relation_meta(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _template_score(
    template: Dict[str, Any],
    stage: str,
    relation_id: str,
    relation_tags: List[str],
    has_external: bool,
    has_alternatives: bool,
) -> int:
    if template.get("stage") != stage:
        return -10_000

    score = 0
    if relation_id in (template.get("applicable_relations") or []):
        score += 10

    template_patterns = set(template.get("applicable_patterns") or [])
    score += 2 * len(template_patterns.intersection(relation_tags))

    required = set(template.get("required_evidence") or [])
    if has_external and "external_text" in required:
        score += 1
    if has_alternatives and "alternatives" in required:
        score += 1
    return score


def select_template(
    stage: str,
    relation_id: str,
    templates: List[Dict[str, Any]],
    relation_meta: Dict[str, List[str]],
    has_external: bool = False,
    has_alternatives: bool = False,
) -> Dict[str, Any]:
    relation_tags = relation_meta.get(relation_id, [])
    best: Optional[Dict[str, Any]] = None
    best_score = -10_000

    for template in templates:
        score = _template_score(
            template=template,
            stage=stage,
            relation_id=relation_id,
            relation_tags=relation_tags,
            has_external=has_external,
            has_alternatives=has_alternatives,
        )
        if score > best_score:
            best_score = score
            best = template

    if best is not None:
        return best

    for template in templates:
        if template.get("stage") == stage and template.get("is_fallback", False):
            return template

    for template in templates:
        if template.get("stage") == stage:
            return template

    raise ValueError(f"No template found for stage={stage!r}")


def choose_primary_template_family(
    relation_id: str,
    relation_meta: Dict[str, List[str]],
    structural_signals: Optional[Dict[str, Any]] = None,
    semantic_signals: Optional[Dict[str, Any]] = None,
    has_external: bool = False,
) -> str:
    tags = relation_meta.get(relation_id, [])
    directional = any(tag in tags for tag in ("antisymmetric", "hierarchical", "compositional"))

    ss = structural_signals or {}
    sem = semantic_signals or {}

    conflict_score = _safe_float(ss.get("conflict_score"), 0.0)
    support_score = _safe_float(ss.get("support_score"), 0.0)
    self_loop = bool(ss.get("self_loop", False))
    reverse_same_relation = bool(ss.get("reverse_same_relation", False))

    support_ratio = _safe_float(sem.get("support_ratio"), 0.0)
    refute_ratio = _safe_float(sem.get("refute_ratio"), 0.0)
    evidence_consistency = _safe_float(sem.get("evidence_consistency"), 0.0)
    top_quality = _safe_float(sem.get("top_quality"), 0.0)

    if has_external and support_ratio >= 0.30 and refute_ratio >= 0.30 and top_quality >= 0.55:
        return "T_conf"

    if conflict_score >= 0.70:
        return "T_rel"

    if directional and (self_loop or reverse_same_relation):
        return "T_rel"

    if has_external and top_quality >= 0.60 and evidence_consistency >= 0.75:
        if max(support_ratio, refute_ratio) >= 0.60 and conflict_score <= 0.40:
            return "T_ext"
        if conflict_score > 0.40:
            return "T_conf"

    if support_score <= 0.35 and conflict_score <= 0.45:
        return "T_path"

    if has_external and top_quality >= 0.45:
        return "T_ext"

    if conflict_score >= 0.45:
        return "T_rel"

    return "T_path"


def _instance_template_score(
    template: Dict[str, Any],
    relation_id: str,
    relation_tags: List[str],
    has_external: bool,
    has_alternatives: bool,
) -> int:
    score = 0

    if relation_id in (template.get("applicable_relations") or []):
        score += 10

    template_patterns = set(template.get("applicable_patterns") or [])
    score += 3 * len(template_patterns.intersection(relation_tags))

    required = set(template.get("required_evidence") or [])
    if "external_text" in required:
        score += 2 if has_external else -3
    if "alternatives" in required:
        score += 2 if has_alternatives else -2
    if "structural_signals" in required:
        score += 1
    if "evidence_package" in required:
        score += 1

    if template.get("is_fallback", False):
        score += 1
    return score


def select_template_two_level(
    relation_id: str,
    templates: List[Dict[str, Any]],
    relation_meta: Dict[str, List[str]],
    structural_signals: Optional[Dict[str, Any]] = None,
    semantic_signals: Optional[Dict[str, Any]] = None,
    has_external: bool = False,
    has_alternatives: bool = False,
    primary_family: Optional[str] = None,
) -> Dict[str, Any]:
    relation_tags = relation_meta.get(relation_id, [])
    family = primary_family or choose_primary_template_family(
        relation_id=relation_id,
        relation_meta=relation_meta,
        structural_signals=structural_signals,
        semantic_signals=semantic_signals,
        has_external=has_external,
    )

    if family not in TEMPLATE_FAMILIES:
        family = "T_path"

    primary_cards = [
        t
        for t in templates
        if t.get("stage") == "cot" and t.get("level") == "family" and t.get("template_family") == family
    ]
    primary_card = primary_cards[0] if primary_cards else {
        "id": f"{family.lower()}_card_fallback",
        "template_family": family,
        "name": family,
    }

    instance_candidates = [
        t
        for t in templates
        if t.get("stage") == "cot" and t.get("level", "instance") == "instance" and t.get("template_family") == family
    ]

    best: Optional[Dict[str, Any]] = None
    best_score = -10_000
    for template in instance_candidates:
        score = _instance_template_score(
            template=template,
            relation_id=relation_id,
            relation_tags=relation_tags,
            has_external=has_external,
            has_alternatives=has_alternatives,
        )
        if score > best_score:
            best_score = score
            best = template

    if best is None:
        best = select_template(
            stage="hybrid",
            relation_id=relation_id,
            templates=templates,
            relation_meta=relation_meta,
            has_external=has_external,
            has_alternatives=has_alternatives,
        )

    routed = dict(best)
    routed["selected_primary_family"] = family
    routed["selected_primary_template_id"] = primary_card.get("id", "")
    routed["selected_primary_template_name"] = primary_card.get("name", family)

    return {
        "primary_family": family,
        "primary_template_card": primary_card,
        "instance_template": routed,
    }


def render_template_instructions(
    template: Dict[str, Any],
    stage: str,
    relation_id: str,
) -> str:
    lines = [
        f"当前推理阶段: {stage}",
        f"relation={relation_id}",
        f"template_id={template.get('id', '')}",
    ]

    family = template.get("selected_primary_family") or template.get("template_family")
    if family:
        lines.append(f"主模板类别={family}")
    if template.get("selected_primary_template_id"):
        lines.append(f"一级模板卡片={template.get('selected_primary_template_id')}")

    lines.append("请严格按照下面模板步骤推理，不要跳步。")

    for idx, step in enumerate(template.get("steps") or [], 1):
        lines.append(f"{idx}. {step}")

    decision_rule = template.get("decision_rule")
    if decision_rule:
        lines.append(f"判定规则: {decision_rule}")

    lines.append(
        "只输出一个 JSON 对象，字段必须包含: "
        "label, confidence, reason, template_id, evidence_used, best_choice, "
        "should_stop, evidence_consistency, risk_flags。"
    )
    lines.append("label 只能是 correct / incorrect / unknown。")
    lines.append("confidence 和 evidence_consistency 都必须在 [0,1]。")
    lines.append("evidence_used 和 risk_flags 用 JSON 数组。")
    lines.append('best_choice 写 "candidate"、"A1"..."Ak" 或空字符串。')
    lines.append("当当前阶段证据已足以高置信判断时，should_stop=true，否则 false。")
    return "\n".join(lines)


def load_default_assets(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parent
    template_path = root / "templates" / "prompt_templates.json"
    template_extra_path = root / "templates" / "prompt_templates_cot_extra.json"
    relation_meta_path = root / "templates" / "relation_meta.json"
    relation_meta_extra_path = root / "templates" / "relation_meta_extra.json"

    templates = load_templates(template_path)
    if template_extra_path.exists():
        try:
            extra_templates = load_templates(template_extra_path)
            if isinstance(extra_templates, list):
                templates.extend(extra_templates)
        except Exception:
            pass

    relation_meta = load_relation_meta(relation_meta_path)
    if relation_meta_extra_path.exists():
        try:
            relation_meta_extra = load_relation_meta(relation_meta_extra_path)
            if isinstance(relation_meta_extra, dict):
                relation_meta.update(relation_meta_extra)
        except Exception:
            pass

    return {
        "templates": templates,
        "relation_meta": relation_meta,
    }
