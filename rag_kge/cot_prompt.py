"""
Chain-of-Thought (CoT) template utilities for KG error detection.

Provides:
- load_templates/load_relation_meta
- select_template: pick a CoT template based on relation + meta tags
- render_cot_instructions: render the selected template steps for prompting
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_templates(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_relation_meta(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _template_score(
    template: Dict[str, Any],
    relation_id: str,
    rel_tags: List[str],
) -> int:
    score = 0
    if relation_id in template.get("applicable_relations", []):
        score += 10
    t_patterns = set(template.get("applicable_patterns", []) or [])
    score += len(t_patterns.intersection(rel_tags))
    return score


def select_template(
    relation_id: str,
    templates: List[Dict[str, Any]],
    relation_meta: Dict[str, List[str]],
    fallback_id: str = "generic_compare_v1",
) -> Dict[str, Any]:
    rel_tags = relation_meta.get(relation_id, [])
    best = None
    best_score = -1
    for tmpl in templates:
        s = _template_score(tmpl, relation_id, rel_tags)
        if s > best_score:
            best_score = s
            best = tmpl
    if best is None:
        # fallback by id
        for tmpl in templates:
            if tmpl.get("id") == fallback_id:
                return tmpl
        # final fallback: first template
        return templates[0]
    return best


def render_cot_instructions(
    template: Dict[str, Any],
    relation_id: str,
) -> str:
    lines = []
    lines.append(f"你需要按照下述思维链步骤推理，relation={relation_id}，template_id={template.get('id','')}:")
    for idx, step in enumerate(template.get("steps", []), 1):
        lines.append(f"{idx}. {step}")
    lines.append("输出时必须给出 JSON：{label, confidence, reason, template_id, evidence_used, best_choice}。")
    lines.append("label ∈ {correct, incorrect, unknown}；如果缺少外部或内部证据，请在 reason 中明确说明。")
    return "\n".join(lines)


def load_default_assets(root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience loader using default template/meta paths under rag_kge/templates/.
    """
    root = root or Path(__file__).resolve().parent
    tmpl_path = root / "templates" / "cot_templates.json"
    meta_path = root / "templates" / "relation_meta.json"
    return {
        "templates": load_templates(tmpl_path),
        "relation_meta": load_relation_meta(meta_path),
    }

