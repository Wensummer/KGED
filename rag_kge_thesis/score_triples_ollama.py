from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional


from rag_kge.context_builder import Triple, build_query_text, get_relation_definition
from rag_kge.external_rag import ExternalTextKB
from rag_kge.internal_graphrag import InternalGraphKB

from .early_stop import (
    build_structural_signals,
    heuristic_structural_stop,
    heuristic_external_confirmation_stop,
    render_structural_signals,
    should_stop_after_llm,
)
from .evidence_package import build_unified_evidence_package, render_evidence_package
from .evidence_quality import (
    aggregate_semantic_signals,
    docs_as_records,
    rank_external_docs,
    render_ranked_docs,
)
from .evidence_stance import NLIStanceClassifier
from .dense_retrieval import DenseRetriever
from .llm_runtime import BaseLLMClient, build_llm_client
from .relation_tagging import infer_relation_tags, merge_relation_tags
from .template_engine import (
    load_default_assets,
    render_template_instructions,
    select_template,
    select_template_two_level,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thesis-oriented KG triple scoring with staged templates and early stopping."
    )
    parser.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr", "fb15k-237", "nell995"])
    parser.add_argument("--internal_kb_dir", type=str, default="graph_kb")
    parser.add_argument("--edges_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample_size", type=int, default=20, help="Use -1 to score all.")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--llm_provider", type=str, default="ollama", choices=["ollama", "openai_compatible"])
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate")
    parser.add_argument("--llm_api_base", type=str, default="", help="Used when llm_provider=openai_compatible.")
    parser.add_argument("--llm_api_key", type=str, default="", help="Used when llm_provider=openai_compatible.")
    parser.add_argument("--llm_timeout", type=int, default=180)
    parser.add_argument("--llm_max_tokens", type=int, default=0, help="0 means provider default.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--max_edges", type=int, default=40)
    parser.add_argument("--max_external_candidates", type=int, default=20)
    parser.add_argument("--top_k_external", type=int, default=4)
    parser.add_argument("--num_alternatives", type=int, default=6)
    parser.add_argument("--alt_pool_size", type=int, default=40)
    parser.add_argument("--kge_model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--struct_confidence_stop", type=float, default=0.9)
    parser.add_argument("--semantic_confidence_stop", type=float, default=0.85)
    parser.add_argument("--semantic_consistency_stop", type=float, default=0.8)
    parser.add_argument("--enable_structural_heuristic_stop", action="store_true")
    parser.add_argument("--retrieval_mode", type=str, default="hybrid", choices=["lexical", "dense", "hybrid"])
    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense_batch_size", type=int, default=24)
    parser.add_argument("--dense_max_chars", type=int, default=1200)
    parser.add_argument("--dense_weight", type=float, default=0.2)
    parser.add_argument("--nli_model", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--nli_neutral_threshold", type=float, default=0.45)
    parser.add_argument("--disable_nli_stance", action="store_true")
    parser.add_argument("--evidence_package_chars", type=int, default=900)
    parser.add_argument("--disable_two_level_template_select", action="store_true")
    parser.add_argument("--enable_external_confirmation_stop", action="store_true")
    parser.add_argument("--external_confirm_threshold", type=float, default=0.85)
    parser.add_argument("--external_support_threshold", type=float, default=0.60)
    parser.add_argument("--external_top_quality_threshold", type=float, default=0.70)
    parser.add_argument("--external_max_conflict", type=float, default=0.35)
    return parser.parse_args()


def load_edges(path: Path) -> List[Dict]:
    edges: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edges.append(json.loads(line))
    return edges


def load_scored_triples(path: Path) -> set[tuple[str, str, str]]:
    scored: set[tuple[str, str, str]] = set()
    if not path.exists():
        return scored
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            h_id = obj.get("h_id")
            r_id = obj.get("r_id")
            t_id = obj.get("t_id")
            if h_id is None or r_id is None or t_id is None:
                continue
            scored.add((str(h_id), str(r_id), str(t_id)))
    return scored



def extract_json_from_text(text: str) -> Optional[Dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _load_kge_model(path: Path) -> Optional[Dict]:
    try:
        import torch
    except Exception:
        return None

    ckpt = torch.load(path, map_location="cpu")
    if "ent_weight" in ckpt and "rel_weight" in ckpt:
        return {
            "type": "distmult",
            "ent2idx": ckpt["ent2idx"],
            "rel2idx": ckpt["rel2idx"],
            "ent_weight": ckpt["ent_weight"],
            "rel_weight": ckpt["rel_weight"],
        }
    if all(k in ckpt for k in ("ent_re", "ent_im", "rel_re", "rel_im")):
        return {
            "type": "complex",
            "ent2idx": ckpt["ent2idx"],
            "rel2idx": ckpt["rel2idx"],
            "ent_re": ckpt["ent_re"],
            "ent_im": ckpt["ent_im"],
            "rel_re": ckpt["rel_re"],
            "rel_im": ckpt["rel_im"],
        }
    return None


def _kge_plausibility(kge_model: Optional[Dict], h_id: str, r_id: str, t_id: str) -> float:
    if kge_model is None:
        return 0.5
    ent2idx = kge_model["ent2idx"]
    rel2idx = kge_model["rel2idx"]
    if h_id not in ent2idx or t_id not in ent2idx or r_id not in rel2idx:
        return 0.5
    try:
        import torch
    except Exception:
        return 0.5

    if kge_model["type"] == "distmult":
        ent_w = kge_model["ent_weight"]
        rel_w = kge_model["rel_weight"]
        h = ent_w[ent2idx[h_id]]
        r = rel_w[rel2idx[r_id]]
        t = ent_w[ent2idx[t_id]]
        score = (h * r * t).sum()
        return float(torch.sigmoid(score).item())

    ent_re = kge_model["ent_re"]
    ent_im = kge_model["ent_im"]
    rel_re = kge_model["rel_re"]
    rel_im = kge_model["rel_im"]
    h_re = ent_re[ent2idx[h_id]]
    h_im = ent_im[ent2idx[h_id]]
    r_re_v = rel_re[rel2idx[r_id]]
    r_im_v = rel_im[rel2idx[r_id]]
    t_re = ent_re[ent2idx[t_id]]
    t_im = ent_im[ent2idx[t_id]]
    score = (
        h_re * r_re_v * t_re
        + h_im * r_re_v * t_im
        + h_re * r_im_v * t_im
        - h_im * r_im_v * t_re
    ).sum()
    return float(torch.sigmoid(score).item())


def _resolve_kge_path(root: Path, dataset: str, kge_model_path: Optional[str]) -> Optional[Path]:
    if kge_model_path:
        path = Path(kge_model_path)
        return path if path.exists() else None

    candidates = [
        root / "rag_kge" / "output" / dataset / "complex.pt",
        root / "rag_kge" / "output" / dataset / "distmult.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_alternatives(
    kb_internal: InternalGraphKB,
    triple: Triple,
    num_alternatives: int,
    alt_pool_size: int,
    seed: int,
) -> List[str]:
    pool = kb_internal.get_alternative_pool(
        h_id=triple.h_id,
        r_id=triple.r_id,
        t_id=triple.t_id,
        max_pool=max(alt_pool_size, num_alternatives),
        seed=seed,
    )
    if len(pool) < num_alternatives:
        filler = kb_internal.get_alternative_triples(
            h_id=triple.h_id,
            r_id=triple.r_id,
            t_id=triple.t_id,
            max_alternatives=max(num_alternatives, alt_pool_size),
            seed=seed,
        )
        seen = set(pool)
        for candidate in filler:
            if candidate in seen:
                continue
            pool.append(candidate)
            seen.add(candidate)
            if len(pool) >= num_alternatives:
                break

    lines = []
    for idx, (h_id, r_id, t_id) in enumerate(pool[:num_alternatives], 1):
        lines.append(
            f"A{idx}: {kb_internal.format_triple(h_id, r_id, t_id)} "
            f"[ids=({h_id}, {r_id}, {t_id})]"
        )
    return lines


def _ensure_json_defaults(parsed: Optional[Dict], template_id: str) -> Optional[Dict]:
    if not isinstance(parsed, dict):
        return parsed
    parsed.setdefault("template_id", template_id)
    parsed.setdefault("evidence_used", [])
    parsed.setdefault("best_choice", "")
    parsed.setdefault("should_stop", False)
    parsed.setdefault("evidence_consistency", parsed.get("confidence", 0.0))
    parsed.setdefault("risk_flags", [])
    return parsed


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _truncate_text(text: str, max_chars: int = 140) -> str:
    text = " ".join((text or "").strip().split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_alternative_lookup(alternatives_text: List[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for line in alternatives_text:
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        lookup[key.strip()] = rest.strip()
    return lookup


def _build_claim_text(
    kb_internal: InternalGraphKB,
    triple: Triple,
) -> str:
    h = kb_internal.entities.get(triple.h_id)
    t = kb_internal.entities.get(triple.t_id)
    r = kb_internal.relations.get(triple.r_id)

    h_name = h.name if h and h.name else triple.h_id
    t_name = t.name if t and t.name else triple.t_id
    r_name = r.name if r and r.name else triple.r_id
    rel_text = " ".join(str(r_name).replace("_", " ").replace("/", " ").split())
    if not rel_text:
        rel_text = triple.r_id
    return f"{h_name} {rel_text} {t_name}."


def build_evidence_chain_summary(
    structural_signals,
    ranked_docs,
    stage_outputs: List[Dict],
    final_parsed: Optional[Dict],
    final_stage: str,
    heuristic,
    alternatives_text: List[str],
) -> Dict:
    chain_steps: List[Dict] = []
    key_evidence: List[Dict] = []
    summary_fragments: List[str] = []

    if structural_signals.self_loop:
        key_evidence.append(
            {
                "type": "structural_signal",
                "detail": "candidate forms a self-loop",
                "score": round(structural_signals.conflict_score, 4),
            }
        )
        summary_fragments.append("存在自环结构信号")

    if structural_signals.reverse_same_relation:
        key_evidence.append(
            {
                "type": "structural_signal",
                "detail": "reverse edge with same relation exists",
                "score": round(structural_signals.conflict_score, 4),
            }
        )
        summary_fragments.append("存在同关系反向边")

    if structural_signals.kge_prior is not None:
        key_evidence.append(
            {
                "type": "kge_prior",
                "detail": "structural anomaly prior from KGE",
                "score": round(structural_signals.kge_prior, 4),
            }
        )

    if structural_signals.same_rel_head_count > 0 or structural_signals.same_rel_tail_count > 0:
        summary_fragments.append(
            "局部图中存在同关系邻域支持"
            f"(head={structural_signals.same_rel_head_count}, tail={structural_signals.same_rel_tail_count})"
        )

    for doc in ranked_docs[:2]:
        snippet = _truncate_text(doc.text)
        key_evidence.append(
            {
                "type": "external_doc",
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "quality": round(doc.quality, 4),
                "dense_score": round(doc.dense_score, 4),
                "stance": doc.stance_label,
                "stance_confidence": round(doc.stance_confidence, 4),
                "detail": snippet,
            }
        )
    if ranked_docs:
        summary_fragments.append(f"外部证据最高质量分为 {ranked_docs[0].quality:.2f}")
        if ranked_docs[0].stance_label:
            summary_fragments.append(
                f"Top1证据态度为{ranked_docs[0].stance_label}(conf={ranked_docs[0].stance_confidence:.2f})"
            )

    if heuristic is not None:
        chain_steps.append(
            {
                "stage": heuristic.stage,
                "label": heuristic.label,
                "confidence": round(heuristic.confidence, 4),
                "stopped": True,
                "reason": heuristic.reason,
            }
        )

    for stage_output in stage_outputs:
        parsed = stage_output.get("parsed")
        chain_steps.append(
            {
                "stage": stage_output.get("stage"),
                "template_id": stage_output.get("template_id"),
                "template_family": stage_output.get("template_family", ""),
                "selected_primary_family": stage_output.get("selected_primary_family", ""),
                "selected_primary_template_id": stage_output.get("selected_primary_template_id", ""),
                "label": parsed.get("label") if isinstance(parsed, dict) else "",
                "confidence": round(_safe_float(parsed.get("confidence"), 0.0), 4)
                if isinstance(parsed, dict)
                else 0.0,
                "stopped": bool(parsed.get("should_stop", False)) if isinstance(parsed, dict) else False,
                "reason": parsed.get("reason", "") if isinstance(parsed, dict) else stage_output.get("error", ""),
            }
        )

    alt_lookup = _format_alternative_lookup(alternatives_text)
    best_choice = ""
    if isinstance(final_parsed, dict):
        best_choice = str(final_parsed.get("best_choice", "")).strip()
    if best_choice and best_choice.lower() != "candidate":
        alt_text = alt_lookup.get(best_choice, best_choice)
        key_evidence.append(
            {
                "type": "alternative_comparison",
                "detail": f"model preferred {best_choice}: {_truncate_text(alt_text, 160)}",
            }
        )
        summary_fragments.append(f"最终对比时更偏向替代候选 {best_choice}")

    if isinstance(final_parsed, dict):
        final_label = str(final_parsed.get("label", ""))
        final_conf = _safe_float(final_parsed.get("confidence"), 0.0)
        final_reason = _truncate_text(str(final_parsed.get("reason", "")), 180)
        summary_text = (
            f"最终在 {final_stage} 阶段给出 {final_label} 判定"
            f"(confidence={final_conf:.2f})。"
        )
        if summary_fragments:
            summary_text += " 关键证据: " + "；".join(summary_fragments[:3]) + "。"
        if final_reason:
            summary_text += " 结论理由: " + final_reason
    else:
        summary_text = (
            f"未能产出最终 JSON 结果，最后状态为 {final_stage}。"
        )
        if summary_fragments:
            summary_text += " 已收集证据包括: " + "；".join(summary_fragments[:3]) + "。"

    return {
        "final_stage": final_stage,
        "final_label": str(final_parsed.get("label", "")) if isinstance(final_parsed, dict) else "",
        "final_confidence": round(_safe_float(final_parsed.get("confidence"), 0.0), 4)
        if isinstance(final_parsed, dict)
        else None,
        "summary_text": summary_text,
        "chain_steps": chain_steps,
        "key_evidence": key_evidence,
    }


def build_structural_prompt(
    dataset: str,
    triple: Triple,
    kb_internal: InternalGraphKB,
    relation_definition: str,
    graph_text: str,
    structural_text: str,
    template: Dict,
) -> str:
    lines = [
        "你正在做知识图谱错误检测。",
        "当前阶段只允许依据结构证据判断，不要假设外部文本已经验证该事实。",
        f"数据集: {dataset}",
        f"候选三元组: {kb_internal.format_triple(triple.h_id, triple.r_id, triple.t_id)}",
        f"三元组ID: ({triple.h_id}, {triple.r_id}, {triple.t_id})",
        f"关系定义: {relation_definition}",
        "",
        "结构信号摘要:",
        structural_text,
        "",
        "局部图证据:",
        graph_text,
        "",
        render_template_instructions(template, stage="structural", relation_id=triple.r_id),
    ]
    return "\n".join(lines)


def build_semantic_prompt(
    dataset: str,
    triple: Triple,
    kb_internal: InternalGraphKB,
    relation_definition: str,
    structural_text: str,
    external_text: str,
    evidence_package_text: str,
    template: Dict,
) -> str:
    lines = [
        "你正在做知识图谱错误检测。",
        "当前阶段重点依据外部文本证据及其质量分数判断语义是否支持该三元组。",
        "如果外部证据不足、缺乏直接支持，优先输出 unknown。",
        f"数据集: {dataset}",
        f"候选三元组: {kb_internal.format_triple(triple.h_id, triple.r_id, triple.t_id)}",
        f"三元组ID: ({triple.h_id}, {triple.r_id}, {triple.t_id})",
        f"关系定义: {relation_definition}",
        "",
        "结构先验摘要:",
        structural_text,
        "",
        "外部证据质量排序:",
        external_text if external_text.strip() else "无可用外部证据。",
        "",
        "统一证据包摘要:",
        evidence_package_text if evidence_package_text.strip() else "无统一证据包。",
        "",
        render_template_instructions(template, stage="semantic", relation_id=triple.r_id),
    ]
    return "\n".join(lines)


def build_hybrid_prompt(
    dataset: str,
    triple: Triple,
    kb_internal: InternalGraphKB,
    relation_definition: str,
    graph_text: str,
    structural_text: str,
    external_text: str,
    evidence_package_text: str,
    alternatives_text: List[str],
    template: Dict,
) -> str:
    lines = [
        "你正在做知识图谱错误检测的最终综合判定。",
        "请联合使用结构证据、外部文本证据、候选替代三元组和风险信号。",
        f"数据集: {dataset}",
        f"候选三元组: {kb_internal.format_triple(triple.h_id, triple.r_id, triple.t_id)}",
        f"三元组ID: ({triple.h_id}, {triple.r_id}, {triple.t_id})",
        f"关系定义: {relation_definition}",
        "",
        "结构信号摘要:",
        structural_text,
        "",
        "局部图证据:",
        graph_text,
        "",
        "外部证据质量排序:",
        external_text if external_text.strip() else "无可用外部证据。",
        "",
        "统一证据包摘要:",
        evidence_package_text if evidence_package_text.strip() else "无统一证据包。",
        "",
    ]
    if alternatives_text:
        lines.append("对比候选三元组:")
        lines.extend(alternatives_text)
        lines.append("")
    lines.append(render_template_instructions(template, stage="hybrid", relation_id=triple.r_id))
    return "\n".join(lines)


def _call_stage(
    stage: str,
    template: Dict,
    prompt: str,
    llm_client: BaseLLMClient,
    temperature: float,
    max_tokens: Optional[int],
) -> Dict:
    raw_response = ""
    parsed = None
    error = None
    try:
        raw = llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_response = str(raw.get("text", "") or "")
        parsed = extract_json_from_text(raw_response)
        parsed = _ensure_json_defaults(parsed, template.get("id", ""))
    except Exception as exc:
        error = str(exc)

    return {
        "stage": stage,
        "template_id": template.get("id"),
        "template_family": template.get("template_family", ""),
        "selected_primary_family": template.get("selected_primary_family", ""),
        "selected_primary_template_id": template.get("selected_primary_template_id", ""),
        "raw_response": raw_response,
        "parsed": parsed,
        "error": error,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = Path(__file__).resolve().parent.parent
    edges_path = (
        Path(args.edges_path)
        if args.edges_path
        else root / args.internal_kb_dir / args.dataset / "edges.jsonl"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else root / "rag_kge_thesis" / "output" / args.dataset / "ollama_scores.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edges = load_edges(edges_path)
    if args.resume:
        scored = load_scored_triples(output_path)
        edges = [
            edge
            for edge in edges
            if (str(edge.get("h_id")), str(edge.get("r_id")), str(edge.get("t_id"))) not in scored
        ]

    if args.sample_size > 0 and args.sample_size < len(edges):
        edges = random.sample(edges, args.sample_size)

    kb_internal = InternalGraphKB.from_dataset(args.dataset, kb_dir=args.internal_kb_dir)
    kb_external = ExternalTextKB.from_dataset(args.dataset)
    assets = load_default_assets(Path(__file__).resolve().parent)
    relation_meta = assets["relation_meta"]
    templates = assets["templates"]

    llm_client = build_llm_client(
        provider=args.llm_provider,
        model=args.model,
        ollama_url=args.ollama_url,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        llm_timeout=args.llm_timeout,
    )
    llm_max_tokens = args.llm_max_tokens if args.llm_max_tokens and args.llm_max_tokens > 0 else None

    kge_path = _resolve_kge_path(root, args.dataset, args.kge_model_path)
    kge_model = _load_kge_model(kge_path) if kge_path else None

    dense_retriever = None
    if args.retrieval_mode in {"dense", "hybrid"}:
        dense_retriever = DenseRetriever(
            model_name=args.dense_model,
            batch_size=args.dense_batch_size,
            max_chars=args.dense_max_chars,
        )

    nli_classifier = None
    if not args.disable_nli_stance:
        nli_classifier = NLIStanceClassifier(
            model_name=args.nli_model,
            neutral_threshold=args.nli_neutral_threshold,
        )

    if dense_retriever is not None and not dense_retriever.available:
        print(f"[warning] dense retriever unavailable, fallback to lexical scoring: {dense_retriever.load_error}")
    if nli_classifier is not None and not nli_classifier.available:
        print(f"[warning] NLI stance model unavailable, fallback to heuristic stance: {nli_classifier.load_error}")

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as fout:
        for idx, edge in enumerate(edges, 1):
            triple = Triple(
                h_id=str(edge["h_id"]),
                r_id=str(edge["r_id"]),
                t_id=str(edge["t_id"]),
            )
            relation_definition = get_relation_definition(args.dataset, triple.r_id, kb_internal)
            query_text = build_query_text(args.dataset, triple.h_id, triple.r_id, triple.t_id, kb_internal)
            claim_text = _build_claim_text(kb_internal, triple)
            graph_text = kb_internal.render_subgraph_as_text(
                h_id=triple.h_id,
                r_id=triple.r_id,
                t_id=triple.t_id,
                hops=args.hops,
                max_edges=args.max_edges,
                include_descriptions=True,
                exclude_candidate_from_neighborhood=True,
            )

            kge_prior = None
            if kge_model is not None:
                plausibility = _kge_plausibility(kge_model, triple.h_id, triple.r_id, triple.t_id)
                kge_prior = round(1.0 - plausibility, 6)

            relation_obj = kb_internal.relations.get(triple.r_id)
            relation_name = relation_obj.name if relation_obj and relation_obj.name else triple.r_id
            relation_tags = merge_relation_tags(
                relation_meta.get(triple.r_id, []),
                infer_relation_tags(args.dataset, triple.r_id, relation_name),
            )
            relation_meta_runtime = relation_meta
            if relation_tags:
                relation_meta_runtime = dict(relation_meta)
                relation_meta_runtime[triple.r_id] = relation_tags

            structural_signals = build_structural_signals(
                kb_internal=kb_internal,
                triple=triple,
                relation_tags=relation_tags,
                kge_prior=kge_prior,
            )
            structural_text = render_structural_signals(structural_signals)

            stage_outputs: List[Dict] = []
            heuristic = None
            final_stage = "pending"
            final_parsed = None
            alternatives_text: List[str] = []
            semantic_signals: Dict = {}
            evidence_package: Dict = {}
            evidence_package_text = ""
            template_route: Dict = {}

            if args.enable_structural_heuristic_stop:
                heuristic = heuristic_structural_stop(structural_signals)
                if heuristic is not None:
                    final_stage = heuristic.stage
                    final_parsed = {
                        "label": heuristic.label,
                        "confidence": round(heuristic.confidence, 4),
                        "reason": heuristic.reason,
                        "template_id": heuristic.stage,
                        "evidence_used": ["structural_signals", "internal_neighborhood", "kge_prior"],
                        "best_choice": "",
                        "should_stop": True,
                        "evidence_consistency": round(heuristic.confidence, 4),
                        "risk_flags": heuristic.risk_flags,
                    }

            ranked_docs = []
            external_text = ""
            if final_parsed is None:
                structural_template = select_template(
                    stage="structural",
                    relation_id=triple.r_id,
                    templates=templates,
                    relation_meta=relation_meta_runtime,
                )
                structural_prompt = build_structural_prompt(
                    dataset=args.dataset,
                    triple=triple,
                    kb_internal=kb_internal,
                    relation_definition=relation_definition,
                    graph_text=graph_text,
                    structural_text=structural_text,
                    template=structural_template,
                )
                structural_output = _call_stage(
                    stage="structural",
                    template=structural_template,
                    prompt=structural_prompt,
                    llm_client=llm_client,
                    temperature=args.temperature,
                    max_tokens=llm_max_tokens,
                )
                stage_outputs.append(structural_output)
                if structural_output.get("parsed") is not None:
                    final_parsed = structural_output.get("parsed")
                    final_stage = "structural"

                if not should_stop_after_llm(final_parsed, args.struct_confidence_stop):
                    ranked_docs = rank_external_docs(
                        kb_external=kb_external,
                        kb_internal=kb_internal,
                        triple=triple,
                        query_text=query_text,
                        top_k=args.top_k_external,
                        max_candidates=args.max_external_candidates,
                        weights=(0.3, 0.35, 0.15, max(0.0, args.dense_weight if args.retrieval_mode != "lexical" else 0.0)),
                        retrieval_mode=args.retrieval_mode,
                        dense_retriever=dense_retriever,
                        claim_text=claim_text if not args.disable_nli_stance else "",
                        nli_classifier=nli_classifier,
                    )
                    external_text = render_ranked_docs(ranked_docs)
                    semantic_signals = aggregate_semantic_signals(ranked_docs)
                    evidence_package = build_unified_evidence_package(structural_signals, ranked_docs)
                    evidence_package_text = render_evidence_package(
                        evidence_package,
                        max_chars=args.evidence_package_chars,
                    )
                    external_confirm = None
                    if args.enable_external_confirmation_stop:
                        external_confirm = heuristic_external_confirmation_stop(
                            signals=structural_signals,
                            semantic_signals=semantic_signals,
                            consistency_threshold=args.external_confirm_threshold,
                            support_ratio_threshold=args.external_support_threshold,
                            top_quality_threshold=args.external_top_quality_threshold,
                            max_conflict_for_confirm=args.external_max_conflict,
                        )

                    if external_confirm is not None:
                        final_stage = external_confirm.stage
                        final_parsed = {
                            "label": external_confirm.label,
                            "confidence": round(external_confirm.confidence, 4),
                            "reason": external_confirm.reason,
                            "template_id": external_confirm.stage,
                            "evidence_used": ["external_docs", "semantic_signals", "evidence_package"],
                            "best_choice": "",
                            "should_stop": True,
                            "evidence_consistency": round(
                                float(semantic_signals.get("evidence_consistency", external_confirm.confidence)),
                                4,
                            ),
                            "risk_flags": external_confirm.risk_flags,
                        }
                    else:
                        semantic_template = select_template(
                            stage="semantic",
                            relation_id=triple.r_id,
                            templates=templates,
                            relation_meta=relation_meta_runtime,
                            has_external=bool(ranked_docs),
                        )
                        semantic_prompt = build_semantic_prompt(
                            dataset=args.dataset,
                            triple=triple,
                            kb_internal=kb_internal,
                            relation_definition=relation_definition,
                            structural_text=structural_text,
                            external_text=external_text,
                            evidence_package_text=evidence_package_text,
                            template=semantic_template,
                        )
                        semantic_output = _call_stage(
                            stage="semantic",
                            template=semantic_template,
                            prompt=semantic_prompt,
                            llm_client=llm_client,
                            temperature=args.temperature,
                            max_tokens=llm_max_tokens,
                        )
                        stage_outputs.append(semantic_output)
                        if semantic_output.get("parsed") is not None:
                            final_parsed = semantic_output.get("parsed")
                            final_stage = "semantic"

                        if not should_stop_after_llm(
                            semantic_output.get("parsed"),
                            args.semantic_confidence_stop,
                            args.semantic_consistency_stop,
                        ):
                            alternatives_text = build_alternatives(
                                kb_internal=kb_internal,
                                triple=triple,
                                num_alternatives=args.num_alternatives,
                                alt_pool_size=args.alt_pool_size,
                                seed=args.seed + idx,
                            )

                            if args.disable_two_level_template_select:
                                template_route = {}
                                hybrid_template = select_template(
                                    stage="hybrid",
                                    relation_id=triple.r_id,
                                    templates=templates,
                                    relation_meta=relation_meta_runtime,
                                    has_external=bool(ranked_docs),
                                    has_alternatives=bool(alternatives_text),
                                )
                            else:
                                template_route = select_template_two_level(
                                    relation_id=triple.r_id,
                                    templates=templates,
                                    relation_meta=relation_meta_runtime,
                                    structural_signals=structural_signals.to_dict(),
                                    semantic_signals=semantic_signals,
                                    has_external=bool(ranked_docs),
                                    has_alternatives=bool(alternatives_text),
                                )
                                hybrid_template = template_route.get("instance_template", {})
                                if not hybrid_template:
                                    hybrid_template = select_template(
                                        stage="hybrid",
                                        relation_id=triple.r_id,
                                        templates=templates,
                                        relation_meta=relation_meta_runtime,
                                        has_external=bool(ranked_docs),
                                        has_alternatives=bool(alternatives_text),
                                    )

                            hybrid_prompt = build_hybrid_prompt(
                                dataset=args.dataset,
                                triple=triple,
                                kb_internal=kb_internal,
                                relation_definition=relation_definition,
                                graph_text=graph_text,
                                structural_text=structural_text,
                                external_text=external_text,
                                evidence_package_text=evidence_package_text,
                                alternatives_text=alternatives_text,
                                template=hybrid_template,
                            )
                            hybrid_output = _call_stage(
                                stage="hybrid",
                                template=hybrid_template,
                                prompt=hybrid_prompt,
                                llm_client=llm_client,
                                temperature=args.temperature,
                                max_tokens=llm_max_tokens,
                            )
                            stage_outputs.append(hybrid_output)
                            if hybrid_output.get("parsed") is not None:
                                final_parsed = hybrid_output.get("parsed")
                                final_stage = "hybrid"

            if not ranked_docs:
                ranked_docs = rank_external_docs(
                    kb_external=kb_external,
                    kb_internal=kb_internal,
                    triple=triple,
                    query_text=query_text,
                    top_k=args.top_k_external,
                    max_candidates=args.max_external_candidates,
                    weights=(0.3, 0.35, 0.15, max(0.0, args.dense_weight if args.retrieval_mode != "lexical" else 0.0)),
                    retrieval_mode=args.retrieval_mode,
                    dense_retriever=dense_retriever,
                    claim_text=claim_text if not args.disable_nli_stance else "",
                    nli_classifier=nli_classifier,
                )

            if not semantic_signals:
                semantic_signals = aggregate_semantic_signals(ranked_docs)
            if not evidence_package:
                evidence_package = build_unified_evidence_package(structural_signals, ranked_docs)
                evidence_package_text = render_evidence_package(
                    evidence_package,
                    max_chars=args.evidence_package_chars,
                )

            if final_parsed is None:
                final_stage = "failed"
            elif isinstance(final_parsed, dict):
                if "evidence_consistency" not in final_parsed or final_parsed.get("evidence_consistency") is None:
                    final_parsed["evidence_consistency"] = evidence_package.get(
                        "evidence_consistency", semantic_signals.get("evidence_consistency", 0.5)
                    )

            evidence_chain_summary = build_evidence_chain_summary(
                structural_signals=structural_signals,
                ranked_docs=ranked_docs,
                stage_outputs=stage_outputs,
                final_parsed=final_parsed,
                final_stage=final_stage,
                heuristic=heuristic,
                alternatives_text=alternatives_text,
            )

            record = {
                "h_id": triple.h_id,
                "r_id": triple.r_id,
                "t_id": triple.t_id,
                "split": edge.get("split", ""),
                "model": args.model,
                "llm_provider": args.llm_provider,
                "relation_tags": relation_tags,
                "final_stage": final_stage,
                "heuristic_stop": heuristic is not None,
                "external_confirmation_stop": final_stage == "external_confirmation",
                "structural_signals": structural_signals.to_dict(),
                "semantic_signals": semantic_signals,
                "evidence_package": evidence_package,
                "retrieval_config": {
                    "retrieval_mode": args.retrieval_mode,
                    "llm_provider": args.llm_provider,
                    "llm_max_tokens": args.llm_max_tokens,
                    "llm_timeout": args.llm_timeout,
                    "ollama_url": args.ollama_url if args.llm_provider == "ollama" else "",
                    "llm_api_base": args.llm_api_base if args.llm_provider == "openai_compatible" else "",
                    "dense_model": args.dense_model if dense_retriever is not None else "",
                    "dense_enabled": dense_retriever is not None,
                    "dense_available": dense_retriever.available if dense_retriever is not None else False,
                    "nli_model": args.nli_model if nli_classifier is not None else "",
                    "nli_enabled": nli_classifier is not None,
                    "nli_available": nli_classifier.available if nli_classifier is not None else False,
                    "two_level_template_select": not args.disable_two_level_template_select,
                    "external_confirm_stop": args.enable_external_confirmation_stop,
                    "external_confirm_threshold": args.external_confirm_threshold,
                    "external_support_threshold": args.external_support_threshold,
                    "external_top_quality_threshold": args.external_top_quality_threshold,
                    "external_max_conflict": args.external_max_conflict,
                },
                "template_route": template_route,
                "external_docs": docs_as_records(ranked_docs),
                "stage_outputs": stage_outputs,
                "parsed": final_parsed,
                "evidence_chain_summary": evidence_chain_summary,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{idx}/{len(edges)}] staged scoring "
                f"({triple.h_id}, {triple.r_id}, {triple.t_id}) -> final_stage={final_stage}"
            )


if __name__ == "__main__":
    main()
