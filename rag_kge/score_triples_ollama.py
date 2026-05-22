import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pickle
import json

import requests

from .context_builder import build_query_text, build_context_for_triple, Triple
from .internal_graphrag import InternalGraphKB
from .external_rag import ExternalTextKB
from .cot_prompt import load_default_assets, select_template, render_cot_instructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score knowledge graph triples with an Ollama-hosted LLM using internal+external RAG context."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wn18rr",
        choices=["wn18rr", "fb15k-237", "nell995"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--internal_kb_dir",
        type=str,
        default="graph_kb",
        help='Internal KB directory name under repo root (default: graph_kb). Use e.g. "graph_kb_train" for train-only KB.',
    )
    parser.add_argument(
        "--edges_path",
        type=str,
        default=None,
        help="Path to edges.jsonl. Defaults to graph_kb/<dataset>/edges.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to write scored triples JSONL. Defaults to rag_kge/output/<dataset>/ollama_scores.jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output_path and skip triples already present in it.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of triples to score (random sample). Use -1 to score all.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama model name, e.g., llama3, llama3:8b, your fine-tuned tag.",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint.",
    )
    parser.add_argument("--hops", type=int, default=1, help="Graph hops for internal context.")
    parser.add_argument("--max_edges", type=int, default=50, help="Max edges in internal subgraph.")
    parser.add_argument(
        "--top_k_external",
        type=int,
        default=4,
        help="External docs for head and tail (each).",
    )
    parser.add_argument(
        "--external_policy",
        type=str,
        default="all",
        choices=["all", "cross", "none"],
        help="How to include external docs in the prompt: all=use selected docs; cross=only docs mentioning BOTH head+tail names; none=omit external evidence.",
    )
    parser.add_argument(
        "--max_external_candidates",
        type=int,
        default=20,
        help="How many external docs to consider before re-ranking.",
    )
    parser.add_argument(
        "--max_internal_candidates",
        type=int,
        default=80,
        help="How many internal edges to consider before re-ranking.",
    )
    parser.add_argument(
        "--num_alternatives",
        type=int,
        default=8,
        help="How many alternative triples (A1..Ak) to include in the contrastive prompt.",
    )
    parser.add_argument(
        "--alt_pool_size",
        type=int,
        default=50,
        help="How many plausible alternatives to collect before selecting --num_alternatives.",
    )
    parser.add_argument(
        "--alt_kge_path",
        "--distmult_path",
        dest="alt_kge_path",
        type=str,
        default=None,
        help="Optional KGE checkpoint (DistMult or ComplEx) to pick hard alternatives (default: rag_kge/output/<dataset>/distmult.pt).",
    )
    parser.add_argument(
        "--external_reranker_path",
        type=str,
        default=None,
        help="Pickle file from retrieval_scorer (kind=external) to re-rank docs.",
    )
    parser.add_argument(
        "--internal_reranker_path",
        type=str,
        default=None,
        help="Pickle file from retrieval_scorer (kind=internal) to re-rank edges.",
    )
    parser.add_argument(
        "--internal_prior_path",
        type=str,
        default=None,
        help="JSONL with prior scores per edge (e.g., from prior_scoring).",
    )
    parser.add_argument(
        "--prior_threshold",
        type=float,
        default=0.8,
        help="If prior >= threshold, mark edge as suspect and optionally filter.",
    )
    parser.add_argument(
        "--filter_suspect",
        action="store_true",
        help="If set, filter out edges with prior >= threshold instead of just marking.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Ollama (0 for deterministic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sampling edges.",
    )
    return parser.parse_args()


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


def load_edges(path: Path) -> List[Dict]:
    edges: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edges.append(json.loads(line))
    return edges


def call_ollama_generate(
    url: str, model: str, prompt: str, temperature: float = 0.0
) -> Dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _load_kge_for_alt(path: Path) -> Optional[Dict]:
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
        return 0.0
    ent2idx = kge_model["ent2idx"]
    rel2idx = kge_model["rel2idx"]
    if h_id not in ent2idx or t_id not in ent2idx or r_id not in rel2idx:
        return 0.0
    try:
        import torch
    except Exception:
        return 0.0
    if kge_model["type"] == "distmult":
        ent_w = kge_model["ent_weight"]
        rel_w = kge_model["rel_weight"]
        h = ent_w[ent2idx[h_id]]
        r = rel_w[rel2idx[r_id]]
        t = ent_w[ent2idx[t_id]]
        s = (h * r * t).sum()
        return float(torch.sigmoid(s).item())
    if kge_model["type"] == "complex":
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
        s = (
            h_re * r_re_v * t_re
            + h_im * r_re_v * t_im
            + h_re * r_im_v * t_im
            - h_im * r_im_v * t_re
        ).sum()
        return float(torch.sigmoid(s).item())
    return 0.0


def _score_doc(model_dict: Dict, doc, h_name: str, t_name: str, query_tokens: List[str]) -> float:
    """
    Compute score for an external doc using the same simple features as retrieval_scorer.
    """
    tokens = re.findall(r"[a-z0-9]+", doc.text.lower())
    overlap = len(set(query_tokens) & set(tokens))
    contains_h = 1.0 if h_name and h_name.lower() in doc.text.lower() else 0.0
    contains_t = 1.0 if t_name and t_name.lower() in doc.text.lower() else 0.0
    length = len(doc.text)
    feats = [[overlap, contains_h, contains_t, length]]

    pairwise = model_dict.get("pairwise", False)
    if model_dict.get("type") == "sklearn_logreg":
        clf = model_dict["model"]
        if pairwise:
            # pairwise model expects diff; use zero baseline
            return float(clf.predict_proba([[0, 0, 0, 0]])[0][1])
        return float(clf.predict_proba(feats)[0][1])
    elif model_dict.get("type") == "prior":
        return float(model_dict.get("prior", 0.5))
    return 0.5


def _score_edge(model_dict: Dict, edge, triple: Triple, kb_int: InternalGraphKB) -> float:
    """
    Compute score for an internal edge using the same simple features as retrieval_scorer.
    """
    same_head = 1.0 if edge.h_id == triple.h_id else 0.0
    same_tail = 1.0 if edge.t_id == triple.t_id else 0.0
    same_rel = 1.0 if edge.r_id == triple.r_id else 0.0
    deg_h = len(kb_int.out_neighbors.get(edge.h_id, [])) + len(kb_int.in_neighbors.get(edge.h_id, []))
    deg_t = len(kb_int.out_neighbors.get(edge.t_id, [])) + len(kb_int.in_neighbors.get(edge.t_id, []))
    feats = [[same_head, same_tail, same_rel, deg_h, deg_t]]
    pairwise = model_dict.get("pairwise", False)
    if model_dict.get("type") == "sklearn_logreg":
        clf = model_dict["model"]
        if pairwise:
            return float(clf.predict_proba([[0, 0, 0, 0, 0]])[0][1])
        return float(clf.predict_proba(feats)[0][1])
    elif model_dict.get("type") == "prior":
        return float(model_dict.get("prior", 0.5))
    return 0.5


def _pairwise_win_scores(features: List[List[float]], clf) -> List[float]:
    """
    Given a list of feature vectors and a pairwise classifier,
    compute win-rate scores for each item vs all others.
    """
    n = len(features)
    if n == 1:
        return [0.5]
    scores = []
    for i in range(n):
        wins = 0.0
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            diff = [[a - b for a, b in zip(features[i], features[j])]]
            p = clf.predict_proba(diff)[0][1]
            wins += p
            cnt += 1
        scores.append(wins / cnt if cnt else 0.5)
    return scores


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Try to extract a JSON object from the LLM response text.
    Looks for the first {...} block and parses it.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


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
        else root
        / "rag_kge"
        / "output"
        / args.dataset
        / "ollama_scores.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edges = load_edges(edges_path)
    print(f"Loaded {len(edges)} edges from {edges_path}")

    if args.resume:
        scored = load_scored_triples(output_path)
        if scored:
            before = len(edges)
            edges = [
                e
                for e in edges
                if (str(e.get("h_id")), str(e.get("r_id")), str(e.get("t_id"))) not in scored
            ]
            print(
                f"Resume enabled: found {len(scored)} scored triples in {output_path}; "
                f"remaining {len(edges)} / {before}."
            )
        else:
            print(f"Resume enabled: no existing scored triples found at {output_path}.")

    if args.sample_size > 0 and args.sample_size < len(edges):
        edges = random.sample(edges, args.sample_size)
        print(f"Sampling {len(edges)} edges for scoring (use --sample_size -1 to score all).")
    else:
        print(f"Scoring all {len(edges)} edges.")

    # load KBs once
    kb_internal = InternalGraphKB.from_dataset(args.dataset, kb_dir=args.internal_kb_dir)
    kb_external = ExternalTextKB.from_dataset(args.dataset)
    cot_assets = load_default_assets(root / "rag_kge")

    # optional DistMult for selecting hard alternatives (more plausible competing triples)
    alt_kge = None
    alt_kge_path = Path(args.alt_kge_path) if args.alt_kge_path else None
    if alt_kge_path is None:
        candidate = root / "rag_kge" / "output" / args.dataset / "distmult.pt"
        if candidate.exists():
            alt_kge_path = candidate
    if alt_kge_path is not None and alt_kge_path.exists():
        alt_kge = _load_kge_for_alt(alt_kge_path)
        if alt_kge is not None:
            print(f"Loaded KGE (for hard alternatives) type={alt_kge['type']} from {alt_kge_path}")
        else:
            print("KGE checkpoint found but torch is unavailable or checkpoint format unsupported; alternatives will be unscored.")

    # load re-rankers if provided
    ext_reranker = None
    int_reranker = None
    if args.external_reranker_path:
        ext_path = Path(args.external_reranker_path)
        if ext_path.exists():
            with ext_path.open("rb") as f:
                ext_reranker = pickle.load(f)
            print(f"Loaded external reranker from {ext_path}")
        else:
            print(f"External reranker not found at {ext_path}")
    if args.internal_reranker_path:
        int_path = Path(args.internal_reranker_path)
        if int_path.exists():
            with int_path.open("rb") as f:
                int_reranker = pickle.load(f)
            print(f"Loaded internal reranker from {int_path}")
        else:
            print(f"Internal reranker not found at {int_path}")

    # load prior scores if provided
    prior_scores: Dict[str, float] = {}
    if args.internal_prior_path:
        ppath = Path(args.internal_prior_path)
        if ppath.exists():
            with ppath.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = f"{obj['h_id']}|{obj['r_id']}|{obj['t_id']}"
                        prior_scores[key] = float(obj.get("prior", 0.0))
                    except Exception:
                        continue
            print(f"Loaded {len(prior_scores)} edge priors from {ppath}")
        else:
            print(f"Internal prior file not found at {ppath}")

    def select_external_docs(triple: Triple):
        # gather candidates
        query_text = build_query_text(args.dataset, triple.h_id, triple.r_id, triple.t_id, kb_internal)
        cand_docs = kb_external.rank_docs_for_entities(
            [triple.h_id, triple.t_id],
            query_text=query_text,
            top_k=args.max_external_candidates,
        )
        if ext_reranker is None:
            return cand_docs[: args.top_k_external]
        q_tokens = re.findall(r"[a-z0-9]+", query_text.lower())
        h_name = kb_internal.entities.get(triple.h_id, None)
        t_name = kb_internal.entities.get(triple.t_id, None)
        h_name = h_name.name if h_name else ""
        t_name = t_name.name if t_name else ""
        # build features
        feats = []
        for doc in cand_docs:
            tokens = re.findall(r"[a-z0-9]+", doc.text.lower())
            overlap = len(set(q_tokens) & set(tokens))
            contains_h = 1.0 if h_name and h_name.lower() in doc.text.lower() else 0.0
            contains_t = 1.0 if t_name and t_name.lower() in doc.text.lower() else 0.0
            length = len(doc.text)
            feats.append([overlap, contains_h, contains_t, length])
        if ext_reranker.get("type") == "sklearn_logreg" and ext_reranker.get("pairwise", False):
            clf = ext_reranker["model"]
            win_scores = _pairwise_win_scores(feats, clf)
            scored = list(zip(cand_docs, win_scores))
        else:
            scored = [
                (doc, _score_doc(ext_reranker, doc, h_name, t_name, q_tokens))
                for doc in cand_docs
            ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[: args.top_k_external]]

    def select_internal_edges(triple: Triple):
        nodes, cand_edges = kb_internal.get_local_subgraph(
            [triple.h_id, triple.t_id], hops=args.hops, max_edges=args.max_internal_candidates
        )
        # apply prior filtering/marking
        filtered_edges = []
        for e in cand_edges:
            # leave-one-out: never include the candidate triple itself as evidence
            if e.h_id == triple.h_id and e.r_id == triple.r_id and e.t_id == triple.t_id:
                continue
            key = f"{e.h_id}|{e.r_id}|{e.t_id}"
            prior = prior_scores.get(key, 0.0)
            if args.filter_suspect and prior >= args.prior_threshold:
                continue
            # attach prior as attribute for rendering
            e.prior = prior  # type: ignore
            filtered_edges.append(e)
        cand_edges = filtered_edges

        if int_reranker is None:
            return cand_edges[: args.max_edges]
        feats = []
        for e in cand_edges:
            same_head = 1.0 if e.h_id == triple.h_id else 0.0
            same_tail = 1.0 if e.t_id == triple.t_id else 0.0
            same_rel = 1.0 if e.r_id == triple.r_id else 0.0
            deg_h = len(kb_internal.out_neighbors.get(e.h_id, [])) + len(kb_internal.in_neighbors.get(e.h_id, []))
            deg_t = len(kb_internal.out_neighbors.get(e.t_id, [])) + len(kb_internal.in_neighbors.get(e.t_id, []))
            feats.append([same_head, same_tail, same_rel, deg_h, deg_t])
        if int_reranker.get("type") == "sklearn_logreg" and int_reranker.get("pairwise", False):
            clf = int_reranker["model"]
            win_scores = _pairwise_win_scores(feats, clf)
            scored = list(zip(cand_edges, win_scores))
        else:
            scored = [(e, _score_edge(int_reranker, e, triple, kb_internal)) for e in cand_edges]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[: args.max_edges]]

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as fout:
        for idx, e in enumerate(edges, start=1):
            triple = Triple(h_id=e["h_id"], r_id=e["r_id"], t_id=e["t_id"])
            # build context with optional re-rankers
            # internal edges
            selected_edges = select_internal_edges(triple)
            # external docs
            selected_docs = select_external_docs(triple)

            # reuse context_builder's formatting utilities
            from .context_builder import ContextPieces, get_relation_definition

            # build graph text manually
            graph_lines = ["Candidate triple:", kb_internal.format_triple(triple.h_id, triple.r_id, triple.t_id), ""]
            for ent_id in [triple.h_id, triple.t_id]:
                ent = kb_internal.entities.get(ent_id)
                if ent:
                    graph_lines.append(f"Entity: {ent.name} (ID={ent.id})")
                    if ent.desc:
                        graph_lines.append(f"Description: {ent.desc}")
                    graph_lines.append("")
            graph_lines.append("Graph neighborhood:")
            for j, edge in enumerate(selected_edges, 1):
                prior = getattr(edge, "prior", 0.0)
                suspect_tag = f" [SUSPECT prior={prior:.2f}]" if prior >= args.prior_threshold else ""
                graph_lines.append(
                    f"{j}. {kb_internal.format_triple(edge.h_id, edge.r_id, edge.t_id)} "
                    f"[split={edge.split}]{suspect_tag}"
                )
            graph_text = "\n".join(graph_lines)

            # external evidence policy
            if args.external_policy == "none":
                selected_docs = []
            elif args.external_policy == "cross":
                h_ent = kb_internal.entities.get(triple.h_id)
                t_ent = kb_internal.entities.get(triple.t_id)
                h_name = (h_ent.name if h_ent and h_ent.name else triple.h_id).lower()
                t_name = (t_ent.name if t_ent and t_ent.name else triple.t_id).lower()
                filtered = []
                for d in selected_docs:
                    txt = (d.text or "").lower()
                    if h_name and t_name and h_name in txt and t_name in txt:
                        filtered.append(d)
                selected_docs = filtered
            external_text = kb_external.render_docs_as_text(selected_docs, max_chars=1600)

            # alternatives: always try to provide a consistent number of candidates.
            # Start with observed alternatives from KG; if insufficient, fill with relation-sampled ones.
            seed = args.seed + idx
            observed_pool = kb_internal.get_alternative_pool(
                h_id=triple.h_id,
                r_id=triple.r_id,
                t_id=triple.t_id,
                max_pool=max(args.alt_pool_size, args.num_alternatives),
                seed=seed,
            )
            observed_set = set(observed_pool)
            if alt_kge is not None and observed_pool:
                scored_obs = [
                    (cand, _kge_plausibility(alt_kge, cand[0], cand[1], cand[2]))
                    for cand in observed_pool
                ]
                scored_obs.sort(key=lambda x: x[1], reverse=True)
                ordered_pool = [c for c, _ in scored_obs]
            else:
                ordered_pool = list(observed_pool)
                random.Random(seed).shuffle(ordered_pool)

            alt_tuples = []
            seen = {(triple.h_id, triple.r_id, triple.t_id)}
            for cand in ordered_pool:
                if cand in seen:
                    continue
                alt_tuples.append(cand)
                seen.add(cand)
                if len(alt_tuples) >= args.num_alternatives:
                    break

            if len(alt_tuples) < args.num_alternatives:
                filler = kb_internal.get_alternative_triples(
                    h_id=triple.h_id,
                    r_id=triple.r_id,
                    t_id=triple.t_id,
                    max_alternatives=max(args.alt_pool_size, args.num_alternatives * 2),
                    seed=seed,
                )
                for cand in filler:
                    if cand in seen:
                        continue
                    alt_tuples.append(cand)
                    seen.add(cand)
                    if len(alt_tuples) >= args.num_alternatives:
                        break
            alternatives_text = []
            for i, (h, r, t) in enumerate(alt_tuples, 1):
                human = kb_internal.format_triple(h, r, t)
                src = "observed" if (h, r, t) in observed_set else "sampled"
                alternatives_text.append(f"A{i}: {human}  [ids=({h}, {r}, {t}); source={src}]")
            rel_def = get_relation_definition(args.dataset, triple.r_id, kb_internal)

            ctx_obj = ContextPieces(
                triple=triple,
                alternatives_text=alternatives_text,
                relation_definition=rel_def,
                graph_text=graph_text,
                external_text=external_text,
            )
            # CoT template selection
            tmpl = select_template(
                relation_id=triple.r_id,
                templates=cot_assets["templates"],
                relation_meta=cot_assets["relation_meta"],
                fallback_id="generic_compare_v1",
            )
            cot_text = render_cot_instructions(tmpl, triple.r_id)

            prompt = ctx_obj.to_prompt() + "\n\n" + cot_text + "\n\n请严格按上述步骤推理，并仅输出一个 JSON。"

            try:
                res = call_ollama_generate(
                    url=args.ollama_url,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                )
                raw_text = res.get("response", "")
            except Exception as exc:
                raw_text = ""
                parsed = None
                print(f"[{idx}/{len(edges)}] Error calling Ollama: {exc}")
            else:
                parsed = extract_json_from_text(raw_text)
                print(
                    f"[{idx}/{len(edges)}] scored triple "
                    f"({triple.h_id}, {triple.r_id}, {triple.t_id}); "
                    f"parsed={parsed is not None}"
                )

            record = {
                "h_id": triple.h_id,
                "r_id": triple.r_id,
                "t_id": triple.t_id,
                "split": e.get("split", ""),
                "model": args.model,
                "template_id": tmpl.get("id"),
                "raw_response": raw_text,
                "parsed": parsed,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Wrote scores to {output_path}")


if __name__ == "__main__":
    main()
