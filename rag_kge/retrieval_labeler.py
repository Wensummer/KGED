"""
Sampling multiple evidence combinations for triples, querying Ollama, and
deriving soft labels for individual evidences (edges/docs) for re-ranking.

Usage (example):
python3 -m rag_kge.retrieval_labeler \
  --dataset wn18rr \
  --triples_path dataset/wn18rr/mixture_anomaly/05/anomaly.jsonl \
  --output_path rag_kge/output/wn18rr/evidence_labels.jsonl \
  --sample_size 50 \
  --num_combos 6 \
  --max_internal 40 \
  --max_external 12 \
  --ollama_model llama3:8b \
  --ollama_url http://localhost:11434/api/generate
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .internal_graphrag import InternalGraphKB
from .external_rag import ExternalTextKB, ExternalDoc
from .context_builder import build_query_text, get_relation_definition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect evidence-level labels via LLM rewards.")
    p.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr", "fb15k-237", "nell995"])
    p.add_argument("--triples_path", type=str, required=True, help="JSONL with h_id,r_id,t_id,label fields.")
    p.add_argument("--output_path", type=str, required=True, help="Where to write evidence labels JSONL.")
    p.add_argument("--sample_size", type=int, default=100, help="How many triples to sample (-1 for all).")
    p.add_argument("--num_combos", type=int, default=6, help="Evidence combos per triple.")
    p.add_argument("--max_internal", type=int, default=40, help="Max internal edges to consider as candidates.")
    p.add_argument("--max_external", type=int, default=12, help="Max external docs to consider as candidates.")
    p.add_argument("--internal_subset", type=str, default="10,20", help="Comma list of internal subset sizes.")
    p.add_argument("--external_subset", type=str, default="4,8", help="Comma list of external subset sizes.")
    p.add_argument("--ollama_model", type=str, default="llama3:8b")
    p.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--importance_tau",
        type=float,
        default=0.05,
        help="Threshold for evidence importance; lower to allow more labels (default 0.05).",
    )
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_triples(path: Path, limit: Optional[int], seed: int) -> List[Dict]:
    triples: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "label" not in obj:
                raise ValueError("triples_path must include 'label' field (1=true,0=noise).")
            triples.append(obj)
    random.Random(seed).shuffle(triples)
    if limit is not None and limit > 0:
        triples = triples[:limit]
    return triples


def call_ollama(prompt: str, url: str, model: str, temperature: float) -> Tuple[str, Optional[Dict]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    text = resp.json().get("response", "")
    parsed = extract_json(text)
    return text, parsed


def extract_json(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_prompt(
    kb_int: InternalGraphKB,
    kb_ext: ExternalTextKB,
    dataset: str,
    triple: Dict,
    edges: List,  # list of InternalGraphKB.Edge
    docs: List[ExternalDoc],
) -> str:
    h_id, r_id, t_id = triple["h_id"], triple["r_id"], triple["t_id"]
    cand_human = kb_int.format_triple(h_id, r_id, t_id)
    lines = []
    lines.append(
        "You are checking whether a candidate knowledge graph triple is correct.\n"
        "Internal graph evidence may contain errors and is NOT ground truth.\n"
        "If evidence is weak or conflicting, prefer 'incorrect' or 'unknown' with low confidence.\n"
        "Return ONLY a JSON object with keys label, confidence, reason.\n"
        "label in ['correct','incorrect','unknown']; confidence in [0,1]; reason concise."
    )
    lines.append("\nRELATION DEFINITION:")
    lines.append(get_relation_definition(dataset, r_id, kb_int))
    lines.append(f"\nCandidate triple: {cand_human}  [ids=({h_id}, {r_id}, {t_id})]\n")
    lines.append("INTERNAL EVIDENCE:\n")
    for idx, e in enumerate(edges, 1):
        # edges are Edge objects, not dicts
        lines.append(f"{idx}. {kb_int.format_triple(e.h_id, e.r_id, e.t_id)}")
    lines.append("\nEXTERNAL EVIDENCE:\n")
    for idx, d in enumerate(docs, 1):
        body = d.text.strip().replace("\n", " ")
        lines.append(f"{idx}. [{d.source}] {d.title} (entity={d.entity_name}) {body}")
    return "\n".join(lines)


def compute_importance(combos: List[Dict], candidates: List[str], key: str, tau: float = 0.2):
    """
    combos: list of {"items": set(ids), "reward": float}
    candidates: list of candidate ids
    key: not used (kept for clarity)
    Notes:
        If a candidate only ever appears in included sets (no exclusion samples),
        we fall back to comparing against a neutral baseline of 0.5 reward so that
        low-signal but present items can still be labeled.
    """
    stats = []
    for cid in candidates:
        inc = [c["reward"] for c in combos if cid in c["items"]]
        exc = [c["reward"] for c in combos if cid not in c["items"]]
        if len(inc) == 0 and len(exc) == 0:
            continue
        inc_avg = sum(inc) / len(inc) if inc else 0.0
        # if never excluded, compare to neutral 0.5 so we can still get a signal
        exc_avg = sum(exc) / len(exc) if exc else 0.5
        imp = inc_avg - exc_avg
        if imp > tau:
            label = 1
        elif imp < -tau:
            label = 0
        else:
            continue
        stats.append({"id": cid, "importance": imp, "label": label})
    return stats


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    triples = load_triples(Path(args.triples_path), args.sample_size, args.seed)
    kb_int = InternalGraphKB.from_dataset(args.dataset)
    kb_ext = ExternalTextKB.from_dataset(args.dataset)

    int_subset_sizes = [int(x) for x in args.internal_subset.split(",") if x.strip()]
    ext_subset_sizes = [int(x) for x in args.external_subset.split(",") if x.strip()]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, triple in enumerate(triples, 1):
            h_id, r_id, t_id = triple["h_id"], triple["r_id"], triple["t_id"]
            # internal candidates
            _, edges = kb_int.get_local_subgraph(
                [h_id, t_id],
                hops=2,
                max_edges=args.max_internal,
                exclude_triples=[(h_id, r_id, t_id)],
            )
            edge_ids = [f"{e.h_id}|{e.r_id}|{e.t_id}" for e in edges]
            # external candidates: rank docs for head+tail using simple lexical re-ranker
            query_text = build_query_text(args.dataset, h_id, r_id, t_id, kb_int)
            docs = kb_ext.rank_docs_for_entities([h_id, t_id], query_text=query_text, top_k=args.max_external)
            doc_ids = [d.doc_id for d in docs]

            combos_edges = []
            combos_docs = []
            combos_reward = []
            # sample combos
            for _ in range(args.num_combos):
                chosen_edges = []
                if edge_ids:
                    sz = random.choice(int_subset_sizes)
                    chosen_edges = random.sample(edge_ids, min(sz, len(edge_ids)))
                chosen_docs = []
                if doc_ids:
                    sz = random.choice(ext_subset_sizes)
                    chosen_docs = random.sample(doc_ids, min(sz, len(doc_ids)))
                # build subsets
                subset_edges = [e for e, cid in zip(edges, edge_ids) if cid in chosen_edges]
                subset_docs = [d for d in docs if d.doc_id in chosen_docs]
                prompt = build_prompt(kb_int, kb_ext, args.dataset, triple, subset_edges, subset_docs)
                try:
                    _, parsed = call_ollama(prompt, args.ollama_url, args.ollama_model, args.temperature)
                    lab = (parsed.get("label") if parsed else "").lower()
                    if lab == "incorrect":
                        reward = 1.0 if triple["label"] == 0 else 0.0
                    elif lab == "correct":
                        reward = 1.0 if triple["label"] == 1 else 0.0
                    else:
                        reward = 0.5
                except Exception as exc:
                    reward = 0.0
                combos_edges.append({"items": set(chosen_edges), "reward": reward})
                combos_docs.append({"items": set(chosen_docs), "reward": reward})
                combos_reward.append(reward)

            edge_stats = compute_importance(combos_edges, edge_ids, "edge", tau=args.importance_tau)
            doc_stats = compute_importance(combos_docs, doc_ids, "doc", tau=args.importance_tau)

            # emit labeled evidences
            for st in edge_stats:
                h, r, t = st["id"].split("|")
                rec = {
                    "kind": "internal",
                    "triple": {"h_id": h_id, "r_id": r_id, "t_id": t_id, "label": triple["label"]},
                    "edge": {"h_id": h, "r_id": r, "t_id": t},
                    "label": st["label"],
                    "importance": st["importance"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for st in doc_stats:
                rec = {
                    "kind": "external",
                    "triple": {"h_id": h_id, "r_id": r_id, "t_id": t_id, "label": triple["label"]},
                    "doc_id": st["id"],
                    "label": st["label"],
                    "importance": st["importance"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if idx % 5 == 0:
                print(f"[{idx}/{len(triples)}] processed triple ({h_id},{r_id},{t_id})")

    print(f"Done. Wrote evidence labels to {out_path}")


if __name__ == "__main__":
    main()
