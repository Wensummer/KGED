"""
Inspect sampled triples for manual labeling.

Reads eval/relevance_sample.jsonl (generated samples), prints internal neighborhood
and top-k external docs for a slice of rows, and writes the dump to a file.

Usage:
  python3 eval/inspect_relevance.py --start 0 --count 5 --output eval/inspect_out.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect sampled triples for manual labeling.")
    p.add_argument("--sample_path", type=str, default="eval/relevance_sample.jsonl")
    p.add_argument("--start", type=int, default=0, help="Start index (0-based).")
    p.add_argument("--count", type=int, default=5, help="Number of rows to inspect.")
    p.add_argument("--hops", type=int, default=1, help="Graph hops for neighborhood.")
    p.add_argument("--max_edges", type=int, default=50, help="Max internal edges.")
    p.add_argument("--top_docs", type=int, default=10, help="Top external docs to show.")
    p.add_argument("--output", type=str, default="eval/inspect_out.txt", help="Where to write the dump.")
    return p.parse_args()


def load_samples(path: Path):
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    args = parse_args()
    # ensure repo root on sys.path
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))

    from rag_kge.internal_graphrag import InternalGraphKB
    from rag_kge.external_rag import ExternalTextKB
    from rag_kge.context_builder import build_query_text
    from rag_kge.score_triples_ollama import Triple

    samples = load_samples(Path(args.sample_path))
    kb_int = InternalGraphKB.from_dataset("wn18rr", kb_dir="graph_kb_noisy")
    kb_ext = ExternalTextKB.from_dataset("wn18rr")

    start = max(0, args.start)
    end = min(len(samples), start + args.count)

    lines = []
    for idx in range(start, end):
        o = samples[idx]
        triple = Triple(h_id=o["h_id"], r_id=o["r_id"], t_id=o["t_id"])
        lines.append(f"=== idx={idx} === {triple}")
        lines.append(f"label={o.get('label')} score={o.get('score')}")

        # internal neighborhood
        _, edges = kb_int.get_local_subgraph(
            [triple.h_id, triple.t_id], hops=args.hops, max_edges=args.max_edges
        )
        for e in edges:
            lines.append(f"INTERNAL: {e.h_id} {e.r_id} {e.t_id} split={e.split}")

        # external docs
        q = build_query_text("wn18rr", triple.h_id, triple.r_id, triple.t_id, kb_int)
        docs = kb_ext.rank_docs_for_entities(
            [triple.h_id, triple.t_id], query_text=q, top_k=args.top_docs
        )
        for d in docs:
            snippet = (d.text or "").replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(f"DOC {d.doc_id} {snippet}")
        lines.append("-" * 60)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()

