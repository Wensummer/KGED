"""
Compute lightweight prior scores for internal edges to filter/mark suspect neighbors.

Heuristics (configurable by weight parameters):
 - hop: 1-hop edges are more reliable than 2-hop (penalize hop=2).
 - degree: very high degree nodes are less informative (penalize degrees).
 - relation match: edges sharing the same relation as the target triple get a small bonus.
 - shared endpoints: edges touching the target head/tail get a bonus.

Output: edges_scored.jsonl with a "prior" field (higher = more suspect).

Usage example:
python3 -m rag_kge.prior_scoring \
  --dataset wn18rr \
  --output_path graph_kb/wn18rr/edges_scored.jsonl \
  --max_edges 200 \
  --hop_penalty 0.2 \
  --deg_penalty 0.01 \
  --rel_mismatch_penalty 0.1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .internal_graphrag import InternalGraphKB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute heuristic prior scores for internal edges.")
    p.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr", "fb15k-237", "nell995"])
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to write edges_scored.jsonl (default: graph_kb/<dataset>/edges_scored.jsonl)",
    )
    p.add_argument("--max_edges", type=int, default=200, help="Max edges to consider per triple when scanning.")
    p.add_argument("--hop_penalty", type=float, default=0.2, help="Penalty added for hop=2 edges.")
    p.add_argument(
        "--deg_penalty",
        type=float,
        default=0.01,
        help="Penalty multiplier for node degree (deg_h + deg_t) * deg_penalty.",
    )
    p.add_argument(
        "--rel_mismatch_penalty",
        type=float,
        default=0.1,
        help="Penalty added when edge relation != target relation.",
    )
    return p.parse_args()


def edge_prior(
    edge,
    triple_rel: str,
    kb: InternalGraphKB,
    hop_penalty: float,
    deg_penalty: float,
    rel_mismatch_penalty: float,
    hop: int,
) -> float:
    """
    Compute a simple prior: higher = more suspect.
    """
    prior = 0.0
    if hop >= 2:
        prior += hop_penalty
    # degree penalty
    deg_h = len(kb.out_neighbors.get(edge.h_id, [])) + len(kb.in_neighbors.get(edge.h_id, []))
    deg_t = len(kb.out_neighbors.get(edge.t_id, [])) + len(kb.in_neighbors.get(edge.t_id, []))
    prior += deg_penalty * (deg_h + deg_t)
    # relation mismatch
    if edge.r_id != triple_rel:
        prior += rel_mismatch_penalty
    # clip to [0,1]
    return min(1.0, max(0.0, prior))


def main() -> None:
    args = parse_args()
    kb = InternalGraphKB.from_dataset(args.dataset)
    out_path = (
        Path(args.output_path)
        if args.output_path
        else Path("graph_kb") / args.dataset / "edges_scored.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a quick lookup of edges for iteration
    all_edges = kb.edges
    scored = []
    for edge in all_edges:
        # default hop=1 for edges directly in the KB (we don't have explicit hop info per target)
        prior = edge_prior(
            edge=edge,
            triple_rel=edge.r_id,
            kb=kb,
            hop_penalty=args.hop_penalty,
            deg_penalty=args.deg_penalty,
            rel_mismatch_penalty=args.rel_mismatch_penalty,
            hop=1,
        )
        scored.append(
            {
                "h_id": edge.h_id,
                "r_id": edge.r_id,
                "t_id": edge.t_id,
                "split": edge.split,
                "prior": prior,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for obj in scored:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(scored)} edges with prior scores to {out_path}")


if __name__ == "__main__":
    main()

