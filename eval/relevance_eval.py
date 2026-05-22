"""
Compute retrieval metrics on a labeled relevance set.

Input: JSONL with at least fields `score` (float) and `label` (0/1).
Outputs: precision/recall@K (absolute and/or percentage) and average precision.

Usage:
  python3 eval/relevance_eval.py \
    --input eval/relevance_sample.jsonl \
    --abs_k 5 10 \
    --pct_k 0.01 0.02 0.05 \
    --output eval/relevance_metrics.tsv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute top-k retrieval metrics on labeled JSONL.")
    p.add_argument("--input", type=str, required=True, help="JSONL with fields: score (float), label (0/1)")
    p.add_argument("--abs_k", type=int, nargs="*", default=[5, 10, 20], help="Absolute cutoffs for top-k.")
    p.add_argument(
        "--pct_k",
        type=float,
        nargs="*",
        default=[0.01, 0.02, 0.05],
        help="Percentage cutoffs (0-1). e.g., 0.01 means top 1%% of samples.",
    )
    p.add_argument("--output", type=str, default="", help="Optional path to save metrics as TSV.")
    return p.parse_args()


def load_scores(path: Path) -> List[Tuple[float, int]]:
    out: List[Tuple[float, int]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = float(obj.get("score", 0.0))
            label = int(obj.get("label", 0))
            out.append((score, label))
    return out


def average_precision(pairs: List[Tuple[float, int]]) -> float:
    """Standard AP: sum precision@i over positives divided by #positives."""
    if not pairs:
        return 0.0
    sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    pos = sum(l for _, l in sorted_pairs)
    if pos == 0:
        return 0.0
    hit = 0
    ap = 0.0
    for i, (_, lbl) in enumerate(sorted_pairs, start=1):
        if lbl:
            hit += 1
            ap += hit / i
    return ap / pos


def topk_metrics(sorted_pairs: List[Tuple[float, int]], k: int, total_pos: int) -> Tuple[float, float, int]:
    k = max(1, min(k, len(sorted_pairs)))
    top = sorted_pairs[:k]
    tp = sum(l for _, l in top)
    prec = tp / k
    rec = tp / total_pos if total_pos else 0.0
    return prec, rec, k


def format_pct(p: float) -> str:
    return f"{p*100:.1f}%"


def main() -> None:
    args = parse_args()
    pairs = load_scores(Path(args.input))
    if not pairs:
        raise SystemExit("No samples found.")

    pairs.sort(key=lambda x: x[0], reverse=True)
    total = len(pairs)
    total_pos = sum(l for _, l in pairs)
    pos_rate = total_pos / total if total else 0.0

    rows = []

    # Absolute k
    for k in sorted(set(args.abs_k)):
        prec, rec, kk = topk_metrics(pairs, k, total_pos)
        rows.append(("abs", kk, prec, rec))

    # Percentage k
    for pct in sorted(set(args.pct_k)):
        if pct <= 0:
            continue
        k = max(1, math.ceil(total * pct))
        prec, rec, kk = topk_metrics(pairs, k, total_pos)
        rows.append((f"pct_{format_pct(pct)}", kk, prec, rec))

    ap = average_precision(pairs)

    lines = []
    lines.append(f"Total={total}, Positives={total_pos} (rate={pos_rate:.3f}), AP={ap:.4f}")
    lines.append("type\tk\tprecision\trecall")
    for t, k, p, r in rows:
        lines.append(f"{t}\t{k}\t{p:.4f}\t{r:.4f}")

    out_text = "\n".join(lines)
    print(out_text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
