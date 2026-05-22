"""
Inspect exported top-K suspects in a human-readable way.

Example:
python3 -m rag_kge.inspect_topk \
  --dataset wn18rr \
  --suspects rag_kge/output/wn18rr/top5_fused_top20_mult.jsonl \
  --llm_scores rag_kge/output/wn18rr/ollama_scores_kge_top20_v3.jsonl \
  --kge_prior_true rag_kge/output/wn18rr/kge_prior_true.jsonl \
  --kge_prior_anomaly rag_kge/output/wn18rr/kge_prior_anomaly.jsonl \
  --top_n 30 --format tsv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .internal_graphrag import InternalGraphKB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect top-K suspects with names and optional LLM/KGE details.")
    p.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr", "fb15k-237", "nell995"])
    p.add_argument("--suspects", type=str, required=True, help="JSONL exported by evaluate_topk --export_topk.")
    p.add_argument("--top_n", type=int, default=30, help="How many rows to display.")
    p.add_argument("--llm_scores", type=str, default=None, help="LLM scores JSONL (optional).")
    p.add_argument("--kge_prior_true", type=str, default=None, help="KGE priors for true triples (optional).")
    p.add_argument("--kge_prior_anomaly", type=str, default=None, help="KGE priors for anomaly triples (optional).")
    p.add_argument(
        "--format",
        type=str,
        default="tsv",
        choices=["tsv", "jsonl"],
        help="Output format: tsv or jsonl.",
    )
    p.add_argument("--show_reason", action="store_true", help="Include LLM reason text (if available).")
    return p.parse_args()


def triple_key(h: str, r: str, t: str) -> str:
    return f"{h}|{r}|{t}"


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_kge_priors(true_path: Optional[Path], anomaly_path: Optional[Path]) -> Dict[str, float]:
    priors: Dict[str, float] = {}
    for p in [true_path, anomaly_path]:
        if p is None or not p.exists():
            continue
        for obj in read_jsonl(p):
            k = triple_key(obj["h_id"], obj["r_id"], obj["t_id"])
            v = obj.get("kge_prior")
            if v is None:
                continue
            try:
                priors[k] = float(v)
            except Exception:
                continue
    return priors


def load_llm(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    for obj in read_jsonl(path):
        k = triple_key(obj["h_id"], obj["r_id"], obj["t_id"])
        out[k] = obj
    return out


def main() -> None:
    args = parse_args()
    kb = InternalGraphKB.from_dataset(args.dataset)

    suspects_path = Path(args.suspects)
    suspects = list(read_jsonl(suspects_path))
    suspects = suspects[: max(0, args.top_n)]

    llm = load_llm(Path(args.llm_scores) if args.llm_scores else None)
    kge = load_kge_priors(
        Path(args.kge_prior_true) if args.kge_prior_true else None,
        Path(args.kge_prior_anomaly) if args.kge_prior_anomaly else None,
    )

    # quick precision if labels exist
    has_label = any("label" in o for o in suspects)
    if has_label:
        tp = sum(1 for o in suspects if int(o.get("label", 0)) == 1)
        print(f"# top_n={len(suspects)} positives={tp} precision={tp/len(suspects):.4f}")

    if args.format == "tsv":
        header = [
            "rank",
            "fused_score",
            "gold_label",
            "kge_prior",
            "llm_label",
            "llm_conf",
            "best_choice",
            "triple_human",
        ]
        if args.show_reason:
            header.append("llm_reason")
        print("\t".join(header))

    for i, obj in enumerate(suspects, 1):
        h, r, t = str(obj["h_id"]), str(obj["r_id"]), str(obj["t_id"])
        k = triple_key(h, r, t)
        fused_score = obj.get("score")
        gold = obj.get("label", "")
        kge_prior = kge.get(k, None)

        llm_obj = llm.get(k, {})
        parsed = llm_obj.get("parsed") if isinstance(llm_obj, dict) else None
        if not isinstance(parsed, dict):
            llm_label, llm_conf, best_choice, reason = "", "", "", ""
        else:
            llm_label = str(parsed.get("label", ""))
            llm_conf = str(parsed.get("confidence", ""))
            best_choice = str(parsed.get("best_choice", ""))
            reason = str(parsed.get("reason", "")).replace("\n", " ").strip()

        triple_human = kb.format_triple(h, r, t)

        row = {
            "rank": i,
            "fused_score": fused_score,
            "gold_label": gold,
            "kge_prior": kge_prior,
            "llm_label": llm_label,
            "llm_conf": llm_conf,
            "best_choice": best_choice,
            "triple_human": triple_human,
        }
        if args.show_reason:
            row["llm_reason"] = reason

        if args.format == "jsonl":
            print(json.dumps(row, ensure_ascii=False))
        else:
            vals = [
                str(row.get("rank", "")),
                "" if row.get("fused_score") is None else str(row.get("fused_score")),
                str(row.get("gold_label", "")),
                "" if row.get("kge_prior") is None else str(row.get("kge_prior")),
                str(row.get("llm_label", "")),
                str(row.get("llm_conf", "")),
                str(row.get("best_choice", "")),
                str(row.get("triple_human", "")),
            ]
            if args.show_reason:
                vals.append(str(row.get("llm_reason", "")))
            print("\t".join(vals))


if __name__ == "__main__":
    main()

