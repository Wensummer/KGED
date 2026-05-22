"""
Evaluate Precision@Top-K and Recall@Top-K for KG error detection.

Positives are anomaly triples (errors).

Supports:
 - KGE-only (e.g., DistMult anomaly prior)
 - LLM-only (from score_triples_ollama outputs)
 - Fused score: alpha * kge_prior + (1-alpha) * llm_error_prob
 - Alpha grid search for fusion

Example (KGE-only):
python3 -m rag_kge.evaluate_topk \
  --true_triples dataset/wn18rr/true.jsonl \
  --anomaly_triples dataset/wn18rr/mixture_anomaly/5/anomaly.jsonl \
  --kge_prior_true rag_kge/output/wn18rr/kge_prior_true.jsonl \
  --kge_prior_anomaly rag_kge/output/wn18rr/kge_prior_anomaly.jsonl

Example (Fusion with LLM outputs):
python3 -m rag_kge.evaluate_topk \
  --true_triples dataset/wn18rr/true.jsonl \
  --anomaly_triples dataset/wn18rr/mixture_anomaly/5/anomaly.jsonl \
  --kge_prior_true rag_kge/output/wn18rr/kge_prior_true.jsonl \
  --kge_prior_anomaly rag_kge/output/wn18rr/kge_prior_anomaly.jsonl \
  --llm_true rag_kge/output/wn18rr/ollama_scores_true_new.jsonl \
  --llm_anomaly rag_kge/output/wn18rr/ollama_scores_anomaly_new.jsonl \
  --alpha_grid 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Precision/Recall@Top-K for KG error detection.")
    p.add_argument("--true_triples", type=str, required=True)
    p.add_argument("--anomaly_triples", type=str, required=True)
    p.add_argument("--kge_prior_true", type=str, required=True)
    p.add_argument("--kge_prior_anomaly", type=str, required=True)
    p.add_argument(
        "--llm_scores",
        type=str,
        default=None,
        help="Single JSONL file from score_triples_ollama containing both true+anomaly triples (optional).",
    )
    p.add_argument("--llm_true", type=str, default=None, help="LLM scores JSONL for true triples (optional).")
    p.add_argument("--llm_anomaly", type=str, default=None, help="LLM scores JSONL for anomaly triples (optional).")
    p.add_argument(
        "--k_list",
        type=str,
        default="0.01,0.02,0.03,0.04,0.05",
        help="Comma-separated top-K proportions (e.g., 0.01=1%).",
    )
    p.add_argument(
        "--unknown_score",
        type=float,
        default=0.7,
        help="LLM error_prob used when label=unknown.",
    )
    p.add_argument(
        "--missing_score",
        type=float,
        default=0.5,
        help="Score used when a source score is missing.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fusion weight for KGE prior (alpha*kge + (1-alpha)*llm). If omitted, only KGE-only/LLM-only printed.",
    )
    p.add_argument(
        "--alpha_grid",
        type=str,
        default=None,
        help="Comma-separated fusion alphas to evaluate.",
    )
    p.add_argument(
        "--fusion",
        type=str,
        default="linear",
        choices=["linear", "mult"],
        help="Fusion method: linear=alpha*kge+(1-alpha)*llm; mult=kge*(1+beta*(llm-0.5)).",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.65,
        help="Beta for mult fusion: kge*(1+beta*(llm-0.5)).",
    )
    p.add_argument(
        "--beta_grid",
        type=str,
        default=None,
        help="Comma-separated betas to evaluate for mult fusion.",
    )
    p.add_argument(
        "--export_topk",
        type=float,
        default=None,
        help="If set, export top-K suspects for the chosen score (kge/llm/fused) to --export_path.",
    )
    p.add_argument("--export_path", type=str, default=None)
    p.add_argument(
        "--export_score",
        type=str,
        default="fused",
        choices=["kge", "llm", "fused"],
        help="Which score to export for --export_topk.",
    )
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


def load_labels(true_path: Path, anomaly_path: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for obj in read_jsonl(true_path):
        k = triple_key(obj["h_id"], obj["r_id"], obj["t_id"])
        labels[k] = 0
    for obj in read_jsonl(anomaly_path):
        k = triple_key(obj["h_id"], obj["r_id"], obj["t_id"])
        labels[k] = 1
    return labels


def load_kge_prior(true_prior: Path, anomaly_prior: Path) -> Dict[str, float]:
    priors: Dict[str, float] = {}
    for obj in read_jsonl(true_prior):
        k = triple_key(obj["h_id"], obj["r_id"], obj["t_id"])
        priors[k] = obj.get("kge_prior")
    for obj in read_jsonl(anomaly_prior):
        k = triple_key(obj["h_id"], obj["r_id"], obj["t_id"])
        priors[k] = obj.get("kge_prior")
    # normalize missing/None to None and then to missing_score later
    cleaned: Dict[str, float] = {}
    for k, v in priors.items():
        if v is None:
            continue
        try:
            cleaned[k] = float(v)
        except Exception:
            continue
    return cleaned


def llm_error_prob(parsed: Optional[Dict], unknown_score: float, missing_score: float) -> float:
    if not isinstance(parsed, dict):
        return missing_score
    label = (parsed.get("label") or "").lower()
    conf = parsed.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    best_choice = parsed.get("best_choice")

    # Map label+confidence to a score in [0,1] but keep low-confidence predictions near 0.5:
    # - incorrect -> (0.5, 1.0]
    # - correct   -> [0.0, 0.5)
    if label == "incorrect":
        base = 0.5 + 0.5 * conf
    elif label == "correct":
        base = 0.5 - 0.5 * conf
    elif label == "unknown":
        base = unknown_score
    else:
        base = missing_score

    # Contrastive signal: if the model picks an alternative as "best_choice",
    # treat the candidate as likely incorrect (even if label is noisy).
    if isinstance(best_choice, str) and best_choice.strip().lower() not in ("", "candidate"):
        base = max(base, 0.5 + 0.5 * conf)
    return base


def load_llm_scores(path: Path, unknown_score: float, missing_score: float) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for obj in read_jsonl(path):
        k = triple_key(obj["h_id"], obj["r_id"], obj["t_id"])
        parsed = obj.get("parsed")
        scores[k] = llm_error_prob(parsed, unknown_score=unknown_score, missing_score=missing_score)
    return scores


def eval_topk(scores: Dict[str, float], labels: Dict[str, int], k_list: List[float]) -> List[Tuple[float, int, float, float]]:
    items = []
    for k, y in labels.items():
        s = scores.get(k)
        if s is None:
            continue
        items.append((s, y))
    items.sort(key=lambda x: x[0], reverse=True)
    n = len(items)
    pos = sum(y for _, y in items)
    rows = []
    for k_prop in k_list:
        k = max(1, int(math.ceil(k_prop * n)))
        top = items[:k]
        tp = sum(y for _, y in top)
        prec = tp / k
        rec = tp / pos if pos else 0.0
        rows.append((k_prop, k, prec, rec))
    return rows


def print_rows(title: str, rows: List[Tuple[float, int, float, float]]) -> None:
    print(title)
    for k_prop, k, prec, rec in rows:
        print(f"  K={k_prop*100:.0f}%, k={k}, precision={prec:.4f}, recall={rec:.4f}")


def fuse_scores(
    labels: Dict[str, int],
    kge: Dict[str, float],
    llm: Dict[str, float],
    alpha: float,
    missing_score: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in labels.keys():
        kge_s = kge.get(key, missing_score)
        llm_s = llm.get(key)
        # two-stage friendly: if no LLM score exists for this triple,
        # keep KGE score unchanged instead of diluting it with a constant.
        if llm_s is None:
            out[key] = kge_s
        else:
            out[key] = alpha * kge_s + (1.0 - alpha) * llm_s
    return out


def fuse_scores_mult(
    labels: Dict[str, int],
    kge: Dict[str, float],
    llm: Dict[str, float],
    beta: float,
    missing_score: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in labels.keys():
        kge_s = kge.get(key, missing_score)
        llm_s = llm.get(key)
        if llm_s is None:
            out[key] = kge_s
        else:
            out[key] = kge_s * (1.0 + beta * (llm_s - 0.5))
    return out


def export_topk(
    labels: Dict[str, int],
    scores: Dict[str, float],
    k_prop: float,
    out_path: Path,
) -> None:
    items = []
    for key, y in labels.items():
        s = scores.get(key)
        if s is None:
            continue
        items.append((s, key, y))
    items.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(math.ceil(k_prop * len(items))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s, key, y in items[:k]:
            h, r, t = key.split("|")
            f.write(json.dumps({"h_id": h, "r_id": r, "t_id": t, "label": y, "score": s}) + "\n")
    print(f"Exported top {k} suspects to {out_path}")


def main() -> None:
    args = parse_args()
    true_path = Path(args.true_triples)
    anomaly_path = Path(args.anomaly_triples)
    labels = load_labels(true_path, anomaly_path)
    k_list = [float(x) for x in args.k_list.split(",") if x.strip()]

    kge = load_kge_prior(Path(args.kge_prior_true), Path(args.kge_prior_anomaly))
    # impute missing to missing_score for evaluation
    kge_scores = {k: (kge.get(k, args.missing_score)) for k in labels.keys()}
    rows = eval_topk(kge_scores, labels, k_list)
    print_rows("KGE prior (anomaly_prior) metrics:", rows)

    llm_scores = None
    llm = None
    if args.llm_scores:
        llm = load_llm_scores(Path(args.llm_scores), args.unknown_score, args.missing_score)
    elif args.llm_true and args.llm_anomaly:
        llm_t = load_llm_scores(Path(args.llm_true), args.unknown_score, args.missing_score)
        llm_a = load_llm_scores(Path(args.llm_anomaly), args.unknown_score, args.missing_score)
        llm = {**llm_t, **llm_a}

    if llm is not None:
        llm_scores = {k: llm.get(k, args.missing_score) for k in labels.keys()}
        rows_llm = eval_topk(llm_scores, labels, k_list)
        print_rows("LLM error_prob metrics:", rows_llm)

        if args.fusion == "linear":
            if args.beta_grid is not None or args.beta != 0.65:
                print("Note: --beta/--beta_grid are ignored when --fusion=linear.")
            if args.alpha is not None:
                fused = fuse_scores(labels, kge_scores, llm, args.alpha, args.missing_score)
                rows_f = eval_topk(fused, labels, k_list)
                print_rows(f"Fused metrics (linear, alpha={args.alpha:.2f}):", rows_f)
            if args.alpha_grid:
                alphas = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
                print("Alpha grid search (fused, linear):")
                for a in alphas:
                    fused = fuse_scores(labels, kge_scores, llm, a, args.missing_score)
                    rows_f = eval_topk(fused, labels, k_list)
                    r5 = next((rec for kp, _, _, rec in rows_f if abs(kp - 0.05) < 1e-9), rows_f[-1][3])
                    print(f"  alpha={a:.2f} recall@5%={r5:.4f}")
        else:
            if args.alpha_grid is not None or args.alpha is not None:
                print("Note: --alpha/--alpha_grid are ignored when --fusion=mult.")
            if args.beta is not None:
                fused = fuse_scores_mult(labels, kge_scores, llm, args.beta, args.missing_score)
                rows_f = eval_topk(fused, labels, k_list)
                print_rows(f"Fused metrics (mult, beta={args.beta:.3f}):", rows_f)
            if args.beta_grid:
                betas = [float(x) for x in args.beta_grid.split(",") if x.strip()]
                print("Beta grid search (fused, mult):")
                for b in betas:
                    fused = fuse_scores_mult(labels, kge_scores, llm, b, args.missing_score)
                    rows_f = eval_topk(fused, labels, k_list)
                    r5 = next((rec for kp, _, _, rec in rows_f if abs(kp - 0.05) < 1e-9), rows_f[-1][3])
                    print(f"  beta={b:.3f} recall@5%={r5:.4f}")

    # export suspects
    if args.export_topk is not None:
        if not args.export_path:
            raise SystemExit("--export_path is required when --export_topk is set.")
        out = Path(args.export_path)
        if args.export_score == "kge":
            export_topk(labels, kge_scores, args.export_topk, out)
        elif args.export_score == "llm":
            if llm_scores is None:
                raise SystemExit("Need --llm_scores or --llm_true/--llm_anomaly to export llm scores.")
            export_topk(labels, llm_scores, args.export_topk, out)
        else:
            if llm is None:
                raise SystemExit("Need --llm_scores (or --llm_true/--llm_anomaly) to export fused scores.")
            if args.fusion == "linear":
                if args.alpha is None:
                    raise SystemExit("Need --alpha when --fusion=linear and exporting fused scores.")
                fused = fuse_scores(labels, kge_scores, llm, args.alpha, args.missing_score)
            else:
                fused = fuse_scores_mult(labels, kge_scores, llm, args.beta, args.missing_score)
            export_topk(labels, fused, args.export_topk, out)


if __name__ == "__main__":
    main()
