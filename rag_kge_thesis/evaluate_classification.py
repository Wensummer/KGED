from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate thesis-stage KG error detection as binary classification."
    )
    parser.add_argument("--true_scores", type=str, required=True, help="JSONL scores for true triples.")
    parser.add_argument("--anomaly_scores", type=str, required=True, help="JSONL scores for anomaly triples.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Predict anomaly if error_prob >= threshold. If omitted, search over --threshold_grid.",
    )
    parser.add_argument(
        "--threshold_grid",
        type=str,
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated thresholds used when --threshold is omitted.",
    )
    parser.add_argument(
        "--unknown_score",
        type=float,
        default=0.5,
        help="Error probability assigned to unknown predictions.",
    )
    parser.add_argument(
        "--missing_score",
        type=float,
        default=0.5,
        help="Error probability assigned when parsed output is missing.",
    )
    parser.add_argument(
        "--alternative_bonus",
        type=float,
        default=0.1,
        help='If best_choice is not "candidate", raise error_prob by this bonus.',
    )
    parser.add_argument(
        "--show_stage_stats",
        action="store_true",
        help="Print counts by final stage and stage-wise positive rates.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default=None,
        help="Optional JSONL path for merged predictions with gold labels and error probabilities.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parsed_to_error_prob(
    parsed: Optional[Dict],
    unknown_score: float,
    missing_score: float,
    alternative_bonus: float,
) -> float:
    if not isinstance(parsed, dict):
        return missing_score

    label = str(parsed.get("label", "")).lower()
    confidence = _safe_float(parsed.get("confidence"), 0.0)
    confidence = max(0.0, min(1.0, confidence))

    if label == "incorrect":
        score = 0.5 + 0.5 * confidence
    elif label == "correct":
        score = 0.5 - 0.5 * confidence
    elif label == "unknown":
        consistency = _safe_float(parsed.get("evidence_consistency"), unknown_score)
        score = 0.5 * unknown_score + 0.5 * consistency
    else:
        score = missing_score

    best_choice = str(parsed.get("best_choice", "")).strip().lower()
    if best_choice and best_choice != "candidate":
        score += alternative_bonus

    return max(0.0, min(1.0, score))


def load_predictions(
    path: Path,
    gold_label: int,
    unknown_score: float,
    missing_score: float,
    alternative_bonus: float,
) -> List[Dict]:
    records: List[Dict] = []
    for obj in read_jsonl(path):
        parsed = obj.get("parsed")
        error_prob = parsed_to_error_prob(
            parsed=parsed,
            unknown_score=unknown_score,
            missing_score=missing_score,
            alternative_bonus=alternative_bonus,
        )
        records.append(
            {
                "h_id": str(obj["h_id"]),
                "r_id": str(obj["r_id"]),
                "t_id": str(obj["t_id"]),
                "gold_label": gold_label,
                "final_stage": obj.get("final_stage", ""),
                "heuristic_stop": bool(obj.get("heuristic_stop", False)),
                "error_prob": error_prob,
                "parsed": parsed,
            }
        )
    return records


def compute_metrics(records: List[Dict], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for record in records:
        pred = 1 if record["error_prob"] >= threshold else 0
        gold = int(record["gold_label"])
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1

    total = max(1, tp + fp + tn + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def search_best_threshold(records: List[Dict], threshold_grid: List[float]) -> Dict[str, float]:
    best = None
    for threshold in threshold_grid:
        metrics = compute_metrics(records, threshold)
        if best is None:
            best = metrics
            continue
        if metrics["f1"] > best["f1"]:
            best = metrics
            continue
        if metrics["f1"] == best["f1"] and metrics["accuracy"] > best["accuracy"]:
            best = metrics
            continue
        if (
            metrics["f1"] == best["f1"]
            and metrics["accuracy"] == best["accuracy"]
            and abs(metrics["threshold"] - 0.5) < abs(best["threshold"] - 0.5)
        ):
            best = metrics
    assert best is not None
    return best


def print_stage_stats(records: List[Dict]) -> None:
    stage_counter = Counter(record.get("final_stage") or "unknown" for record in records)
    print("Stage statistics:")
    for stage, count in sorted(stage_counter.items()):
        stage_records = [record for record in records if (record.get("final_stage") or "unknown") == stage]
        positive_rate = sum(record["gold_label"] for record in stage_records) / max(1, len(stage_records))
        avg_prob = sum(record["error_prob"] for record in stage_records) / max(1, len(stage_records))
        print(
            f"  stage={stage:>20} count={count:>6} "
            f"gold_positive_rate={positive_rate:.4f} avg_error_prob={avg_prob:.4f}"
        )


def export_predictions(records: List[Dict], threshold: float, export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as f:
        for record in records:
            pred_label = 1 if record["error_prob"] >= threshold else 0
            payload = dict(record)
            payload["pred_label"] = pred_label
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = load_predictions(
        path=Path(args.true_scores),
        gold_label=0,
        unknown_score=args.unknown_score,
        missing_score=args.missing_score,
        alternative_bonus=args.alternative_bonus,
    )
    records.extend(
        load_predictions(
            path=Path(args.anomaly_scores),
            gold_label=1,
            unknown_score=args.unknown_score,
            missing_score=args.missing_score,
            alternative_bonus=args.alternative_bonus,
        )
    )

    if not records:
        raise SystemExit("No predictions loaded.")

    if args.threshold is not None:
        chosen = compute_metrics(records, args.threshold)
        print("Fixed-threshold metrics:")
    else:
        grid = [_safe_float(item.strip(), 0.5) for item in args.threshold_grid.split(",") if item.strip()]
        chosen = search_best_threshold(records, grid)
        print("Best threshold on provided grid:")

    print(f"  threshold={chosen['threshold']:.4f}")
    print(f"  accuracy={chosen['accuracy']:.4f}")
    print(f"  precision={chosen['precision']:.4f}")
    print(f"  recall={chosen['recall']:.4f}")
    print(f"  f1={chosen['f1']:.4f}")
    print(
        f"  confusion_matrix=tp:{chosen['tp']} fp:{chosen['fp']} "
        f"tn:{chosen['tn']} fn:{chosen['fn']}"
    )

    if args.show_stage_stats:
        print_stage_stats(records)

    if args.export_path:
        export_predictions(records, chosen["threshold"], Path(args.export_path))
        print(f"Exported merged predictions to {args.export_path}")


if __name__ == "__main__":
    main()
