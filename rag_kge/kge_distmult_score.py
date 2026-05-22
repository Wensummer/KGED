"""
Score triples with a trained DistMult model and output an anomaly prior.

anomaly_prior = 1 - sigmoid(distmult_score)

Usage:
python3 -m rag_kge.kge_distmult_score \
  --dataset wn18rr \
  --model_path rag_kge/output/wn18rr/distmult.pt \
  --triples_path dataset/wn18rr/true.jsonl \
  --output_path rag_kge/output/wn18rr/kge_prior_true.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score triples using DistMult as anomaly prior.")
    p.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr", "fb15k-237", "nell995"])
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--triples_path", type=str, required=True, help="JSONL with h_id,r_id,t_id.")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for scoring, e.g. "cpu" or "cuda". Default is cpu to avoid CUDA init warnings.',
    )
    return p.parse_args()


def load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu")
    ent2idx = ckpt["ent2idx"]
    rel2idx = ckpt["rel2idx"]
    ent_w = ckpt["ent_weight"].to(device)
    rel_w = ckpt["rel_weight"].to(device)
    return ent2idx, rel2idx, ent_w, rel_w


def score_triple(ent2idx, rel2idx, ent_w, rel_w, h_id: str, r_id: str, t_id: str) -> Optional[float]:
    if h_id not in ent2idx or t_id not in ent2idx or r_id not in rel2idx:
        return None
    h = ent_w[ent2idx[h_id]]
    r = rel_w[rel2idx[r_id]]
    t = ent_w[ent2idx[t_id]]
    s = (h * r * t).sum()
    p = torch.sigmoid(s).item()
    return 1.0 - p


def main() -> None:
    args = parse_args()
    ent2idx, rel2idx, ent_w, rel_w = load_model(Path(args.model_path), args.device)

    in_path = Path(args.triples_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            h_id, r_id, t_id = obj["h_id"], obj["r_id"], obj["t_id"]
            prior = score_triple(ent2idx, rel2idx, ent_w, rel_w, h_id, r_id, t_id)
            rec = {"h_id": h_id, "r_id": r_id, "t_id": t_id, "kge_prior": prior}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} scored triples to {out_path}")


if __name__ == "__main__":
    main()
