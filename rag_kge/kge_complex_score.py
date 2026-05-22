"""
Score triples with a trained ComplEx model and output an anomaly prior.

anomaly_prior = 1 - sigmoid(complex_score)

Usage:
python3 -m rag_kge.kge_complex_score \
  --dataset wn18rr \
  --model_path rag_kge/output/wn18rr/complex.pt \
  --triples_path dataset/wn18rr/true.jsonl \
  --output_path rag_kge/output/wn18rr/kge_prior_true_complex.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score triples using ComplEx as anomaly prior.")
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
    ent_re = ckpt["ent_re"].to(device)
    ent_im = ckpt["ent_im"].to(device)
    rel_re = ckpt["rel_re"].to(device)
    rel_im = ckpt["rel_im"].to(device)
    return ent2idx, rel2idx, ent_re, ent_im, rel_re, rel_im


def score_triple(
    ent2idx,
    rel2idx,
    ent_re,
    ent_im,
    rel_re,
    rel_im,
    h_id: str,
    r_id: str,
    t_id: str,
) -> Optional[float]:
    if h_id not in ent2idx or t_id not in ent2idx or r_id not in rel2idx:
        return None
    h_i = ent2idx[h_id]
    r_i = rel2idx[r_id]
    t_i = ent2idx[t_id]

    h_re = ent_re[h_i]
    h_im = ent_im[h_i]
    r_re_v = rel_re[r_i]
    r_im_v = rel_im[r_i]
    t_re = ent_re[t_i]
    t_im = ent_im[t_i]

    s = (
        h_re * r_re_v * t_re
        + h_im * r_re_v * t_im
        + h_re * r_im_v * t_im
        - h_im * r_im_v * t_re
    ).sum()
    p = torch.sigmoid(s).item()
    return 1.0 - p


def main() -> None:
    args = parse_args()
    ent2idx, rel2idx, ent_re, ent_im, rel_re, rel_im = load_model(Path(args.model_path), args.device)

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
            prior = score_triple(ent2idx, rel2idx, ent_re, ent_im, rel_re, rel_im, h_id, r_id, t_id)
            rec = {"h_id": h_id, "r_id": r_id, "t_id": t_id, "kge_prior": prior}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} scored triples to {out_path}")


if __name__ == "__main__":
    main()
