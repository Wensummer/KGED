"""
Train a lightweight ComplEx KGE model as an anomaly prior.

ComplEx can model asymmetric relations (unlike DistMult), which is often
helpful on WN18RR / FB15k-237.

We train on dataset/<dataset>/train.txt using simple negative sampling.

Outputs:
  rag_kge/output/<dataset>/complex.pt

Usage:
python3 -m rag_kge.kge_complex_train --dataset wn18rr --epochs 10 --dim 200 --batch_size 2048
python3 -m rag_kge.kge_complex_train --dataset wn18rr --splits all --epochs 10 --dim 200 --batch_size 2048
python3 -m rag_kge.kge_complex_train --dataset wn18rr --triples_path graph_kb_noisy/wn18rr/edges.jsonl --epochs 10 --dim 200 --batch_size 2048
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ComplEx KGE prior.")
    p.add_argument("--dataset", type=str, default="wn18rr", choices=["wn18rr", "fb15k-237", "nell995"])
    p.add_argument("--dim", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--neg_ratio", type=int, default=1, help="Negatives per positive.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--splits",
        type=str,
        default="train",
        help='Comma-separated dataset splits to train on (train,dev,test) or "all".',
    )
    p.add_argument(
        "--triples_path",
        type=str,
        default=None,
        help="Optional custom triples file (.txt or .jsonl with h_id,r_id,t_id). If set, overrides --splits.",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the checkpoint (default: rag_kge/output/<dataset>/complex.pt).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for training, e.g. "cpu" or "cuda". Default is cpu to avoid CUDA init warnings.',
    )
    return p.parse_args()


def load_support(dataset_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    ent = json.load(open(dataset_dir / "support" / "entity.json"))
    rel = json.load(open(dataset_dir / "support" / "relation.json"))
    ent_ids = sorted(ent.keys())
    rel_ids = sorted(rel.keys())
    ent2idx = {eid: i for i, eid in enumerate(ent_ids)}
    rel2idx = {rid: i for i, rid in enumerate(rel_ids)}
    return ent2idx, rel2idx


def load_triples_from_txt(path: Path) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            triples.append((h, r, t))
    return triples


def load_triples_from_jsonl(path: Path) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            h = obj.get("h_id")
            r = obj.get("r_id")
            t = obj.get("t_id")
            if h is None or r is None or t is None:
                continue
            triples.append((str(h), str(r), str(t)))
    return triples


def load_triples_from_path(path: Path) -> List[Tuple[str, str, str]]:
    if path.suffix.lower() == ".jsonl":
        return load_triples_from_jsonl(path)
    return load_triples_from_txt(path)


def load_triples(dataset_dir: Path, splits: List[str]) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for split in splits:
        split = split.strip().lower()
        if not split:
            continue
        if split == "all":
            return load_triples(dataset_dir, ["train", "dev", "test"])

        # Support both dev/valid naming.
        candidates = [dataset_dir / f"{split}.txt"]
        if split == "dev":
            candidates.append(dataset_dir / "valid.txt")
        if split == "valid":
            candidates.append(dataset_dir / "dev.txt")

        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Split file not found for split={split!r} under {dataset_dir}")
        triples.extend(load_triples_from_txt(path))
    return triples


class TripleDataset(Dataset):
    def __init__(self, triples_idx: List[Tuple[int, int, int]]):
        self.triples = triples_idx

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        return torch.tensor(self.triples[idx], dtype=torch.long)


class ComplEx(nn.Module):
    def __init__(self, num_ent: int, num_rel: int, dim: int):
        super().__init__()
        self.ent_re = nn.Embedding(num_ent, dim)
        self.ent_im = nn.Embedding(num_ent, dim)
        self.rel_re = nn.Embedding(num_rel, dim)
        self.rel_im = nn.Embedding(num_rel, dim)
        for emb in (self.ent_re, self.ent_im, self.rel_re, self.rel_im):
            nn.init.xavier_uniform_(emb.weight)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        h_re = self.ent_re(triples[:, 0])
        h_im = self.ent_im(triples[:, 0])
        r_re = self.rel_re(triples[:, 1])
        r_im = self.rel_im(triples[:, 1])
        t_re = self.ent_re(triples[:, 2])
        t_im = self.ent_im(triples[:, 2])

        # Re(<h, r, conj(t)>) = sum(h_re*r_re*t_re + h_im*r_re*t_im + h_re*r_im*t_im - h_im*r_im*t_re)
        s = (
            h_re * r_re * t_re
            + h_im * r_re * t_im
            + h_re * r_im * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)
        return s

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        return self.score_triples(triples)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "dataset" / args.dataset
    out_dir = root / "rag_kge" / "output" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_path) if args.output_path else out_dir / "complex.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ent2idx, rel2idx = load_support(dataset_dir)
    if args.triples_path:
        triples_path = Path(args.triples_path)
        if not triples_path.exists():
            raise FileNotFoundError(f"--triples_path not found: {triples_path}")
        splits = ["custom"]
        triples = load_triples_from_path(triples_path)
    else:
        splits_arg = args.splits.strip().lower()
        splits = ["train", "dev", "test"] if splits_arg == "all" else [s.strip() for s in splits_arg.split(",")]
        splits = [s for s in splits if s]
        triples = load_triples(dataset_dir, splits)
    triples_idx: List[Tuple[int, int, int]] = []
    for h, r, t in triples:
        if h not in ent2idx or t not in ent2idx or r not in rel2idx:
            continue
        triples_idx.append((ent2idx[h], rel2idx[r], ent2idx[t]))

    num_ent = len(ent2idx)
    num_rel = len(rel2idx)
    print(f"Train triples: {len(triples_idx)}; entities={num_ent}, relations={num_rel}")

    ds = TripleDataset(triples_idx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = ComplEx(num_ent, num_rel, args.dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in dl:
            batch = batch.to(args.device)
            pos = batch
            pos_score = model(pos)

            # negatives: corrupt head or tail
            negs = []
            for _ in range(args.neg_ratio):
                neg = pos.clone()
                mask = torch.rand(len(pos), device=args.device) < 0.5
                rand_ent = torch.randint(0, num_ent, (len(pos),), device=args.device)
                neg[mask, 0] = rand_ent[mask]
                neg[~mask, 2] = rand_ent[~mask]
                negs.append(neg)
            neg = torch.cat(negs, dim=0)
            neg_score = model(neg)

            loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            loss = loss + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())

        avg = total_loss / max(1, len(dl))
        print(f"epoch {epoch}/{args.epochs} loss={avg:.4f}")

    payload = {
        "model_type": "complex",
        "dim": args.dim,
        "splits": splits,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "neg_ratio": args.neg_ratio,
        "ent2idx": ent2idx,
        "rel2idx": rel2idx,
        "ent_re": model.ent_re.weight.detach().cpu(),
        "ent_im": model.ent_im.weight.detach().cpu(),
        "rel_re": model.rel_re.weight.detach().cpu(),
        "rel_im": model.rel_im.weight.detach().cpu(),
    }
    torch.save(payload, out_path)
    print(f"Saved ComplEx model to {out_path}")


if __name__ == "__main__":
    main()
