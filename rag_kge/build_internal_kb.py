import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight internal graph knowledge base for GraphRAG."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wn18rr",
        choices=["wn18rr", "fb15k-237", "nell995"],
        help="Which dataset under ./dataset/ to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="graph_kb",
        help="Directory to store the constructed knowledge base.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev,test",
        help='Comma-separated splits to include in edges (train,dev,test) or "all".',
    )
    parser.add_argument(
        "--extra_edges",
        type=str,
        default=None,
        help="Optional extra triples file to append into edges (JSONL with h_id,r_id,t_id or TXT h\\tr\\tt).",
    )
    parser.add_argument(
        "--extra_split",
        type=str,
        default="extra",
        help='Split label to use for --extra_edges (default: "extra", e.g. "anomaly").',
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            # ensure_ascii keeps the file pure ASCII by escaping non-ASCII
            json.dump(obj, f, ensure_ascii=True)
            f.write("\n")


def build_nodes(entity_json: Path, relation_json: Path) -> List[Dict]:
    entities = load_json(entity_json)
    relations = load_json(relation_json)

    nodes: List[Dict] = []

    # entity nodes
    for ent_id, info in entities.items():
        name = str(info.get("name", "")).strip()
        desc = str(info.get("desc", "")).strip()
        text_parts = [p for p in (name, desc) if p]
        text = " ".join(text_parts)
        nodes.append(
            {
                "id": str(ent_id),
                "type": "entity",
                "name": name,
                "desc": desc,
                "text": text,
            }
        )

    # relation nodes
    for rel_id, info in relations.items():
        # relation.json may contain extra keys; we only care about "name"
        name = str(info.get("name", rel_id)).strip()
        text = name
        nodes.append(
            {
                "id": str(rel_id),
                "type": "relation",
                "name": name,
                "desc": "",
                "text": text,
            }
        )

    return nodes


def iter_triples(dataset_dir: Path, splits: List[str]) -> Iterable[Dict]:
    for split in splits:
        triple_path = dataset_dir / f"{split}.txt"
        if not triple_path.exists():
            continue
        with triple_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    h, r, t = line.split("\t")
                except ValueError:
                    # skip malformed lines
                    continue
                yield {
                    "h_id": h,
                    "r_id": r,
                    "t_id": t,
                    "split": split,
                }

def iter_extra_triples(path: Path, split: str) -> Iterable[Dict]:
    if path.suffix.lower() == ".jsonl":
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
                yield {"h_id": str(h), "r_id": str(r), "t_id": str(t), "split": split}
        return

    # default: tab-separated txt
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            yield {"h_id": h, "r_id": r, "t_id": t, "split": split}


def main() -> None:
    args = parse_args()

    # this file lives in rag_kge/, so repo root is one level up
    # e.g. /path/to/CCA-main/rag_kge/build_internal_kb.py -> /path/to/CCA-main
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "dataset" / args.dataset
    support_dir = dataset_dir / "support"

    entity_json = support_dir / "entity.json"
    relation_json = support_dir / "relation.json" 

    if not entity_json.exists() or not relation_json.exists():
        raise FileNotFoundError(
            f"Expected support files at {entity_json} and {relation_json}"
        )

    output_root = root / args.output_dir / args.dataset
    nodes_path = output_root / "nodes.jsonl"
    edges_path = output_root / "edges.jsonl"
    stats_path = output_root / "stats.json"

    print(f"Building internal KB for dataset '{args.dataset}'")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Output dir:  {output_root}")

    # build nodes
    nodes = build_nodes(entity_json, relation_json)
    write_jsonl(nodes_path, nodes)

    # build edges
    splits_arg = args.splits.strip().lower()
    splits = ["train", "dev", "test"] if splits_arg == "all" else [s.strip() for s in splits_arg.split(",")]
    splits = [s for s in splits if s]
    edges = list(iter_triples(dataset_dir, splits))
    if args.extra_edges:
        extra_path = Path(args.extra_edges)
        if not extra_path.exists():
            raise FileNotFoundError(f"--extra_edges not found: {extra_path}")
        edges.extend(list(iter_extra_triples(extra_path, args.extra_split)))
    write_jsonl(edges_path, edges)

    stats = {
        "dataset": args.dataset,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "splits": splits,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=True)

    print("Done.")
    print(f"Nodes written to: {nodes_path}")
    print(f"Edges written to: {edges_path}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
