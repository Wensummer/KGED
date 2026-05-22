#!/usr/bin/env python3
"""Prepare NELL-995 files in this repository's dataset layout.

The repository expects:

  dataset/<name>/
    train.txt
    dev.txt
    test.txt
    true.jsonl
    support/entity.json
    support/relation.json

This script imports common NELL-995 raw layouts where triples are stored as
tab/space separated h r t files. It can also generate minimal entity/relation
metadata from the triples when no support JSON files are available.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Triple = Tuple[str, str, str]


SPLIT_CANDIDATES = {
    "train": ("train.txt", "train.tsv", "train2id.txt"),
    "dev": ("dev.txt", "valid.txt", "validation.txt", "valid.tsv", "dev.tsv", "valid2id.txt"),
    "test": ("test.txt", "test.tsv", "test2id.txt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local NELL-995 raw directory to dataset/nell995."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Directory containing NELL-995 raw split files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/nell995",
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--entity_json",
        type=str,
        default=None,
        help="Optional existing entity support JSON to reuse.",
    )
    parser.add_argument(
        "--relation_json",
        type=str,
        default=None,
        help="Optional existing relation support JSON to reuse.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def find_split_file(raw_dir: Path, split: str) -> Path:
    for name in SPLIT_CANDIDATES[split]:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    names = ", ".join(SPLIT_CANDIDATES[split])
    raise FileNotFoundError(f"Could not find {split} split under {raw_dir}; tried: {names}")


def normalize_name(identifier: str) -> str:
    text = identifier.strip()
    if text.startswith("/"):
        text = text.strip("/").split("/")[-1]
    if ":" in text:
        text = text.split(":")[-1]
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or identifier


def load_openke_mapping(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first and line.isdigit():
                first = False
                continue
            first = False
            parts = line.split()
            if len(parts) < 2:
                continue
            raw_id = parts[-1]
            raw_name = " ".join(parts[:-1])
            mapping[raw_id] = raw_name
    return mapping


def is_count_header(line: str) -> bool:
    return bool(line.strip()) and line.strip().isdigit()


def read_triples(
    path: Path,
    entity_id_to_name: Optional[Dict[str, str]] = None,
    relation_id_to_name: Optional[Dict[str, str]] = None,
) -> List[Triple]:
    triples: List[Triple] = []
    entity_id_to_name = entity_id_to_name or {}
    relation_id_to_name = relation_id_to_name or {}
    with path.open("r", encoding="utf-8") as f:
        first_content = True
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if first_content and is_count_header(line):
                first_content = False
                continue
            first_content = False
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed triple in {path}:{line_no}: {line!r}")

            # OpenKE train2id/valid2id/test2id format is: head_id tail_id relation_id.
            # The repository format is: head relation tail.
            looks_openke = (
                path.name.endswith("2id.txt")
                or (parts[0] in entity_id_to_name and parts[1] in entity_id_to_name and parts[2] in relation_id_to_name)
            )
            if looks_openke:
                h = entity_id_to_name.get(parts[0], parts[0])
                t = entity_id_to_name.get(parts[1], parts[1])
                r = relation_id_to_name.get(parts[2], parts[2])
            else:
                h, r, t = (part.strip() for part in parts)

            if not h or not r or not t:
                raise ValueError(f"Empty field in {path}:{line_no}: {line!r}")
            triples.append((h, r, t))
    return triples


def write_triples(path: Path, triples: Sequence[Triple], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_entity_support(triples_by_split: Dict[str, List[Triple]], provided: Optional[Dict]) -> Dict:
    if provided is not None:
        return provided
    entity_ids = sorted({h for triples in triples_by_split.values() for h, _, _ in triples}.union(
        {t for triples in triples_by_split.values() for _, _, t in triples}
    ))
    return {
        ent_id: {
            "name": normalize_name(ent_id),
            "desc": normalize_name(ent_id),
        }
        for ent_id in entity_ids
    }


def build_relation_support(triples_by_split: Dict[str, List[Triple]], provided: Optional[Dict]) -> Dict:
    if provided is not None:
        return provided
    relation_ids = sorted({r for triples in triples_by_split.values() for _, r, _ in triples})
    return {
        rel_id: {
            "name": normalize_name(rel_id),
        }
        for rel_id in relation_ids
    }


def write_json(path: Path, obj: Dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_true_jsonl(path: Path, triples_by_split: Dict[str, List[Triple]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for split in ("train", "dev", "test"):
            for h, r, t in triples_by_split[split]:
                f.write(
                    json.dumps(
                        {"h_id": h, "r_id": r, "t_id": t, "split": split},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    entity_id_to_name = load_openke_mapping(raw_dir / "entity2id.txt")
    relation_id_to_name = load_openke_mapping(raw_dir / "relation2id.txt")

    triples_by_split: Dict[str, List[Triple]] = {}
    for split in ("train", "dev", "test"):
        split_path = find_split_file(raw_dir, split)
        triples_by_split[split] = read_triples(
            split_path,
            entity_id_to_name=entity_id_to_name,
            relation_id_to_name=relation_id_to_name,
        )
        write_triples(output_dir / f"{split}.txt", triples_by_split[split], args.overwrite)

    entity_support = build_entity_support(triples_by_split, load_json(args.entity_json))
    relation_support = build_relation_support(triples_by_split, load_json(args.relation_json))
    write_json(output_dir / "support" / "entity.json", entity_support, args.overwrite)
    write_json(output_dir / "support" / "relation.json", relation_support, args.overwrite)
    write_true_jsonl(output_dir / "true.jsonl", triples_by_split, args.overwrite)

    stats = {
        "dataset": "nell995",
        "num_entities": len(entity_support),
        "num_relations": len(relation_support),
        "num_train": len(triples_by_split["train"]),
        "num_dev": len(triples_by_split["dev"]),
        "num_test": len(triples_by_split["test"]),
        "num_triples": sum(len(v) for v in triples_by_split.values()),
    }
    write_json(output_dir / "stats.json", stats, args.overwrite)

    print(f"Prepared NELL-995 at {output_dir}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
