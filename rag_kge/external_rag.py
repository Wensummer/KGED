import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import re


def _tokenize(text: str) -> List[str]:
    """Very simple tokenizer: lowercase and keep alnum tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class ExternalDoc:
    doc_id: str
    entity_id: str
    entity_name: str
    title: str
    text: str
    source: str
    url: str
    time: str


class ExternalTextKB:
    """
    Lightweight external text KB for RAG.

    - Loads docs from external_kb/<dataset>/docs.jsonl
    - Provides:
        * docs_by_entity: lookup by entity_id
        * simple lexical search over text
    """

    def __init__(self, kb_root: Path):
        self.kb_root = kb_root
        self.docs: List[ExternalDoc] = []
        self.docs_by_id: Dict[str, ExternalDoc] = {}
        self.docs_by_entity: Dict[str, List[ExternalDoc]] = {}
        self.doc_tokens: Dict[str, List[str]] = {}

        self._load()

    @classmethod
    def from_dataset(cls, dataset: str, kb_dir: str = "external_kb") -> "ExternalTextKB":
        root = Path(__file__).resolve().parent.parent  # repo root
        kb_root = root / kb_dir / dataset
        return cls(kb_root)

    def _load(self) -> None:
        docs_path = self.kb_root / "docs.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(
                f"Expected external docs at {docs_path}. "
                "Run rag_kge/build_external_kb.py first."
            )

        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                doc = ExternalDoc(
                    doc_id=str(data.get("doc_id", "")),
                    entity_id=str(data.get("entity_id", "")),
                    entity_name=str(data.get("entity_name", "")),
                    title=str(data.get("title", "")),
                    text=str(data.get("text", "")),
                    source=str(data.get("source", "")),
                    url=str(data.get("url", "")) if data.get("url") is not None else "",
                    time=str(data.get("time", "")) if data.get("time") is not None else "",
                )
                self.docs.append(doc)
                self.docs_by_id[doc.doc_id] = doc
                if doc.entity_id:
                    self.docs_by_entity.setdefault(doc.entity_id, []).append(doc)

        # build token cache for lexical search
        for doc in self.docs:
            self.doc_tokens[doc.doc_id] = _tokenize(doc.text)

    # ------------------------------------------------------------------
    # Retrieval APIs
    # ------------------------------------------------------------------
    def get_docs_for_entity(self, entity_id: str, top_k: int = 5) -> List[ExternalDoc]:
        docs = self.docs_by_entity.get(entity_id, [])
        return docs[:top_k]

    def rank_docs_for_entities(
        self, entity_ids: Iterable[str], query_text: str, top_k: int = 8
    ) -> List[ExternalDoc]:
        """
        Simple lexical re-ranker over the docs for the given entities.
        Scores by token overlap with query_text, plus a small bonus if the
        doc.entity_id is exactly one of the requested IDs.
        """
        q_tokens = _tokenize(query_text)
        q_set = set(q_tokens)

        candidates: List[ExternalDoc] = []
        for ent_id in entity_ids:
            candidates.extend(self.docs_by_entity.get(ent_id, []))

        scored: List[Tuple[ExternalDoc, float]] = []
        for doc in candidates:
            tokens = self.doc_tokens.get(doc.doc_id, [])
            if not tokens:
                continue
            overlap = len(q_set.intersection(tokens)) if q_set else 0
            score = (overlap / len(q_set)) if q_set else 0.0
            if doc.entity_id in entity_ids:
                score += 0.1  # small bonus for exact entity match
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            return [d for d, _ in scored[:top_k]]

        # fallback: if query_text has no useful tokens or yields no overlap,
        # still return a few docs for the requested entities (entity docs are
        # already filtered by entity_id, so they're usually relevant).
        fallback: List[ExternalDoc] = []
        for ent_id in entity_ids:
            fallback.extend(self.docs_by_entity.get(ent_id, []))
        return fallback[:top_k]

    def search_by_text(self, query: str, top_k: int = 5) -> List[Tuple[ExternalDoc, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)

        scored: List[Tuple[ExternalDoc, float]] = []
        for doc in self.docs:
            tokens = self.doc_tokens.get(doc.doc_id, [])
            if not tokens:
                continue
            overlap = len(q_set.intersection(tokens))
            if overlap == 0:
                continue
            score = overlap / len(q_set)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Rendering helper
    # ------------------------------------------------------------------
    def render_docs_as_text(
        self, docs: Iterable[ExternalDoc], max_chars: int = 1200
    ) -> str:
        """
        Serialize a small set of documents into a compact text block for LLM.
        """
        lines: List[str] = []
        total = 0

        for idx, doc in enumerate(docs, start=1):
            header = f"{idx}. [{doc.source}] {doc.title} (entity={doc.entity_name})"
            body = doc.text.strip().replace("\n", " ")
            piece = header + "\n" + body
            if total + len(piece) > max_chars and lines:
                break
            lines.append(piece)
            total += len(piece) + 1

        return "\n\n".join(lines)


# ----------------------------------------------------------------------
# Simple CLI for testing
# ----------------------------------------------------------------------

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: retrieve external text docs for an entity or text query."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wn18rr",
        choices=["wn18rr", "fb15k-237", "nell995"],
        help="Dataset name under external_kb/.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--entity_id", type=str, help="Entity ID to retrieve docs for.")
    group.add_argument("--query", type=str, help="Free text query.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of docs to show.")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    kb = ExternalTextKB.from_dataset(args.dataset)

    if args.entity_id:
        docs = kb.get_docs_for_entity(args.entity_id, top_k=args.top_k)
        print(kb.render_docs_as_text(docs))
    else:
        results = kb.search_by_text(args.query, top_k=args.top_k)
        docs = [d for d, _ in results]
        print(kb.render_docs_as_text(docs))


if __name__ == "__main__":
    main()
