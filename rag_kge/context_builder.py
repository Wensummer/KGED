import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .internal_graphrag import InternalGraphKB
from .external_rag import ExternalTextKB


@dataclass
class Triple:
    h_id: str
    r_id: str
    t_id: str


@dataclass
class ContextPieces:
    triple: Triple
    alternatives_text: List[str]
    relation_definition: str
    graph_text: str
    external_text: str

    def to_prompt(self) -> str:
        """
        Build a contrastive prompt:
        - provide relation definition
        - compare candidate against plausible alternatives
        """
        cand_str = f"({self.triple.h_id}, {self.triple.r_id}, {self.triple.t_id})"
        parts: List[str] = []
        parts.append(
            "You are checking whether a candidate knowledge graph triple is factually correct.\n"
            "Internal graph evidence may contain errors and should not be treated as ground truth.\n"
            "Compare the candidate with alternative triples that share similar structure.\n"
            "If evidence is weak or conflicting, prefer 'incorrect' or 'unknown' with low confidence.\n"
            "Return ONLY a JSON object on a single line; do not include any other text.\n"
        )
        parts.append("RELATION DEFINITION:")
        parts.append(self.relation_definition.strip() + "\n")
        parts.append(f"CANDIDATE TRIPLE: {cand_str}\n")
        if self.alternatives_text:
            parts.append("ALTERNATIVE TRIPLES (from KG, for comparison):")
            parts.extend(self.alternatives_text)
            parts.append("")
        parts.append("INTERNAL GRAPH EVIDENCE (neighborhood excluding candidate):")
        parts.append(self.graph_text.strip() + "\n")
        if self.external_text.strip():
            parts.append("EXTERNAL TEXT EVIDENCE:")
            parts.append(self.external_text.strip() + "\n")
        parts.append(
            "Return ONLY JSON with keys: label, confidence, reason, best_choice.\n"
            ' - label in ["correct","incorrect","unknown"].\n'
            ' - best_choice should be "candidate" or one of ["A1","A2",...].\n'
            ' - confidence in [0,1].\n'
            ' - reason concise.\n'
        )
        return "\n".join(parts)


WN18RR_REL_DEFS: dict = {
    "_hypernym": "hypernym: the tail is a more general concept/category of the head. In other words, head is a kind of tail.",
    "_instance_hypernym": "instance hypernym: the tail is a class/category, and the head is a specific instance of that class.",
    "_member_meronym": "member meronym: the tail is a member of the group denoted by the head.",
    "_has_part": "has part: the head concept/entity contains the tail as a part or component.",
    "_also_see": "also see: the head is related to the tail as a loosely associated or recommended related concept.",
    "_similar_to": "similar to: head and tail are semantically similar concepts/synsets.",
    "_derivationally_related_form": "derivationally related form: head and tail are different word forms derived from each other (e.g., noun/verb/adjective variants).",
    "_verb_group": "verb group: head and tail are verbs in the same semantic group, often near-synonyms.",
    "_synset_domain_topic_of": "domain topic of: the head belongs to the topical domain denoted by the tail.",
    "_member_of_domain_usage": "member of domain usage: the head is used in the usage domain denoted by the tail (e.g., slang, regional usage).",
    "_member_of_domain_region": "member of domain region: the head belongs to or is used in the region/domain denoted by the tail.",
}


def get_relation_definition(dataset: str, r_id: str, kb_internal: InternalGraphKB) -> str:
    """
    Get a natural language definition for a relation.
    Falls back to relation name if no hand-written definition.
    """
    r_name = kb_internal.relations.get(r_id)
    r_name = r_name.name if r_name else r_id
    if dataset == "wn18rr":
        return WN18RR_REL_DEFS.get(r_id, f"{r_name}: relation between head and tail.")
    return f"{r_name}: relation between head and tail."


def build_query_text(dataset: str, h_id: str, r_id: str, t_id: str, kb_internal: InternalGraphKB) -> str:
    """
    Build a human-readable query string for external retrieval.

    Using raw IDs (e.g., WN18RR synset IDs) makes lexical retrieval fail.
    """
    h = kb_internal.entities.get(h_id)
    t = kb_internal.entities.get(t_id)
    r = kb_internal.relations.get(r_id)
    h_name = h.name if h and h.name else h_id
    t_name = t.name if t and t.name else t_id
    r_name = r.name if r and r.name else r_id
    # Keep it short; entity docs are already scoped by entity_id.
    return f"{h_name} {r_name} {t_name}"


def build_context_for_triple(
    dataset: str,
    triple: Triple,
    hops: int = 1,
    max_edges: int = 50,
    top_k_external: int = 8,
    kb_internal: Optional[InternalGraphKB] = None,
    kb_external: Optional[ExternalTextKB] = None,
) -> ContextPieces:
    """
    Build internal+external evidence text for a given triple.

    If kb_internal / kb_external are not provided, they will be created
    from the given dataset name using default paths.
    """
    if kb_internal is None:
        kb_internal = InternalGraphKB.from_dataset(dataset)
    if kb_external is None:
        kb_external = ExternalTextKB.from_dataset(dataset)

    # internal graph context (exclude candidate from neighborhood)
    graph_text = kb_internal.render_subgraph_as_text(
        h_id=triple.h_id,
        r_id=triple.r_id,
        t_id=triple.t_id,
        hops=hops,
        max_edges=max_edges,
        include_descriptions=True,
        exclude_candidate_from_neighborhood=True,
    )

    # external text context: take docs for head and tail entities
    # external text context: rank docs for head+tail using a simple lexical re-ranker
    query_text = build_query_text(dataset, triple.h_id, triple.r_id, triple.t_id, kb_internal)
    ranked_docs = kb_external.rank_docs_for_entities(
        entity_ids=[triple.h_id, triple.t_id],
        query_text=query_text,
        top_k=top_k_external,
    )
    external_text = kb_external.render_docs_as_text(ranked_docs, max_chars=1600)

    # alternatives for contrastive prompt
    alt_tuples = kb_internal.get_alternative_triples(
        h_id=triple.h_id, r_id=triple.r_id, t_id=triple.t_id, max_alternatives=5, seed=0
    )
    alternatives_text: List[str] = []
    for i, (h, r, t) in enumerate(alt_tuples, 1):
        human = kb_internal.format_triple(h, r, t)
        alternatives_text.append(f"A{i}: {human}  [ids=({h}, {r}, {t})]")
    rel_def = get_relation_definition(dataset, triple.r_id, kb_internal)

    return ContextPieces(
        triple=triple,
        alternatives_text=alternatives_text,
        relation_definition=rel_def,
        graph_text=graph_text,
        external_text=external_text,
    )


# ----------------------------------------------------------------------
# Simple CLI demo: print prompt for a triple
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined internal+external context for a triple."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wn18rr",
        choices=["wn18rr", "fb15k-237", "nell995"],
        help="Dataset name for internal/external KBs.",
    )
    parser.add_argument("--h_id", type=str, required=True, help="Head entity ID")
    parser.add_argument("--r_id", type=str, required=True, help="Relation ID")
    parser.add_argument("--t_id", type=str, required=True, help="Tail entity ID")
    parser.add_argument("--hops", type=int, default=1, help="Graph hops for internal context")
    parser.add_argument(
        "--max_edges",
        type=int,
        default=50,
        help="Maximum number of edges in internal subgraph",
    )
    parser.add_argument(
        "--top_k_external",
        type=int,
        default=4,
        help="Number of external docs for head and tail to include",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    triple = Triple(h_id=args.h_id, r_id=args.r_id, t_id=args.t_id)
    ctx = build_context_for_triple(
        dataset=args.dataset,
        triple=triple,
        hops=args.hops,
        max_edges=args.max_edges,
        top_k_external=args.top_k_external,
    )
    print(ctx.to_prompt())


if __name__ == "__main__":
    main()
