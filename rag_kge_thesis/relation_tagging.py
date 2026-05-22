from __future__ import annotations

import re
from typing import List


def _normalize_relation_text(relation_id: str, relation_name: str) -> str:
    text = f"{relation_id} {relation_name}".lower()
    text = text.replace(":", " ").replace("/", " ").replace("_", " ").replace(".", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _add_if_any(text: str, tags: List[str], candidates: List[str], target_tag: str) -> None:
    if any(token in text for token in candidates) and target_tag not in tags:
        tags.append(target_tag)


def infer_relation_tags(dataset: str, relation_id: str, relation_name: str = "") -> List[str]:
    text = _normalize_relation_text(relation_id, relation_name or relation_id)
    compact = text.replace(" ", "")
    tags: List[str] = []

    _add_if_any(
        text,
        tags,
        [
            "hypernym",
            "hyponym",
            "subclass",
            "instance",
            "type of",
            "kind of",
            "parent",
            "child",
            "capital of",
            "administrative division",
            "first level division",
            "second level division",
        ],
        "hierarchical",
    )
    _add_if_any(
        compact,
        tags,
        [
            "suchas",
            "istypeof",
            "iskindof",
            "capitalof",
            "locatedwithin",
            "firstleveldivisionof",
            "secondleveldivision",
        ],
        "hierarchical",
    )

    _add_if_any(
        text,
        tags,
        [
            "part of",
            "has part",
            "member of",
            "contains",
            "partially contains",
            "including",
            "component",
            "ingredient",
            "roster",
            "team",
            "group membership",
        ],
        "compositional",
    )
    _add_if_any(
        compact,
        tags,
        [
            "partof",
            "haspart",
            "memberof",
            "contains",
            "including",
            "roster",
            "membership",
        ],
        "compositional",
    )

    _add_if_any(
        text,
        tags,
        [
            "topic",
            "domain",
            "genre",
            "category",
            "field",
            "subject",
            "industry",
            "profession",
            "religion",
            "sport",
            "language",
            "award",
            "music",
            "film",
            "tv",
            "book",
        ],
        "domain_topic",
    )

    _add_if_any(
        text,
        tags,
        [
            "date",
            "year",
            "month",
            "season",
            "tenure",
            "current",
            "former",
            "birth",
            "death",
            "release",
            "runtime",
            "founded",
            "founded by",
            "at date",
            "start",
            "end",
        ],
        "time_sensitive",
    )
    _add_if_any(
        compact,
        tags,
        [
            "atdate",
            "birth",
            "death",
            "releasedate",
            "runtime",
            "founded",
            "current",
            "former",
        ],
        "time_sensitive",
    )

    _add_if_any(
        text,
        tags,
        [
            "person",
            "country",
            "city",
            "state",
            "location",
            "language",
            "profession",
            "gender",
            "team",
            "organization",
            "company",
            "film",
            "music",
            "sport",
            "disease",
            "drug",
            "animal",
            "river",
            "university",
            "capital",
        ],
        "type_sensitive",
    )

    _add_if_any(
        text,
        tags,
        [
            "capital",
            "located in",
            "place of birth",
            "place of death",
            "headquartered",
            "ceo",
            "employ",
            "parent",
            "child",
            "part of",
            "contains",
            "member of",
            "country",
            "city",
            "state",
        ],
        "antisymmetric",
    )
    _add_if_any(
        compact,
        tags,
        [
            "capitalof",
            "locatedin",
            "bornin",
            "headquarteredin",
            "ceoof",
            "parent",
            "child",
            "partof",
            "contains",
            "memberof",
        ],
        "antisymmetric",
    )

    if "suchas" in compact and "hierarchical" not in tags:
        tags.append("hierarchical")
    if "alsoknownas" in compact and "symmetric_or_near_synonym" not in tags:
        tags.append("symmetric_or_near_synonym")
    if any(x in compact for x in ("collaborateswith", "friend", "sibling", "competeswith")):
        if "symmetric_or_near_synonym" not in tags:
            tags.append("symmetric_or_near_synonym")

    if dataset == "nell995":
        if relation_id.startswith("concept:") and "type_sensitive" not in tags:
            tags.append("type_sensitive")
        if any(x in compact for x in ("playsforteam", "coachesteam", "athleteplays", "team")) and "domain_topic" not in tags:
            tags.append("domain_topic")
    elif dataset == "fb15k-237":
        if relation_id.startswith("/people/") and "type_sensitive" not in tags:
            tags.append("type_sensitive")
        if relation_id.startswith("/location/") and "type_sensitive" not in tags:
            tags.append("type_sensitive")
        if relation_id.startswith("/film/") or relation_id.startswith("/music/") or relation_id.startswith("/tv/"):
            if "domain_topic" not in tags:
                tags.append("domain_topic")

    return tags


def merge_relation_tags(*tag_groups: List[str]) -> List[str]:
    merged: List[str] = []
    for group in tag_groups:
        for tag in group or []:
            tag_str = str(tag).strip()
            if not tag_str:
                continue
            if tag_str not in merged:
                merged.append(tag_str)
    return merged
