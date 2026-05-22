import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import time
import requests
from urllib.parse import quote

# Descriptive User-Agent per Wikimedia policy:
# https://meta.wikimedia.org/wiki/User-Agent_policy
WIKIPEDIA_USER_AGENT = (
    "CCA-RAG-KG/0.1 (academic research; contact: you@example.com)"
)
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an external text knowledge base for RAG from web sources."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wn18rr",
        choices=["wn18rr", "fb15k-237", "nell995"],
        help="Which dataset under ./dataset/ to use.",
    )
    parser.add_argument(
        "--max_entities",
        type=int,
        default=500,
        help="Maximum number of entities to fetch (for quick prototyping).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="external_kb",
        help="Directory to store the external knowledge base.",
    )
    parser.add_argument(
        "--include_wikipedia",
        action="store_true",
        default=True,
        help="Fetch Wikipedia summaries (on by default).",
    )
    parser.add_argument(
        "--news_api_key",
        type=str,
        default=None,
        help="If provided (or via env NEWSAPI_KEY), fetch recent news via NewsAPI.org.",
    )
    parser.add_argument(
        "--news_max_per_entity",
        type=int,
        default=2,
        help="Max news articles per entity when using NewsAPI.",
    )
    return parser.parse_args()


def load_entities(entity_json: Path) -> Dict[str, Dict]:
    with entity_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_wikipedia_summary(title: str) -> Optional[Dict]:
    """
    Fetch a short summary for an entity from Wikipedia.
    This uses the public REST API and may fail for many titles.
    """
    # basic normalization
    normalized = title.replace(" ", "_")
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + quote(normalized, safe="_")
    )
    headers = {
        "User-Agent": WIKIPEDIA_USER_AGENT,
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return None

    if resp.status_code == 404:
        # page not found
        return None
    if resp.status_code in (429, 503):
        # too many requests / service unavailable: back off a bit
        time.sleep(1.0)
        return None
    if resp.status_code != 200:
        # other errors
        return None

    try:
        data = resp.json()
    except ValueError:
        return None

    text = data.get("extract", "") or ""
    if not text.strip():
        return None

    return {
        "title": data.get("title", title),
        "text": text,
        "url": data.get("content_urls", {})
        .get("desktop", {})
        .get("page", url),
        "source": "wikipedia",
        "time": None,
    }


def fetch_news_articles(query: str, api_key: str, max_items: int = 2) -> List[Dict]:
    """
    Fetch recent news articles from NewsAPI.org.
    Returns a list of dicts with text/title/url/source/time.
    """
    if not api_key:
        return []
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_items,
    }
    headers = {
        "User-Agent": WIKIPEDIA_USER_AGENT,
        "Accept": "application/json",
        "X-Api-Key": api_key,
    }
    try:
        resp = requests.get(NEWSAPI_ENDPOINT, params=params, headers=headers, timeout=10)
    except Exception:
        return []

    if resp.status_code != 200:
        return []

    try:
        data = resp.json()
    except ValueError:
        return []

    articles = []
    for art in data.get("articles", [])[:max_items]:
        content = art.get("content") or art.get("description") or ""
        if not content:
            continue
        articles.append(
            {
                "title": art.get("title") or query,
                "text": content,
                "url": art.get("url"),
                "source": f"newsapi:{(art.get('source') or {}).get('name','unknown')}",
                "time": art.get("publishedAt"),
            }
        )
    return articles


def split_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    """
    Simple paragraph-based chunking with a soft max length.
    """
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 1 > max_chars and current:
            parts.append(" ".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 1

    if current:
        parts.append(" ".join(current))

    return parts


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            json.dump(obj, f, ensure_ascii=True)
            f.write("\n")


def main() -> None:
    args = parse_args()

    # this file lives in rag_kge/, so repo root is one level up
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "dataset" / args.dataset
    support_dir = dataset_dir / "support"
    entity_json = support_dir / "entity.json"

    if not entity_json.exists():
        raise FileNotFoundError(f"Expected entity file at {entity_json}")

    output_root = root / args.output_dir / args.dataset
    docs_path = output_root / "docs.jsonl"

    entities = load_entities(entity_json)
    print(f"Loaded {len(entities)} entities from {entity_json}")

    docs: List[Dict] = []
    num_done = 0
    news_api_key = args.news_api_key or os.environ.get("NEWSAPI_KEY")

    for ent_id, info in entities.items():
        if num_done >= args.max_entities:
            break
        name = str(info.get("name", "")).strip()
        if not name:
            continue

        # Wikipedia
        if args.include_wikipedia:
            summary = fetch_wikipedia_summary(name)
            if summary:
                chunks = split_into_chunks(summary["text"])
                for idx, chunk in enumerate(chunks):
                    doc_id = f"{ent_id}::wiki::{idx}"
                    docs.append(
                        {
                            "doc_id": doc_id,
                            "entity_id": ent_id,
                            "entity_name": name,
                            "title": summary["title"],
                            "text": chunk,
                            "source": summary["source"],
                            "url": summary["url"],
                            "time": summary["time"],
                        }
                    )

        # NewsAPI (optional)
        if news_api_key:
            news_list = fetch_news_articles(name, news_api_key, args.news_max_per_entity)
            for idx, art in enumerate(news_list):
                chunks = split_into_chunks(art["text"])
                for c_idx, chunk in enumerate(chunks):
                    doc_id = f"{ent_id}::news::{idx}_{c_idx}"
                    docs.append(
                        {
                            "doc_id": doc_id,
                            "entity_id": ent_id,
                            "entity_name": name,
                            "title": art["title"],
                            "text": chunk,
                            "source": art["source"],
                            "url": art["url"],
                            "time": art["time"],
                        }
                    )

        num_done += 1
        if num_done % 50 == 0:
            print(f"Fetched external docs for {num_done} entities...")

    write_jsonl(docs_path, docs)
    print(f"Finished. Wrote {len(docs)} documents to {docs_path}")


if __name__ == "__main__":
    main()
