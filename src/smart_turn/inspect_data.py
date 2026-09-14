"""
Fast metadata-only inspection of the smart-turn dataset via the HF
datasets-server REST API, instead of streaming/downloading parquet shards.

Why: the parquet shards interleave raw audio bytes with metadata, so pulling
a shard just to read `language`/`synthetic`/`dataset` columns means
downloading hundreds of MB you don't need - and is what's causing the
timeouts. This hits datasets-server's /rows endpoint instead, which returns
JSON rows (audio field comes back as a URL, not raw bytes) and paginates
cheaply.

Use this to answer: how much hin/eng data exists, real vs synthetic, before
committing to any parquet-shard streaming/downloading at all.
"""

import os
import time
from collections import Counter
from typing import Optional

import requests

DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"
API_BASE = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # datasets-server max per request is typically 100

HF_TOKEN = os.environ.get("HF_TOKEN")


def _headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def fetch_rows(offset: int, length: int = PAGE_SIZE, retries: int = 6) -> list:
    params = {
        "dataset": DATASET_NAME,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(API_BASE, params=params, headers=_headers(), timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                print(f"  429 rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()["rows"]
        except (requests.exceptions.RequestException, KeyError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} after error: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch rows at offset {offset} after {retries} retries")


def inspect_language_distribution(n: int = 5000) -> dict:
    lang_counts = Counter()
    synthetic_by_lang = Counter()
    dataset_src_by_lang = {}

    offset = 0
    scanned = 0
    while scanned < n:
        length = min(PAGE_SIZE, n - scanned)
        rows = fetch_rows(offset, length)
        if not rows:
            break  # exhausted dataset

        for row in rows:
            r = row["row"]
            lang = r["language"]
            lang_counts[lang] += 1
            if r["synthetic"]:
                synthetic_by_lang[lang] += 1
            dataset_src_by_lang.setdefault(lang, Counter())[r["dataset"]] += 1

        scanned += len(rows)
        offset += len(rows)
        print(f"  scanned {scanned}/{n}...")
        time.sleep(0.5)  # be polite between successful requests too

    return {
        "n_scanned": scanned,
        "lang_counts": dict(lang_counts),
        "synthetic_count_by_lang": dict(synthetic_by_lang),
        "dataset_source_by_lang": {k: dict(v) for k, v in dataset_src_by_lang.items()},
    }


if __name__ == "__main__":
    stats = inspect_language_distribution(n=5000)
    print(f"\nScanned {stats['n_scanned']} rows")
    print("Language counts:", stats["lang_counts"])
    print("Synthetic counts by language:", stats["synthetic_count_by_lang"])
    for lang in ("hin", "eng"):
        if lang in stats["dataset_source_by_lang"]:
            print(f"  {lang} sources:", stats["dataset_source_by_lang"][lang])