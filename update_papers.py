#!/usr/bin/env python3
"""
update_papers.py

Half-month paper fetcher based on Artur's hackathon code https://github.com/aozalevsky/structhunt/ #ты козел))))

Might be worth thinking about with https://docs.langchain.com/oss/python/integrations/document_loaders? need to revisit 

What it does:
- Fetches bioRxiv (JSON API) and arXiv (arxiv) for current half-month window
- Filters by keywords in keywords.txt (one per line, supports comments starting with #)
- Updates README.md (compact table) and metadata.jsonl (one JSON per paper)
- Optionally embeds abstracts (--with-embeds) and/or downloads PDFs (--with-pdfs); haven't tested PDF function or embedding functions

Usage examples:
  python update_papers.py                 # default run
  python update_papers.py --with-embeds   # also produce embeds.jsonl (requires OPENAI_API_KEY; or Gemini)
  python update_papers.py --start 2026-03-01 --end 2026-03-15
"""

from __future__ import annotations
import argparse
import calendar
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
import arxiv  # pip install arxiv

# ---------- Configurable defaults ----------
KEYWORDS_FILE = "keywords.txt"
OUTPUT_README = "README.md"
METADATA_JSONL = "metadata.jsonl"
EMBEDS_JSONL = "embeds.jsonl"
BIORXIV_PAGE = 100
ARXIV_MAX_RESULTS = 150
ARXIV_DELAY = 3.0          # seconds, polite pause after arXiv fetch
REQUEST_TIMEOUT = 30       # increased timeout to avoid ReadTimeouts
ABSTRACT_SNIP = 500
POLITE_SLEEP = 0.4         # seconds between paged requests (bioRxiv)
BIO_RETRIES = 3
# -------------------------------------------

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    link: str
    date: str
    source: str


def half_month_window_for_date(d: Optional[datetime.date] = None):
    d = d or datetime.date.today()
    y, m, day = d.year, d.month, d.day
    if day <= 15:
        return datetime.date(y, m, 1), datetime.date(y, m, 15)
    last = calendar.monthrange(y, m)[1]
    return datetime.date(y, m, 16), datetime.date(y, m, last)


def load_keywords(path: str = KEYWORDS_FILE) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
            return [ln.lower() for ln in lines]
    except FileNotFoundError:
        logging.warning("keywords.txt not found — no keywords loaded (nothing will match).")
        return []


def snippet(text: str, n: int = ABSTRACT_SNIP) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def write_jsonl(items: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    logging.info("Wrote %d lines to %s", len(items), path)


# ---------- Fetchers with retries ----------
def fetch_biorxiv(start: datetime.date, end: datetime.date, server: str = "biorxiv") -> List[Paper]:
    base = "https://api.biorxiv.org/details"
    s_iso, e_iso = start.isoformat(), end.isoformat()
    items: List[Paper] = []
    cursor = 0
    session = requests.Session()
    logging.info("Fetching bioRxiv: %s → %s", s_iso, e_iso)

    while True:
        url = f"{base}/{server}/{s_iso}/{e_iso}/{cursor}"
        attempt = 0
        while True:
            try:
                attempt += 1
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                j = resp.json()
                break
            except Exception as exc:
                logging.warning("bioRxiv request failed (attempt %d/%d): %s", attempt, BIO_RETRIES, exc)
                if attempt >= BIO_RETRIES:
                    logging.error("bioRxiv request failed after %d attempts; aborting bioRxiv fetch for this window.", BIO_RETRIES)
                    return items
                backoff = 2 ** (attempt - 1)
                logging.info("Sleeping %ds before retrying bioRxiv...", backoff)
                time.sleep(backoff)

        coll = j.get("collection", [])
        if not coll:
            break
        for it in coll:
            doi = it.get("doi", "") or it.get("url", "")
            link = it.get("url") or (f"https://www.biorxiv.org/content/{doi}" if doi else "")
            items.append(Paper(
                id=(doi or link),
                title=(it.get("title") or "").strip(),
                abstract=(it.get("abstract") or "").strip(),
                link=link,
                date=(it.get("date") or ""),
                source="bioRxiv",
            ))
        cursor += BIORXIV_PAGE
        msgs = j.get("messages", [])
        if msgs and msgs[0].get("count") and len(items) >= int(msgs[0]["count"]):
            break
        time.sleep(POLITE_SLEEP)
    logging.info("bioRxiv fetched: %d", len(items))
    return items


def fetch_arxiv(start: datetime.date, end: datetime.date, query: str = "cat:q-bio") -> List[Paper]:
    logging.info("Querying arXiv (%s)...", query)
    s = arxiv.Search(
        query=query,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    items: List[Paper] = []
    for res in s.results():
        pub_date = res.published.date()
        if start <= pub_date <= end:
            items.append(Paper(
                id=(res.get_short_id() or res.entry_id),
                title=(res.title or "").strip(),
                abstract=(res.summary or "").strip(),
                link=res.entry_id,
                date=pub_date.isoformat(),
                source="arXiv",
            ))
    time.sleep(ARXIV_DELAY)
    logging.info("arXiv fetched (post-filter): %d", len(items))
    return items


def filter_by_keywords(papers: List[Paper], keywords: List[str]) -> List[Paper]:
    if not keywords:
        return []
    out = []
    kw_lower = [k.lower() for k in keywords]
    for p in papers:
        txt = (p.abstract or "").lower()
        if any(k in txt for k in kw_lower):
            out.append(p)
    return out


def write_readme(papers: List[Paper], start: datetime.date, end: datetime.date, path: str = OUTPUT_README):
    header = f"# Papers {start.isoformat()} → {end.isoformat()}\n\n_Updated: {datetime.date.today().isoformat()}_\n\n"
    header += "| Title | Abstract (snippet) | Link |\n|---|---|---|\n"
    lines = [header]
    if not papers:
        lines.append("| _No matching papers found in this window._ | | |")
    else:
        for p in sorted(papers, key=lambda x: x.date, reverse=True):
            t = p.title.replace("|", "╱")
            lines.append(f"| **{t}** | {snippet(p.abstract)} | [paper]({p.link}) |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logging.info("Wrote README with %d papers", len(papers))


# Embedding and PDF helpers omitted here for brevity (unchanged)...
# For full script include embed_abstracts_to_jsonl and download_pdf_and_extract_text from your current file.

# ---------- Main (same CLI) ----------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Fetch bioRxiv + arXiv for half-month windows, filter by keywords, optionally embed.")
    p.add_argument("--start", help="Start date (YYYY-MM-DD). Default: start of current half-month", default=None)
    p.add_argument("--end", help="End date (YYYY-MM-DD). Default: end of current half-month", default=None)
    p.add_argument("--with-embeds", action="store_true", help="Also embed abstracts (requires OPENAI_API_KEY).")
    p.add_argument("--with-pdfs", action="store_true", help="Attempt to download PDFs and extract full text (disabled by default).")
    p.add_argument("--arxiv-query", default="cat:q-bio", help="arXiv query string (default: cat:q-bio).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.start and args.end:
        start = datetime.date.fromisoformat(args.start)
        end = datetime.date.fromisoformat(args.end)
    else:
        start, end = half_month_window_for_date()

    keywords = load_keywords()

    try:
        biorxiv = fetch_biorxiv(start, end)
    except Exception as e:
        logging.error("bioRxiv fetch error: %s", e)
        biorxiv = []

    try:
        arx = fetch_arxiv(start, end, query=args.arxiv_query)
    except Exception as e:
        logging.error("arXiv fetch error: %s", e)
        arx = []

    all_papers = biorxiv + arx
    metadata_items = [asdict(p) for p in all_papers]
    write_jsonl(metadata_items, METADATA_JSONL)

    matched = filter_by_keywords(all_papers, keywords)
    write_readme(matched, start, end)

    # Optional steps (embedding / pdf extraction) unchanged — keep gated by flags.

    logging.info("Done.")


if __name__ == "__main__":
    main()
