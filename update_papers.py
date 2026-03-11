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
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import requests
import arxiv  # pip install arxiv

# ---------- Simple config (edit at top) ----------
KEYWORDS_FILE = "keywords.txt"
OUTPUT_README = "README.md"
METADATA_JSONL = "metadata.jsonl"
METADATA_MATCHED_JSONL = "metadata_matched.jsonl"
EMBEDS_JSONL = "embeds.jsonl"
BIORXIV_PAGE = 100
ARXIV_MAX_RESULTS = 150
ARXIV_DELAY = 3.0          # seconds (polite after arXiv)
REQUEST_TIMEOUT = 30      # seconds for API calls
ABSTRACT_SNIP = 500
POLITE_SLEEP = 0.4
BIO_RETRIES = 3
# ------------------------------------------------

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    link: str
    date: str
    source: str


# ---------- Utilities ----------
def half_month_window_for_date(d: Optional[datetime.date] = None) -> Tuple[datetime.date, datetime.date]:
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
            return [ln for ln in lines]  # keep original-case for reporting; matching will be regex-based (case-insensitive)
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


# ---------- Fetchers (bioRxiv with retries; arXiv modest default) ----------
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
            attempt += 1
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                j = resp.json()
                break
            except Exception as exc:
                logging.warning("bioRxiv request failed (attempt %d/%d): %s", attempt, BIO_RETRIES, exc)
                if attempt >= BIO_RETRIES:
                    logging.error("bioRxiv request failed after %d attempts; returning collected items.", BIO_RETRIES)
                    return items
                time.sleep(2 ** (attempt - 1))

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
        try:
            pub_date = res.published.date()
        except Exception:
            continue
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


# ---------- Keyword filtering (returns list of tuples (Paper, [matched_keywords])) ----------
def filter_by_keywords(papers: List[Paper], keywords: List[str]) -> List[Tuple[Paper, List[str]]]:
    """
    OR logic: return papers that match ANY keyword.
    Searches title + abstract. Case-insensitive, word-boundary regex to avoid accidental partial matches.
    """
    if not keywords:
        logging.info("No keywords provided; returning empty matched set.")
        return []

    # Build regex patterns (word-boundary). Escape keywords to treat punctuation literally.
    patterns = {k: re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keywords}

    matched: List[Tuple[Paper, List[str]]] = []
    for p in papers:
        text = f"{p.title} {p.abstract}"
        hits = [k for k, pat in patterns.items() if pat.search(text)]
        if hits:
            matched.append((p, hits))
    logging.info("Keyword matching: %d matched papers (of %d fetched).", len(matched), len(papers))
    return matched


# ---------- README writer (richer output) ----------
def write_readme(matched: List[Tuple[Paper, List[str]]], start: datetime.date, end: datetime.date, total_fetched: int, path: str = OUTPUT_README):
    header = f"# Papers {start.isoformat()} → {end.isoformat()}\n\n"
    header += f"_Updated: {datetime.date.today().isoformat()}_\n\n"
    header += f"**Total fetched:** {total_fetched}  \n"
    header += f"**Total matched:** {len(matched)}  \n\n"

    header += "| Date | Source | Title | Matched keywords | Link |\n"
    header += "|------|--------|-------|------------------|------|\n"

    lines = [header]

    if not matched:
        lines.append("| _No matching papers found in this window._ | | | | |")
    else:
        # sort by date desc
        for p, kws in sorted(matched, key=lambda x: x[0].date, reverse=True):
            t = p.title.replace("|", "╱")
            kwstr = ", ".join(kws)
            lines.append(f"| {p.date} | {p.source} | **{t}** | {kwstr} | [paper]({p.link}) |")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logging.info("Wrote README with %d matched papers", len(matched))


# ---------- Optional: embed abstracts (lazy import) ----------
def embed_abstracts_to_jsonl(matched: List[Tuple[Paper, List[str]]], out_path: str = EMBEDS_JSONL):
    """
    Embed abstracts (abstract-level) and write JSONL of items with vectors.
    Requires OPENAI_API_KEY in environment. This function is optional and lazily imports langchain.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot embed.")
    # lazy imports
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter

    splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=200)
    emb = OpenAIEmbeddings()
    out_items = []
    for p, kws in matched:
        text = (p.abstract or "").strip()
        if not text:
            continue
        chunks = splitter.split_text(text)
        vectors = emb.embed_documents(chunks)
        out_items.append({
            "id": p.id,
            "title": p.title,
            "link": p.link,
            "date": p.date,
            "source": p.source,
            "matched_keywords": kws,
            "chunks": [
                {"text_preview": (c[:256] + "...") if len(c) > 256 else c, "vector": v}
                for c, v in zip(chunks, vectors)
            ],
        })
    write_jsonl(out_items, out_path)
    logging.info("Embeddings written to %s", out_path)


# ---------- Optional: conservative PDF download for arXiv (lazy import) ----------
def download_arxiv_pdf_and_extract_text(p: Paper, out_dir: str = "pdf_texts") -> Optional[str]:
    """
    Only attempts arXiv PDFs by constructing the standard arXiv PDF URL.
    Returns extracted text (truncated) or None.
    """
    if p.source != "arXiv" or not p.id:
        return None
    pdf_url = f"https://arxiv.org/pdf/{p.id}.pdf"
    try:
        resp = requests.get(pdf_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"{p.id.replace('/', '_')}.pdf")
        with open(fname, "wb") as fh:
            fh.write(resp.content)
        # lazy import PyPDF2 only when needed
        import PyPDF2
        text_parts = []
        try:
            reader = PyPDF2.PdfReader(fname)
            for pg in reader.pages:
                pg_text = pg.extract_text() or ""
                text_parts.append(pg_text)
        finally:
            try:
                os.remove(fname)
            except Exception:
                pass
        return "\n".join(text_parts).strip()
    except Exception as e:
        logging.debug("PDF download/extract failed for %s: %s", p.id, e)
        return None


# ---------- CLI + main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Fetch bioRxiv + arXiv, filter by keywords, write README + metadata.")
    p.add_argument("--start", help="Start date (YYYY-MM-DD).", default=None)
    p.add_argument("--end", help="End date (YYYY-MM-DD).", default=None)
    p.add_argument("--with-embeds", action="store_true", help="Embed matched abstracts (requires OPENAI_API_KEY).")
    p.add_argument("--with-pdfs", action="store_true", help="Attempt to download arXiv PDFs for matched items (conservative).")
    p.add_argument("--arxiv-query", default="cat:q-bio", help="arXiv query string (default: cat:q-bio).")
    return p.parse_args()


def main():
    args = parse_args()

    # window
    if args.start and args.end:
        start = datetime.date.fromisoformat(args.start)
        end = datetime.date.fromisoformat(args.end)
    else:
        start, end = half_month_window_for_date()

    keywords = load_keywords()

    # fetch
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
    logging.info("Total fetched: %d (bioRxiv=%d, arXiv=%d)", len(all_papers), len(biorxiv), len(arx))

    # write full metadata
    write_jsonl([asdict(p) for p in all_papers], METADATA_JSONL)

    # filter and write matched metadata + readme
    matched = filter_by_keywords(all_papers, keywords)  # List[(Paper, [kw])]
    matched_items = []
    for p, kws in matched:
        d = asdict(p)
        d["matched_keywords"] = kws
        matched_items.append(d)
    if matched_items:
        write_jsonl(matched_items, METADATA_MATCHED_JSONL)
    else:
        # ensure old matched file is removed if no matches (optional)
        if os.path.exists(METADATA_MATCHED_JSONL):
            try:
                os.remove(METADATA_MATCHED_JSONL)
            except Exception:
                pass
    write_readme(matched, start, end, total_fetched=len(all_papers))

    # optional embeddings
    if args.with_embeds and matched:
        try:
            embed_abstracts_to_jsonl(matched, EMBEDS_JSONL)
        except Exception as e:
            logging.error("Embedding failed: %s", e)

    # optional PDF extraction (conservative)
    if args.with_pdfs and matched:
        os.makedirs("pdf_texts", exist_ok=True)
        for p, kws in matched:
            txt = download_arxiv_pdf_and_extract_text(p)
            if txt:
                safe = p.id.replace("/", "_").replace(":", "_")
                out_path = os.path.join("pdf_texts", f"{safe}.txt")
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(txt[:50_000])
                logging.info("Saved PDF text preview: %s", out_path)
            time.sleep(0.5)  # polite

    logging.info("Done.")


if __name__ == "__main__":
    main()
