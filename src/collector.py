from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import arxiv
import feedparser

from src.logger import setup_logger

log = setup_logger("collector")

RSS_URL = "http://rss.arxiv.org/rss/{category}"
BATCH_SIZE = 100  # arxiv API page size limit
INCLUDE_ANNOUNCE_TYPES = {"new", "cross"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    comments: str | None
    subjects: list[str]
    pdf_url: str
    html_url: str
    published_date: str
    announce_type: str = "new"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Paper:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def get_previous_business_day(reference_date: date | None = None) -> date:
    d = reference_date or date.today()
    d -= timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d


def _fetch_rss_ids(category: str = "cs.AI") -> list[dict]:
    """Fetch today's announced paper IDs from the arXiv RSS feed.

    Returns a list of dicts with 'arxiv_id' and 'announce_type'.
    Filters to new submissions and cross-listings only.
    """
    url = RSS_URL.format(category=category)
    log.info(f"Fetching RSS feed: {url}")
    feed = feedparser.parse(url)

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"RSS feed parse error: {feed.bozo_exception}")

    papers = []
    for entry in feed.entries:
        announce_type = entry.get("arxiv_announce_type", "unknown")
        if announce_type not in INCLUDE_ANNOUNCE_TYPES:
            continue

        link = entry.get("link", "")
        arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else ""
        if not arxiv_id:
            continue

        papers.append({"arxiv_id": arxiv_id, "announce_type": announce_type})

    log.info(
        f"RSS feed: {len(feed.entries)} total entries, "
        f"{len(papers)} after filtering to {INCLUDE_ANNOUNCE_TYPES}"
    )
    return papers


def _fetch_metadata_batch(arxiv_ids: list[str]) -> dict[str, arxiv.Result]:
    """Fetch full metadata for a batch of arxiv IDs using the arxiv API."""
    client = arxiv.Client(page_size=BATCH_SIZE, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(id_list=arxiv_ids)

    results = {}
    for result in client.results(search):
        clean_id = result.entry_id.split("/")[-1]
        # Strip version suffix (e.g., "2602.16714v1" -> "2602.16714")
        if "v" in clean_id:
            clean_id = clean_id[: clean_id.rindex("v")]
        results[clean_id] = result

    return results


def _build_paper(arxiv_id: str, result: arxiv.Result, announce_type: str) -> Paper:
    """Build a Paper dataclass from an arxiv API result."""
    clean_id = arxiv_id
    if "v" in clean_id:
        clean_id = clean_id[: clean_id.rindex("v")]

    return Paper(
        arxiv_id=clean_id,
        title=result.title.strip(),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip(),
        comments=result.comment,
        subjects=result.categories,
        pdf_url=f"https://arxiv.org/pdf/{clean_id}",
        html_url=f"https://arxiv.org/html/{clean_id}",
        published_date=result.published.date().isoformat(),
        announce_type=announce_type,
    )


def fetch_papers(category: str = "cs.AI") -> list[Paper]:
    """Fetch today's announced papers with full metadata.

    Uses a hybrid approach:
    1. RSS feed for the correct daily announcement list (IDs + announce type)
    2. arxiv API batch lookup for full metadata (authors, comments, etc.)
    """
    rss_entries = _fetch_rss_ids(category)

    if not rss_entries:
        log.warning("No papers found in RSS feed — may be a holiday or weekend")
        return []

    # Batch fetch metadata from arxiv API
    all_ids = [e["arxiv_id"] for e in rss_entries]
    announce_map = {e["arxiv_id"]: e["announce_type"] for e in rss_entries}

    log.info(f"Fetching metadata for {len(all_ids)} papers via arxiv API...")

    # Process in batches to respect API limits
    metadata: dict[str, arxiv.Result] = {}
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i : i + BATCH_SIZE]
        log.info(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} papers")
        batch_results = _fetch_metadata_batch(batch)
        metadata.update(batch_results)
        if i + BATCH_SIZE < len(all_ids):
            time.sleep(3)

    # Build Paper objects, matching RSS IDs to API results
    papers = []
    missing = 0
    for arxiv_id in all_ids:
        if arxiv_id in metadata:
            paper = _build_paper(arxiv_id, metadata[arxiv_id], announce_map[arxiv_id])
            papers.append(paper)
        else:
            missing += 1
            log.debug(f"No metadata found for {arxiv_id}, skipping")

    if missing:
        log.warning(f"{missing} papers had no metadata — skipped")

    # Deduplicate by arxiv_id
    seen = set()
    unique_papers = []
    for paper in papers:
        if paper.arxiv_id not in seen:
            seen.add(paper.arxiv_id)
            unique_papers.append(paper)

    dupes = len(papers) - len(unique_papers)
    if dupes:
        log.info(f"Removed {dupes} duplicate papers")

    log.info(f"Final paper count: {len(unique_papers)}")
    return unique_papers
