"""ArXiv AI Digest — Main Pipeline Orchestrator.

Runs the full pipeline: collect → rank → analyze → compose → send.
Set DRY_RUN=true to skip email delivery.
"""

from __future__ import annotations

import os
import time
from datetime import date

from dotenv import load_dotenv

from src.analyzer import analyze_top_papers
from src.blurb_generator import generate_blurbs
from src.collector import Paper, fetch_papers
from src.email_composer import (
    compose_email,
    compose_error_email,
    compose_quiet_day_email,
)
from src.email_sender import send_email
from src.logger import setup_logger
from src.ranker import rank_papers

log = setup_logger("pipeline")


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() in ("true", "1", "yes")


def _build_deep_summary_data(
    papers: list[Paper],
    ranked: dict,
    summaries: dict[str, str],
) -> list[dict]:
    """Assemble deep dive data for the email template."""
    paper_lookup = {p.arxiv_id: p for p in papers}
    deep_dives = []

    for rank_info in ranked["top_papers"]:
        if rank_info["tier"] != "deep_dive":
            continue

        arxiv_id = rank_info["arxiv_id"]
        if arxiv_id not in summaries:
            continue

        paper = paper_lookup.get(arxiv_id)
        if not paper:
            continue

        venue = None
        if paper.comments:
            venue_keywords = [
                "accepted",
                "published",
                "appear",
                "conference",
                "workshop",
            ]
            if any(kw in paper.comments.lower() for kw in venue_keywords):
                venue = paper.comments

        deep_dives.append(
            {
                "rank": rank_info["rank"],
                "title": paper.title,
                "authors": paper.authors,
                "arxiv_id": arxiv_id,
                "summary": summaries[arxiv_id],
                "venue": venue,
                "relevance_tags": rank_info.get("relevance_tags", []),
                "is_wildcard": rank_info.get("is_wildcard", False),
            }
        )

    return deep_dives


def _build_blurb_data(
    papers: list[Paper],
    ranked: dict,
    blurbs: list[dict],
) -> list[dict]:
    """Assemble blurb data for the email template."""
    paper_lookup = {p.arxiv_id: p for p in papers}
    blurb_lookup = {b["arxiv_id"]: b for b in blurbs}
    blurb_data = []

    for rank_info in ranked["top_papers"]:
        if rank_info["tier"] != "blurb":
            continue

        arxiv_id = rank_info["arxiv_id"]
        blurb = blurb_lookup.get(arxiv_id)
        paper = paper_lookup.get(arxiv_id)

        if not blurb or not paper:
            continue

        blurb_data.append(
            {
                "rank": rank_info["rank"],
                "title": paper.title,
                "authors": paper.authors,
                "arxiv_id": arxiv_id,
                "blurb": blurb["blurb"],
                "read_this_if": blurb.get("read_this_if", ""),
                "is_wildcard": rank_info.get("is_wildcard", False),
            }
        )

    return blurb_data


def run_pipeline():
    """Execute the full digest pipeline."""
    load_dotenv()
    start_time = time.time()
    dry_run = _is_dry_run()
    today = date.today().isoformat()

    log.info("Starting ArXiv AI Digest pipeline")
    log.info(f"Date: {today} | DRY_RUN: {dry_run}")
    log.info("=" * 60)

    # ── Stage: Data Collector ──
    log.info("── Stage: Data Collector ──")
    papers = fetch_papers()

    if not papers:
        log.info("No papers found — sending quiet day notice")
        html = compose_quiet_day_email(today)
        subject = f"ArXiv AI Digest — Quiet Day ({today})"
        if dry_run:
            log.info(f"DRY RUN — would send quiet day email: {subject}")
        else:
            send_email(subject=subject, html_body=html)
            log.info("Quiet day email sent")
        return

    log.info(f"Collected {len(papers)} papers")

    # ── Stage 1: Ranker ──
    log.info("── Stage 1: Ranker ──")
    ranked = rank_papers(papers)
    log.info(f"Ranked top {len(ranked['top_papers'])} papers")

    for p in ranked["top_papers"]:
        tier = "DD" if p["tier"] == "deep_dive" else "BL"
        wildcard = " [WILDCARD]" if p.get("is_wildcard") else ""
        log.info(f"  [{tier}] #{p['rank']}: {p['title'][:70]}{wildcard}")

    # ── Stage 2: Deep Analysis ──
    log.info("── Stage 2: Deep Analysis ──")
    summaries = analyze_top_papers(papers, ranked)
    log.info(f"Generated {len(summaries)} deep dive summaries")

    for arxiv_id, summary in summaries.items():
        log.info(f"  {arxiv_id}: {len(summary.split())} words")

    # ── Stage 2: Blurb Generation ──
    log.info("── Stage 2: Blurb Generation ──")
    blurbs = generate_blurbs(papers, ranked)
    log.info(f"Generated {len(blurbs)} blurbs")

    # ── Email Composition ──
    log.info("── Email Composition ──")
    deep_dive_data = _build_deep_summary_data(papers, ranked, summaries)
    blurb_data = _build_blurb_data(papers, ranked, blurbs)

    stats = {
        "total_papers": ranked.get("total_papers_evaluated", len(papers)),
        "date": today,
        "profile_name": "Andres — AI Research Digest",
    }

    html = compose_email(deep_dive_data, blurb_data, stats)
    subject = f"ArXiv AI Digest — {today}"

    # ── Email Delivery ──
    if dry_run:
        log.info("── DRY RUN — Email not sent ──")
        log.info(f"Subject: {subject}")
        log.info(f"Email size: {len(html):,} chars")

        preview_path = f".dev_cache/spot_checks/digest_{today}.html"
        from pathlib import Path

        Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preview_path).write_text(html)
        log.info(f"Preview saved to {preview_path}")
    else:
        log.info("── Email Delivery ──")
        result = send_email(subject=subject, html_body=html)
        log.info(f"Delivery: {result['sent']} sent, {result['failed']} failed")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    log.info(f"Pipeline complete in {minutes}m {seconds}s")


def main():
    try:
        run_pipeline()
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)

        try:
            today = date.today().isoformat()
            error_html = compose_error_email(
                stage="Pipeline",
                error=e,
                context={"date": today},
            )
            if not _is_dry_run():
                send_email(
                    subject=f"ArXiv Digest FAILED — {today}",
                    html_body=error_html,
                )
                log.info("Error notification email sent")
            else:
                log.info("DRY RUN — error email not sent")
        except Exception as email_err:
            log.error(f"Failed to send error email: {email_err}")

        raise


if __name__ == "__main__":
    main()
