from __future__ import annotations

import json
import time
from pathlib import Path

from src.collector import Paper
from src.llm import call_claude
from src.logger import setup_logger
from src.pdf_extractor import download_and_extract

log = setup_logger("analyzer")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
TEMPERATURE = 0.5
MAX_OUTPUT_TOKENS = 4096
PDF_DOWNLOAD_DELAY = 3.0

SYSTEM_PROMPT = """You are a world-class science communicator who makes cutting-edge AI research \
accessible to smart professionals who aren't necessarily researchers. Think of \
your audience as a tech-savvy executive or senior engineer who wants to \
understand what's happening in AI without reading full papers.

Your writing style is:
- Clear and direct, never condescending
- Uses analogies and examples to explain complex ideas
- Includes specific results and numbers when they tell a story
- Honest about limitations — you don't overhype
- Engaging — the reader should want to finish the summary

You write in a professional but warm tone. No jargon without explanation. \
No hand-waving. Every claim is grounded in what the paper actually says."""

USER_PROMPT_TEMPLATE = """## Reader Profile (for context on what matters to them)

{user_profile_json}

## Paper Metadata

Title: {title}
Authors: {authors}
arXiv ID: {arxiv_id}
Subjects: {subjects}
Comments: {comments}
Venue: {venue}

## Ranking Context

This paper was ranked #{rank} out of {total_papers} papers today.
Ranking justification: {justification}
Relevance tags: {relevance_tags}

## Full Paper Text

{full_paper_text}

## Instructions

Write a detailed summary of this paper following this EXACT structure.
Target length: 1,000-2,000 words total across all sections.

### 1. The "So What?" (1 paragraph)
Open with why this paper matters in plain language. What problem does it \
solve? Why should a busy professional care? Lead with impact, not \
technical details.

### 2. The Core Idea (2-3 paragraphs)
Explain the key insight or method in accessible terms. Use analogies \
where they genuinely help. Assume the reader is intelligent but not a \
specialist in this specific sub-field.

### 3. How It Works (2-3 paragraphs)
Walk through the approach clearly. Not a full technical deep dive, but \
enough that the reader understands the mechanism. Skip heavy math — \
explain the intuition behind the math. If there's a novel architecture \
or pipeline, describe it step by step.

### 4. Key Results (1-2 paragraphs)
What did they find? How does it compare to previous work? Include \
specific numbers, percentages, or benchmarks when they tell a \
compelling story. Don't list every result — highlight the ones that \
matter most.

### 5. Limitations & Open Questions (1 paragraph)
Be honest. What doesn't the paper address? What assumptions does it \
make? What would need to happen for this to be practically useful at \
scale? This section builds trust with the reader.

### 6. Real-World Applications & Opportunities (1-2 paragraphs)
Concrete examples of how this research could be applied in industry, \
products, or businesses. What opportunities does it create? Who should \
be paying attention — and what could they build with this? Connect it \
to the reader's interests where relevant.

FORMAT RULES:
- Use markdown headers (### ) for each section
- Write in flowing prose, not bullet points (except where a short list genuinely aids clarity)
- Do NOT include the paper title or authors in the summary body — those are in the email template
- Do NOT start with "This paper..." — lead with the problem or insight
- Aim for the high end of the word range (closer to 2,000 than 1,000) when the paper warrants it"""


def _load_user_profile() -> dict:
    profile_path = CONFIG_DIR / "user_profile.json"
    with open(profile_path) as f:
        return json.load(f)


def _extract_venue(comments: str | None) -> str:
    if not comments:
        return "Not specified"
    venue_keywords = [
        "accepted",
        "published",
        "appear",
        "to appear",
        "conference",
        "workshop",
    ]
    if any(kw in comments.lower() for kw in venue_keywords):
        return comments
    return "Not specified"


def build_analysis_prompt(
    paper: Paper,
    full_text: str,
    rank_info: dict,
    total_papers: int,
) -> tuple[str, str]:
    """Build the system and user prompts for deep paper analysis."""
    profile = _load_user_profile()

    authors_str = ", ".join(paper.authors[:15])
    if len(paper.authors) > 15:
        authors_str += f" (+ {len(paper.authors) - 15} more)"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_profile_json=json.dumps(profile, indent=2),
        title=paper.title,
        authors=authors_str,
        arxiv_id=paper.arxiv_id,
        subjects=", ".join(paper.subjects),
        comments=paper.comments or "None",
        venue=_extract_venue(paper.comments),
        rank=rank_info["rank"],
        total_papers=total_papers,
        justification=rank_info["justification"],
        relevance_tags=", ".join(rank_info.get("relevance_tags", [])),
        full_paper_text=full_text,
    )

    return SYSTEM_PROMPT, user_prompt


def analyze_paper(
    paper: Paper, full_text: str, rank_info: dict, total_papers: int = 169
) -> str:
    """Generate a deep analysis summary for a single paper.

    Returns the markdown summary text (1,000-2,000 words).
    """
    system_prompt, user_prompt = build_analysis_prompt(
        paper, full_text, rank_info, total_papers
    )

    log.info(
        f"Analyzing paper {rank_info['rank']}: {paper.title[:60]}... "
        f"(~{len(user_prompt) // 4:,} input tokens estimated)"
    )

    response = call_claude(
        system=system_prompt,
        user_prompt=user_prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    summary = response.content[0].text
    word_count = len(summary.split())

    log.info(
        f"Analysis complete: {word_count} words, "
        f"{response.usage.input_tokens:,} input / {response.usage.output_tokens:,} output tokens"
    )

    if word_count < 500:
        log.warning(
            f"Summary unusually short ({word_count} words) for {paper.arxiv_id}"
        )

    return summary


def analyze_top_papers(papers: list[Paper], ranked: dict) -> dict[str, str]:
    """Analyze all deep-dive papers (top 5). Returns {arxiv_id: summary_text}.

    If a paper fails, logs the error and continues with the rest.
    """
    deep_dive_papers = [p for p in ranked["top_papers"] if p["tier"] == "deep_dive"]
    total_papers = ranked.get("total_papers_evaluated", len(papers))
    paper_lookup = {p.arxiv_id: p for p in papers}

    summaries: dict[str, str] = {}

    for i, rank_info in enumerate(deep_dive_papers):
        arxiv_id = rank_info["arxiv_id"]
        paper = paper_lookup.get(arxiv_id)

        if not paper:
            log.error(f"Paper {arxiv_id} not found in paper list — skipping")
            continue

        log.info(
            f"[{i + 1}/{len(deep_dive_papers)}] Processing {arxiv_id}: {paper.title[:60]}..."
        )

        try:
            full_text = download_and_extract(arxiv_id)

            if i > 0:
                time.sleep(PDF_DOWNLOAD_DELAY)

            summary = analyze_paper(paper, full_text, rank_info, total_papers)
            summaries[arxiv_id] = summary

        except Exception as e:
            log.error(f"Failed to analyze {arxiv_id}: {e}")
            continue

    if len(summaries) < len(deep_dive_papers):
        log.warning(
            f"Only {len(summaries)}/{len(deep_dive_papers)} papers analyzed successfully"
        )

    return summaries
