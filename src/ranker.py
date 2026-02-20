from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.collector import Paper
from src.llm import call_claude
from src.logger import setup_logger

log = setup_logger("ranker")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = """You are an expert AI research curator. Your job is to read through a day's \
worth of arXiv papers and identify the ones most relevant, impactful, and \
interesting to a specific reader based on their profile.

You are thorough, fair, and intellectually curious. You don't just match \
keywords — you understand the significance of research contributions and \
can identify genuinely important work even when it doesn't perfectly match \
the reader's stated interests."""

USER_PROMPT_TEMPLATE = """## Reader Profile

{user_profile_json}

## Today's Papers ({paper_count} total)

{papers_block}

## Instructions

Evaluate ALL papers above against the reader's profile and return the top 10 \
most relevant papers, ranked from most to least important.

For each selected paper, provide:
1. The arxiv_id
2. A justification (2-3 sentences) explaining why this paper matters to this specific reader
3. Relevance tags (which profile interests it matches)
4. Whether this is a "wildcard" pick (innovative/revolutionary but outside the reader's stated interests)

RANKING CRITERIA (in priority order):
1. Topic relevance to primary interests (highest weight)
2. Novelty — new paradigm or method vs. incremental improvement
3. Practical applicability — can it be used in industry?
4. Source credibility — from a prioritized institution/lab? (boost, not gatekeeper)
5. Venue acceptance — accepted at a top conference? (noted in comments)
6. Breadth of impact — relevant across domains?
7. Wildcard potential — genuinely revolutionary even if outside stated interests?

IMPORTANT RULES:
- Primary interests get MUCH higher weight than secondary
- Papers matching "deprioritize" topics should be excluded unless they are truly exceptional
- Include at least 1 wildcard pick if anything qualifies
- If a paper is from a prioritized source, note it but don't rank it higher JUST for that — \
quality and relevance come first
- A great paper from an unknown lab should absolutely make the list

RETURN THIS EXACT JSON STRUCTURE:
{{
  "total_papers_evaluated": <number>,
  "ranking_date": "<YYYY-MM-DD>",
  "top_papers": [
    {{
      "rank": <1-10>,
      "arxiv_id": "<id>",
      "title": "<paper title>",
      "tier": "deep_dive" | "blurb",
      "justification": "<2-3 sentences>",
      "relevance_tags": ["<matching interest 1>", "<matching interest 2>"],
      "source_match": "<institution/venue note or null>",
      "is_wildcard": <true|false>
    }}
  ]
}}

Papers ranked 1-5 should have tier "deep_dive".
Papers ranked 6-10 should have tier "blurb".

Return ONLY valid JSON. No markdown, no commentary outside the JSON."""


def _load_user_profile() -> dict:
    profile_path = CONFIG_DIR / "user_profile.json"
    with open(profile_path) as f:
        return json.load(f)


def _format_paper_block(index: int, paper: Paper) -> str:
    authors_str = ", ".join(paper.authors[:10])
    if len(paper.authors) > 10:
        authors_str += f" (+ {len(paper.authors) - 10} more)"

    return (
        f"---\n"
        f"[{index}] arxiv_id: {paper.arxiv_id}\n"
        f"Title: {paper.title}\n"
        f"Authors: {authors_str}\n"
        f"Abstract: {paper.abstract}\n"
        f"Comments: {paper.comments or 'None'}\n"
        f"Subjects: {', '.join(paper.subjects)}\n"
        f"---"
    )


def build_ranking_prompt(papers: list[Paper]) -> tuple[str, str]:
    """Build the system and user prompts for the ranking LLM call.

    Returns (system_prompt, user_prompt).
    """
    profile = _load_user_profile()
    profile_json = json.dumps(profile, indent=2)

    papers_block = "\n\n".join(
        _format_paper_block(i + 1, p) for i, p in enumerate(papers)
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_profile_json=profile_json,
        paper_count=len(papers),
        papers_block=papers_block,
    )

    return SYSTEM_PROMPT, user_prompt


def _parse_ranking_response(text: str) -> dict:
    """Parse the LLM response as JSON, stripping any markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rindex("```")]
    cleaned = cleaned.strip()

    return json.loads(cleaned)


def _validate_ranking(result: dict, paper_count: int) -> dict:
    """Validate and normalize the ranking response structure."""
    if "top_papers" not in result:
        raise ValueError("Response missing 'top_papers' key")

    papers = result["top_papers"]
    if len(papers) < 5:
        raise ValueError(f"Expected at least 5 ranked papers, got {len(papers)}")

    for i, paper in enumerate(papers):
        required_keys = {"rank", "arxiv_id", "title", "tier", "justification"}
        missing = required_keys - set(paper.keys())
        if missing:
            raise ValueError(f"Paper #{i + 1} missing keys: {missing}")

        if paper["tier"] not in ("deep_dive", "blurb"):
            raise ValueError(f"Paper #{i + 1} invalid tier: {paper['tier']}")

        paper.setdefault("relevance_tags", [])
        paper.setdefault("source_match", None)
        paper.setdefault("is_wildcard", False)

    # Ensure correct tiers: top 5 = deep_dive, rest = blurb
    for paper in papers:
        if paper["rank"] <= 5:
            paper["tier"] = "deep_dive"
        else:
            paper["tier"] = "blurb"

    result.setdefault("total_papers_evaluated", paper_count)
    result.setdefault("ranking_date", date.today().isoformat())

    return result


def rank_papers(papers: list[Paper]) -> dict:
    """Rank papers using Claude and return structured results.

    Makes one LLM call, with one retry if the response is malformed JSON.
    """
    system_prompt, user_prompt = build_ranking_prompt(papers)

    log.info(
        f"Ranking {len(papers)} papers "
        f"(~{len(user_prompt) // 4:,} input tokens estimated)"
    )

    last_error = None

    for attempt in range(2):
        if attempt > 0:
            log.warning(f"Retry attempt {attempt + 1} due to: {last_error}")

        response = call_claude(
            system=system_prompt,
            user_prompt=user_prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        raw_text = response.content[0].text

        try:
            result = _parse_ranking_response(raw_text)
            validated = _validate_ranking(result, len(papers))
            log.info(
                f"Ranking complete: {len(validated['top_papers'])} papers selected"
            )
            return validated
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            log.warning(f"Failed to parse ranking response: {e}")

    raise RuntimeError(
        f"Failed to get valid ranking after 2 attempts. Last error: {last_error}"
    )
