from __future__ import annotations

import json
from pathlib import Path

from src.collector import Paper
from src.llm import call_claude
from src.logger import setup_logger

log = setup_logger("blurb_generator")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
TEMPERATURE = 0.4
MAX_OUTPUT_TOKENS = 2048

SYSTEM_PROMPT = """You are a concise AI research curator. You write sharp, informative blurbs \
that help busy professionals decide whether to read a paper. Every word earns \
its place."""

USER_PROMPT_TEMPLATE = """## Reader Profile

{user_profile_json}

## Papers to Summarize

{papers_block}

## Instructions

For EACH paper above, write a blurb of 100-150 words that:

1. States the core contribution in 1-2 crisp sentences
2. Explains why it's noteworthy (what's new or different)
3. Ends with a "Read this if:" tag pointing to who should care

RETURN THIS EXACT JSON STRUCTURE:
{{
  "blurbs": [
    {{
      "arxiv_id": "<id>",
      "rank": <number>,
      "blurb": "<100-150 word blurb text>",
      "read_this_if": "<one-line tag, e.g., 'you're building multi-agent systems'>"
    }}
  ]
}}

Return ONLY valid JSON. No markdown, no commentary outside the JSON."""


def _load_user_profile() -> dict:
    profile_path = CONFIG_DIR / "user_profile.json"
    with open(profile_path) as f:
        return json.load(f)


def _format_blurb_paper(paper: Paper, rank_info: dict) -> str:
    authors_str = ", ".join(paper.authors[:10])
    if len(paper.authors) > 10:
        authors_str += f" (+ {len(paper.authors) - 10} more)"

    return (
        f"---\n"
        f"[{rank_info['rank']}] arxiv_id: {paper.arxiv_id}\n"
        f"Title: {paper.title}\n"
        f"Authors: {authors_str}\n"
        f"Abstract: {paper.abstract}\n"
        f"Comments: {paper.comments or 'None'}\n"
        f"Ranking justification: {rank_info['justification']}\n"
        f"---"
    )


def _parse_blurb_response(text: str) -> list[dict]:
    """Parse the LLM response as JSON, stripping any markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rindex("```")]
    cleaned = cleaned.strip()

    result = json.loads(cleaned)
    return result["blurbs"]


def generate_blurbs(papers: list[Paper], ranked: dict) -> list[dict]:
    """Generate short blurbs for blurb-tier papers (typically #6-10).

    Makes a single LLM call for all blurbs. Returns list of blurb dicts.
    """
    blurb_papers = [p for p in ranked["top_papers"] if p["tier"] == "blurb"]
    paper_lookup = {p.arxiv_id: p for p in papers}

    papers_block_parts = []
    for rank_info in blurb_papers:
        paper = paper_lookup.get(rank_info["arxiv_id"])
        if not paper:
            log.warning(f"Paper {rank_info['arxiv_id']} not found — skipping blurb")
            continue
        papers_block_parts.append(_format_blurb_paper(paper, rank_info))

    if not papers_block_parts:
        log.error("No papers found for blurb generation")
        return []

    profile = _load_user_profile()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_profile_json=json.dumps(profile, indent=2),
        papers_block="\n\n".join(papers_block_parts),
    )

    log.info(f"Generating blurbs for {len(papers_block_parts)} papers")

    last_error = None
    for attempt in range(2):
        if attempt > 0:
            log.warning(f"Retry attempt {attempt + 1} due to: {last_error}")

        response = call_claude(
            system=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        try:
            blurbs = _parse_blurb_response(response.content[0].text)
            log.info(f"Generated {len(blurbs)} blurbs")
            return blurbs
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            log.warning(f"Failed to parse blurb response: {e}")

    raise RuntimeError(
        f"Failed to generate blurbs after 2 attempts. Last error: {last_error}"
    )
