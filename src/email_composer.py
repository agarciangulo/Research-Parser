from __future__ import annotations

import re
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader

from src.logger import setup_logger

log = setup_logger("email_composer")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )


def _markdown_to_html(text: str) -> str:
    """Convert markdown summary to HTML for email rendering."""
    html = markdown.markdown(text, extensions=["extra"])
    # Strip wrapping <p> tags from headers that markdown sometimes adds
    html = re.sub(r"<p>(<h[1-6]>)", r"\1", html)
    html = re.sub(r"(</h[1-6]>)</p>", r"\1", html)
    return html


def compose_email(
    deep_summaries: list[dict],
    blurbs: list[dict],
    stats: dict,
) -> str:
    """Render the digest email HTML from summaries and blurbs.

    deep_summaries: list of dicts with keys:
        rank, title, authors, arxiv_id, summary (markdown), venue, relevance_tags, is_wildcard
    blurbs: list of dicts with keys:
        rank, title, authors, arxiv_id, blurb, read_this_if
    stats: dict with keys: total_papers, date, profile_name
    """
    for paper in deep_summaries:
        paper["summary"] = _markdown_to_html(paper["summary"])

    env = _get_jinja_env()
    template = env.get_template("digest_email.html")
    html = template.render(
        deep_summaries=deep_summaries,
        blurbs=blurbs,
        stats=stats,
    )

    log.info(f"Email composed: {len(html):,} chars")
    return html


def compose_quiet_day_email(date: str) -> str:
    """Render the quiet day email (no papers found)."""
    env = _get_jinja_env()
    template = env.get_template("quiet_day_email.html")
    html = template.render(date=date)
    log.info("Quiet day email composed")
    return html


def compose_error_email(
    stage: str,
    error: Exception,
    context: dict | None = None,
) -> str:
    """Compose a simple error notification email (no template needed)."""
    import traceback

    tb = traceback.format_exception(type(error), error, error.__traceback__)
    tb_text = "".join(tb[-5:])  # Last 5 frames

    context_html = ""
    if context:
        context_items = "".join(
            f"<li><strong>{k}:</strong> {v}</li>" for k, v in context.items()
        )
        context_html = f"<ul>{context_items}</ul>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, sans-serif; padding: 24px; color: #1a1a1a; }}
  .error-box {{ background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 20px; margin: 16px 0; }}
  .error-box h3 {{ color: #991b1b; margin: 0 0 8px; }}
  pre {{ background: #f8f8f8; padding: 16px; border-radius: 4px; overflow-x: auto; font-size: 13px; }}
</style></head><body>
<h2>Pipeline Error — ArXiv AI Digest</h2>
<div class="error-box">
  <h3>Failed at: {stage}</h3>
  <p><strong>Error:</strong> {type(error).__name__}: {error}</p>
</div>
{f"<h3>Context</h3>{context_html}" if context_html else ""}
<h3>Traceback (last 5 frames)</h3>
<pre>{tb_text}</pre>
<p style="color: #666; font-size: 13px;">This is an automated error notification from the ArXiv AI Digest pipeline.</p>
</body></html>"""

    log.info(f"Error email composed for stage: {stage}")
    return html
