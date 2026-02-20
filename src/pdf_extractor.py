from __future__ import annotations

import re
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

from src.logger import setup_logger

log = setup_logger("pdf_extractor")

PDF_URL_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}"
HTML_URL_TEMPLATE = "https://arxiv.org/html/{arxiv_id}"
DOWNLOAD_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT = 60
MAX_TOKEN_ESTIMATE = 80_000
CHARS_PER_TOKEN = 4


def download_pdf(arxiv_id: str, output_dir: str | None = None) -> Path:
    """Download a PDF from arXiv and return the local file path."""
    url = PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id)
    log.info(f"Downloading PDF: {url}")

    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    if "application/pdf" not in response.headers.get("content-type", ""):
        raise ValueError(f"Expected PDF but got {response.headers.get('content-type')}")

    dir_path = output_dir or tempfile.mkdtemp(prefix="arxiv_")
    file_path = Path(dir_path) / f"{arxiv_id.replace('/', '_')}.pdf"
    file_path.write_bytes(response.content)

    log.info(f"Downloaded {len(response.content):,} bytes to {file_path}")
    return file_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    text = "\n".join(pages)
    return _clean_text(text)


def extract_from_html(arxiv_id: str) -> str | None:
    """Fallback: extract paper text from arXiv's HTML version.

    Returns None if no HTML version is available.
    """
    url = HTML_URL_TEMPLATE.format(arxiv_id=arxiv_id)
    log.info(f"Trying HTML fallback: {url}")

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 404:
            log.info("No HTML version available")
            return None
        response.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"HTML fallback failed: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # arXiv HTML papers use <article> as the main content container
    article = soup.find("article")
    if not article:
        # Some papers use a different structure
        article = soup.find("div", class_="ltx_page_content")

    if not article:
        log.warning("Could not find article content in HTML")
        return None

    text = article.get_text(separator="\n", strip=True)
    return _clean_text(text)


def _clean_text(text: str) -> str:
    """Clean up extracted text by removing artifacts."""
    # Collapse multiple blank lines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers that appear on their own line (common in PDFs)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    # Remove repeated headers/footers (lines that appear many times identically)
    lines = text.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Lines appearing 4+ times are likely headers/footers
    repeated = {
        line for line, count in line_counts.items() if count >= 4 and len(line) < 100
    }
    if repeated:
        lines = [line for line in lines if line.strip() not in repeated]
        log.debug(f"Removed {len(repeated)} repeated header/footer patterns")

    text = "\n".join(lines).strip()
    return text


def truncate_if_needed(text: str) -> str:
    """Truncate text if it exceeds the token budget.

    Preserves the beginning of the paper (abstract, intro, methodology)
    and truncates from the end.
    """
    estimated_tokens = len(text) // CHARS_PER_TOKEN
    if estimated_tokens <= MAX_TOKEN_ESTIMATE:
        return text

    max_chars = MAX_TOKEN_ESTIMATE * CHARS_PER_TOKEN
    log.warning(
        f"Text exceeds token budget ({estimated_tokens:,} tokens estimated). "
        f"Truncating to ~{MAX_TOKEN_ESTIMATE:,} tokens."
    )
    truncated = text[:max_chars]

    # Try to truncate at a paragraph boundary
    last_double_newline = truncated.rfind("\n\n")
    if last_double_newline > max_chars * 0.8:
        truncated = truncated[:last_double_newline]

    truncated += "\n\n[... remainder truncated due to length ...]"
    return truncated


def download_and_extract(arxiv_id: str) -> str:
    """Download a paper and extract its text. Falls back to HTML if PDF fails.

    This is the main entry point for the extraction pipeline.
    """
    # Try PDF first
    try:
        pdf_path = download_pdf(arxiv_id)
        text = extract_text_from_pdf(pdf_path)
        pdf_path.unlink(missing_ok=True)

        word_count = len(text.split())
        if word_count < 200:
            log.warning(
                f"PDF extraction yielded only {word_count} words — trying HTML fallback"
            )
            raise ValueError("PDF extraction produced too little text")

        log.info(f"Extracted {word_count:,} words from PDF")
        return truncate_if_needed(text)

    except Exception as e:
        log.warning(f"PDF extraction failed for {arxiv_id}: {e}")

    # Fall back to HTML
    html_text = extract_from_html(arxiv_id)
    if html_text:
        word_count = len(html_text.split())
        log.info(f"Extracted {word_count:,} words from HTML fallback")
        return truncate_if_needed(html_text)

    raise RuntimeError(f"Failed to extract text for {arxiv_id} via both PDF and HTML")
