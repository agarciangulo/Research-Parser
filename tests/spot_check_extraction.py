"""Spot-check script: downloads a paper and writes the extracted text to a markdown file."""

from pathlib import Path

from src.pdf_extractor import download_and_extract

ARXIV_ID = "2602.16715"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / ".dev_cache" / "spot_checks"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {ARXIV_ID}...")
    text = download_and_extract(ARXIV_ID)

    output_path = OUTPUT_DIR / f"{ARXIV_ID.replace('/', '_')}.md"
    output_path.write_text(
        f"# Spot Check: {ARXIV_ID}\n\n"
        f"- **PDF**: https://arxiv.org/pdf/{ARXIV_ID}\n"
        f"- **Abstract**: https://arxiv.org/abs/{ARXIV_ID}\n"
        f"- **Word count**: {len(text.split()):,}\n"
        f"- **Char count**: {len(text):,}\n\n"
        f"---\n\n"
        f"{text}\n"
    )

    print(f"Written to {output_path}")
    print(f"  {len(text.split()):,} words / {len(text):,} chars")


if __name__ == "__main__":
    main()
