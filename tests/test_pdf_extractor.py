from pathlib import Path

import pytest

from src.pdf_extractor import (
    _clean_text,
    download_and_extract,
    truncate_if_needed,
)


class TestCleanText:
    def test_collapses_multiple_blank_lines(self):
        text = "line 1\n\n\n\n\nline 2"
        assert _clean_text(text) == "line 1\n\nline 2"

    def test_removes_standalone_page_numbers(self):
        text = "some content\n  42  \nmore content"
        result = _clean_text(text)
        assert "42" not in result
        assert "some content" in result
        assert "more content" in result

    def test_removes_repeated_headers(self):
        header = "Conference on AI 2026"
        lines = []
        for i in range(5):
            lines.append(header)
            lines.append(f"Content block {i} with actual paper text here.")
        text = "\n".join(lines)
        result = _clean_text(text)
        assert header not in result
        assert "Content block 0" in result

    def test_preserves_normal_text(self):
        text = "This is a normal paragraph.\n\nThis is another paragraph."
        assert _clean_text(text) == text

    def test_strips_outer_whitespace(self):
        text = "  \n\n content here \n\n  "
        result = _clean_text(text)
        assert result == "content here"


class TestTruncateIfNeeded:
    def test_short_text_unchanged(self):
        text = "word " * 1000
        result = truncate_if_needed(text)
        assert len(result.split()) == len(text.split())

    def test_long_text_truncated(self):
        text = "word " * 200_000
        result = truncate_if_needed(text)
        assert len(result.split()) < len(text.split())
        assert "[... remainder truncated due to length ...]" in result

    def test_truncated_text_within_token_budget(self):
        text = "word " * 200_000
        result = truncate_if_needed(text)
        estimated_tokens = len(result) // 4
        assert estimated_tokens <= 85_000  # some buffer for the truncation message

    def test_exactly_at_limit_not_truncated(self):
        chars_at_limit = 80_000 * 4
        text = "a" * chars_at_limit
        result = truncate_if_needed(text)
        assert "[... remainder truncated" not in result


class TestDownloadAndExtract:
    def test_falls_back_to_html_on_pdf_failure(self, mocker):
        mocker.patch(
            "src.pdf_extractor.download_pdf",
            side_effect=RuntimeError("PDF download failed"),
        )
        mocker.patch(
            "src.pdf_extractor.extract_from_html",
            return_value="This is the HTML extracted text with enough words. " * 20,
        )

        result = download_and_extract("2602.99999")
        assert "HTML extracted text" in result

    def test_falls_back_when_pdf_too_short(self, mocker):
        mock_path = mocker.MagicMock(spec=Path)
        mocker.patch("src.pdf_extractor.download_pdf", return_value=mock_path)
        mocker.patch(
            "src.pdf_extractor.extract_text_from_pdf", return_value="Too short."
        )
        mocker.patch(
            "src.pdf_extractor.extract_from_html",
            return_value="Good HTML content. " * 100,
        )

        result = download_and_extract("2602.99999")
        assert "Good HTML content" in result

    def test_raises_when_both_fail(self, mocker):
        mocker.patch(
            "src.pdf_extractor.download_pdf",
            side_effect=RuntimeError("PDF failed"),
        )
        mocker.patch("src.pdf_extractor.extract_from_html", return_value=None)

        with pytest.raises(RuntimeError, match="Failed to extract text"):
            download_and_extract("2602.99999")

    def test_uses_pdf_when_extraction_succeeds(self, mocker):
        mock_path = mocker.MagicMock(spec=Path)
        mocker.patch("src.pdf_extractor.download_pdf", return_value=mock_path)
        mocker.patch(
            "src.pdf_extractor.extract_text_from_pdf",
            return_value="Good PDF content with many words. " * 100,
        )

        result = download_and_extract("2602.99999")
        assert "Good PDF content" in result
        # HTML fallback should NOT have been called
