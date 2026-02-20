import pytest

from src.email_composer import (
    _markdown_to_html,
    compose_email,
    compose_error_email,
    compose_quiet_day_email,
)


MOCK_DEEP_SUMMARIES = [
    {
        "rank": i,
        "title": f"Test Paper Title #{i}",
        "authors": ["Alice Smith", "Bob Jones"],
        "arxiv_id": f"2602.{17660 + i}",
        "summary": f"### 1. The So What\n\nSummary for paper {i}.\n\n### 2. Core Idea\n\nDetails here.",
        "venue": "ICLR 2026" if i == 1 else None,
        "relevance_tags": ["LLM agents", "AI reasoning"],
        "is_wildcard": i == 4,
    }
    for i in range(1, 6)
]

MOCK_BLURBS = [
    {
        "rank": i,
        "title": f"Blurb Paper #{i}",
        "authors": ["Author X"],
        "arxiv_id": f"2602.{17670 + i}",
        "blurb": f"This paper proposes approach {i}. It is noteworthy.",
        "read_this_if": "you care about agents",
    }
    for i in range(6, 11)
]

MOCK_STATS = {
    "total_papers": 187,
    "date": "2026-02-20",
    "profile_name": "Andres - AI Research Digest",
}


class TestMarkdownToHtml:
    def test_converts_headers(self):
        result = _markdown_to_html("### Section Title\n\nParagraph text.")
        assert "<h3>" in result
        assert "Section Title" in result

    def test_converts_paragraphs(self):
        result = _markdown_to_html("First paragraph.\n\nSecond paragraph.")
        assert "<p>" in result

    def test_converts_bold(self):
        result = _markdown_to_html("This is **bold** text.")
        assert "<strong>bold</strong>" in result


class TestComposeEmail:
    def test_renders_without_error(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        assert len(html) > 1000

    def test_contains_deep_dive_titles(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        for paper in MOCK_DEEP_SUMMARIES:
            assert paper["title"] in html

    def test_contains_blurb_titles(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        for blurb in MOCK_BLURBS:
            assert blurb["title"] in html

    def test_contains_arxiv_links(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        assert "arxiv.org/abs/2602.17661" in html
        assert "arxiv.org/pdf/2602.17661" in html

    def test_contains_venue_badge(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        assert "ICLR 2026" in html

    def test_contains_wildcard_marker(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        assert "WILDCARD" in html

    def test_contains_stats(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        assert "187 papers reviewed" in html
        assert "2026-02-20" in html

    def test_contains_read_this_if(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        assert "you care about agents" in html

    def test_empty_blurbs(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), [], MOCK_STATS)
        assert len(html) > 500
        assert "Deep Dives" in html

    def test_summary_markdown_converted(self):
        html = compose_email(MOCK_DEEP_SUMMARIES.copy(), MOCK_BLURBS.copy(), MOCK_STATS)
        assert "<h3>" in html


class TestComposeQuietDayEmail:
    def test_renders_without_error(self):
        html = compose_quiet_day_email("2026-02-20")
        assert "Quiet Day" in html
        assert "2026-02-20" in html

    def test_contains_explanation(self):
        html = compose_quiet_day_email("2026-02-20")
        assert "No new cs.AI papers" in html


class TestComposeErrorEmail:
    def test_renders_without_error(self):
        try:
            raise ValueError("Test error message")
        except Exception as e:
            html = compose_error_email("Stage 1: Ranker", e, {"papers": 100})
        assert "Stage 1: Ranker" in html
        assert "Test error message" in html

    def test_includes_context(self):
        try:
            raise RuntimeError("fail")
        except Exception as e:
            html = compose_error_email("Collector", e, {"date": "2026-02-20"})
        assert "2026-02-20" in html

    def test_works_without_context(self):
        try:
            raise RuntimeError("fail")
        except Exception as e:
            html = compose_error_email("Collector", e)
        assert "Collector" in html


class TestSendEmail:
    def test_skips_when_no_recipients(self, mocker):
        from src.email_sender import send_email

        result = send_email(
            subject="Test",
            html_body="<p>test</p>",
            recipients=[],
        )
        assert result["sent"] == 0
        assert result["failed"] == 0

    def test_raises_without_credentials(self, mocker):
        mocker.patch.dict("os.environ", {"GMAIL_ADDRESS": "", "GMAIL_APP_PASSWORD": ""})

        from src.email_sender import send_email

        with pytest.raises(RuntimeError, match="must be set"):
            send_email(
                subject="Test",
                html_body="<p>test</p>",
                recipients=[{"email": "test@example.com", "name": "Test"}],
            )
