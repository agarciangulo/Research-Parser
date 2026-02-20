from datetime import date

import pytest

from src.collector import Paper, fetch_papers, get_previous_business_day


class TestGetPreviousBusinessDay:
    def test_tuesday_returns_monday(self):
        assert get_previous_business_day(date(2026, 2, 24)) == date(2026, 2, 23)

    def test_monday_returns_friday(self):
        assert get_previous_business_day(date(2026, 2, 23)) == date(2026, 2, 20)

    def test_saturday_returns_friday(self):
        assert get_previous_business_day(date(2026, 2, 21)) == date(2026, 2, 20)

    def test_sunday_returns_friday(self):
        assert get_previous_business_day(date(2026, 2, 22)) == date(2026, 2, 20)

    def test_thursday_returns_wednesday(self):
        assert get_previous_business_day(date(2026, 2, 19)) == date(2026, 2, 18)

    def test_default_uses_today(self):
        result = get_previous_business_day()
        assert isinstance(result, date)
        assert result < date.today()


class TestPaper:
    def test_construction(self):
        paper = Paper(
            arxiv_id="2602.12345",
            title="Test Paper",
            authors=["Alice", "Bob"],
            abstract="This is a test abstract.",
            comments="Accepted at ICLR 2026",
            subjects=["cs.AI", "cs.CL"],
            pdf_url="https://arxiv.org/pdf/2602.12345",
            html_url="https://arxiv.org/html/2602.12345",
            published_date="2026-02-19",
            announce_type="new",
        )
        assert paper.arxiv_id == "2602.12345"
        assert len(paper.authors) == 2
        assert paper.comments == "Accepted at ICLR 2026"

    def test_to_dict_roundtrip(self):
        paper = Paper(
            arxiv_id="2602.12345",
            title="Test Paper",
            authors=["Alice"],
            abstract="Abstract.",
            comments=None,
            subjects=["cs.AI"],
            pdf_url="https://arxiv.org/pdf/2602.12345",
            html_url="https://arxiv.org/html/2602.12345",
            published_date="2026-02-19",
        )
        d = paper.to_dict()
        restored = Paper.from_dict(d)
        assert restored.arxiv_id == paper.arxiv_id
        assert restored.title == paper.title
        assert restored.comments is None

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "arxiv_id": "2602.12345",
            "title": "Test",
            "authors": [],
            "abstract": "",
            "comments": None,
            "subjects": [],
            "pdf_url": "",
            "html_url": "",
            "published_date": "2026-02-19",
            "announce_type": "new",
            "extra_key": "should be ignored",
        }
        paper = Paper.from_dict(d)
        assert paper.arxiv_id == "2602.12345"


class TestFetchPapers:
    def test_returns_empty_list_when_rss_empty(self, mocker):
        mock_feed = mocker.MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = []
        mocker.patch("src.collector.feedparser.parse", return_value=mock_feed)

        papers = fetch_papers()
        assert papers == []

    def test_filters_replace_and_replace_cross(self, mocker):
        entries = [
            {"link": "https://arxiv.org/abs/2602.001", "arxiv_announce_type": "new"},
            {"link": "https://arxiv.org/abs/2602.002", "arxiv_announce_type": "cross"},
            {
                "link": "https://arxiv.org/abs/2602.003",
                "arxiv_announce_type": "replace",
            },
            {
                "link": "https://arxiv.org/abs/2602.004",
                "arxiv_announce_type": "replace-cross",
            },
        ]
        mock_feed = mocker.MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = entries
        mocker.patch("src.collector.feedparser.parse", return_value=mock_feed)

        mock_result = mocker.MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2602.001v1"
        mock_result.title = "Paper 1"
        mock_result.authors = []
        mock_result.summary = "Abstract 1"
        mock_result.comment = None
        mock_result.categories = ["cs.AI"]
        mock_result.published.date.return_value = date(2026, 2, 19)

        mock_result2 = mocker.MagicMock()
        mock_result2.entry_id = "http://arxiv.org/abs/2602.002v1"
        mock_result2.title = "Paper 2"
        mock_result2.authors = []
        mock_result2.summary = "Abstract 2"
        mock_result2.comment = "ICLR 2026"
        mock_result2.categories = ["cs.CL", "cs.AI"]
        mock_result2.published.date.return_value = date(2026, 2, 19)

        mock_client = mocker.MagicMock()
        mock_client.results.return_value = iter([mock_result, mock_result2])
        mocker.patch("src.collector.arxiv.Client", return_value=mock_client)

        papers = fetch_papers()
        assert len(papers) == 2
        assert papers[0].arxiv_id == "2602.001"
        assert papers[1].arxiv_id == "2602.002"
        assert papers[1].comments == "ICLR 2026"

    def test_deduplicates_by_arxiv_id(self, mocker):
        entries = [
            {"link": "https://arxiv.org/abs/2602.001", "arxiv_announce_type": "new"},
            {"link": "https://arxiv.org/abs/2602.001", "arxiv_announce_type": "cross"},
        ]
        mock_feed = mocker.MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = entries
        mocker.patch("src.collector.feedparser.parse", return_value=mock_feed)

        mock_result = mocker.MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2602.001v1"
        mock_result.title = "Paper 1"
        mock_result.authors = []
        mock_result.summary = "Abstract"
        mock_result.comment = None
        mock_result.categories = ["cs.AI"]
        mock_result.published.date.return_value = date(2026, 2, 19)

        mock_client = mocker.MagicMock()
        mock_client.results.return_value = iter([mock_result])
        mocker.patch("src.collector.arxiv.Client", return_value=mock_client)

        papers = fetch_papers()
        assert len(papers) == 1

    def test_raises_on_rss_error(self, mocker):
        mock_feed = mocker.MagicMock()
        mock_feed.bozo = True
        mock_feed.entries = []
        mock_feed.bozo_exception = Exception("Network error")
        mocker.patch("src.collector.feedparser.parse", return_value=mock_feed)

        with pytest.raises(RuntimeError, match="RSS feed parse error"):
            fetch_papers()
