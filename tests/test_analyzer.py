from src.analyzer import _extract_venue, build_analysis_prompt
from src.collector import Paper


def _make_paper(**overrides) -> Paper:
    defaults = {
        "arxiv_id": "2602.00001",
        "title": "Test Paper on LLM Agents",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We introduce a new approach.",
        "comments": "Accepted at NeurIPS 2026",
        "subjects": ["cs.AI", "cs.CL"],
        "pdf_url": "https://arxiv.org/pdf/2602.00001",
        "html_url": "https://arxiv.org/html/2602.00001",
        "published_date": "2026-02-19",
        "announce_type": "new",
    }
    defaults.update(overrides)
    return Paper(**defaults)


RANK_INFO = {
    "rank": 1,
    "arxiv_id": "2602.00001",
    "title": "Test Paper",
    "tier": "deep_dive",
    "justification": "Highly relevant to agents.",
    "relevance_tags": ["LLM agents"],
    "source_match": None,
    "is_wildcard": False,
}


class TestExtractVenue:
    def test_detects_accepted_paper(self):
        assert _extract_venue("Accepted at NeurIPS 2026") == "Accepted at NeurIPS 2026"

    def test_detects_published(self):
        assert _extract_venue("Published in ICML 2026") == "Published in ICML 2026"

    def test_no_venue(self):
        assert _extract_venue("12 pages, 5 figures") == "Not specified"

    def test_none_comments(self):
        assert _extract_venue(None) == "Not specified"


class TestBuildAnalysisPrompt:
    def test_returns_system_and_user(self):
        paper = _make_paper()
        system, user = build_analysis_prompt(paper, "full text here", RANK_INFO, 100)
        assert isinstance(system, str)
        assert len(system) > 0
        assert isinstance(user, str)
        assert len(user) > 0

    def test_contains_paper_metadata(self):
        paper = _make_paper(title="Unique Title XYZ")
        _, user = build_analysis_prompt(paper, "full text", RANK_INFO, 100)
        assert "Unique Title XYZ" in user
        assert "Alice Smith" in user
        assert "2602.00001" in user

    def test_contains_full_text(self):
        paper = _make_paper()
        _, user = build_analysis_prompt(paper, "THE FULL PAPER CONTENT", RANK_INFO, 100)
        assert "THE FULL PAPER CONTENT" in user

    def test_contains_ranking_context(self):
        paper = _make_paper()
        _, user = build_analysis_prompt(paper, "text", RANK_INFO, 150)
        assert "#1 out of 150" in user
        assert "Highly relevant to agents" in user

    def test_contains_section_headers(self):
        paper = _make_paper()
        _, user = build_analysis_prompt(paper, "text", RANK_INFO, 100)
        assert 'The "So What?"' in user
        assert "Core Idea" in user
        assert "How It Works" in user
        assert "Key Results" in user
        assert "Limitations" in user
        assert "Real-World Applications" in user

    def test_truncates_long_author_lists(self):
        many_authors = [f"Author {i}" for i in range(20)]
        paper = _make_paper(authors=many_authors)
        _, user = build_analysis_prompt(paper, "text", RANK_INFO, 100)
        assert "(+ 5 more)" in user


def _mock_llm_response(mocker, text):
    resp = mocker.MagicMock()
    resp.content = [mocker.MagicMock(text=text)]
    resp.usage.input_tokens = 5000
    resp.usage.output_tokens = 1500
    return resp


class TestAnalyzePaper:
    def test_returns_summary(self, mocker):
        mock_response = _mock_llm_response(
            mocker, "### The So What\n\nSummary text. " * 50
        )
        mocker.patch("src.analyzer.call_claude", return_value=mock_response)

        from src.analyzer import analyze_paper

        paper = _make_paper()
        result = analyze_paper(paper, "full text", RANK_INFO)
        assert "Summary text" in result
        assert len(result.split()) > 100


class TestAnalyzeTopPapers:
    def test_analyzes_all_deep_dive_papers(self, mocker):
        mocker.patch(
            "src.analyzer.download_and_extract",
            return_value="Extracted paper text. " * 100,
        )
        mock_response = _mock_llm_response(mocker, "Deep analysis summary. " * 80)
        mocker.patch("src.analyzer.call_claude", return_value=mock_response)

        from src.analyzer import analyze_top_papers

        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(1, 11)]
        ranked = {
            "total_papers_evaluated": 100,
            "top_papers": [
                {
                    "rank": i,
                    "arxiv_id": f"2602.{i:05d}",
                    "title": f"Paper {i}",
                    "tier": "deep_dive" if i <= 5 else "blurb",
                    "justification": "Relevant.",
                    "relevance_tags": ["agents"],
                }
                for i in range(1, 11)
            ],
        }

        summaries = analyze_top_papers(papers, ranked)
        assert len(summaries) == 5

    def test_continues_on_individual_failure(self, mocker):
        call_count = {"n": 0}

        def mock_extract(arxiv_id):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("PDF download failed")
            return "Paper text. " * 100

        mocker.patch("src.analyzer.download_and_extract", side_effect=mock_extract)
        mock_response = _mock_llm_response(mocker, "Summary. " * 80)
        mocker.patch("src.analyzer.call_claude", return_value=mock_response)

        from src.analyzer import analyze_top_papers

        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(1, 11)]
        ranked = {
            "total_papers_evaluated": 100,
            "top_papers": [
                {
                    "rank": i,
                    "arxiv_id": f"2602.{i:05d}",
                    "title": f"Paper {i}",
                    "tier": "deep_dive" if i <= 5 else "blurb",
                    "justification": "Relevant.",
                    "relevance_tags": ["agents"],
                }
                for i in range(1, 11)
            ],
        }

        summaries = analyze_top_papers(papers, ranked)
        assert len(summaries) == 4
