import json

import pytest

from src.collector import Paper
from src.ranker import (
    _parse_ranking_response,
    _validate_ranking,
    build_ranking_prompt,
)


def _make_paper(**overrides) -> Paper:
    defaults = {
        "arxiv_id": "2602.00001",
        "title": "Test Paper on LLM Agents",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We introduce a new approach to LLM agent coordination.",
        "comments": "Accepted at NeurIPS 2026",
        "subjects": ["cs.AI", "cs.CL"],
        "pdf_url": "https://arxiv.org/pdf/2602.00001",
        "html_url": "https://arxiv.org/html/2602.00001",
        "published_date": "2026-02-19",
        "announce_type": "new",
    }
    defaults.update(overrides)
    return Paper(**defaults)


VALID_RANKING_RESPONSE = {
    "total_papers_evaluated": 100,
    "ranking_date": "2026-02-20",
    "top_papers": [
        {
            "rank": i,
            "arxiv_id": f"2602.{i:05d}",
            "title": f"Paper #{i}",
            "tier": "deep_dive" if i <= 5 else "blurb",
            "justification": f"This paper is relevant because reason {i}.",
            "relevance_tags": ["LLM agents"],
            "source_match": None,
            "is_wildcard": i == 10,
        }
        for i in range(1, 11)
    ],
}


class TestBuildRankingPrompt:
    def test_returns_system_and_user_prompts(self):
        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(3)]
        system, user = build_ranking_prompt(papers)
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0

    def test_user_prompt_contains_profile(self):
        papers = [_make_paper()]
        _, user = build_ranking_prompt(papers)
        assert "primary_interests" in user
        assert "LLM agents" in user

    def test_user_prompt_contains_all_papers(self):
        papers = [
            _make_paper(arxiv_id=f"2602.{i:05d}", title=f"Paper {i}") for i in range(5)
        ]
        _, user = build_ranking_prompt(papers)
        for p in papers:
            assert p.arxiv_id in user
            assert p.title in user

    def test_user_prompt_contains_paper_count(self):
        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(12)]
        _, user = build_ranking_prompt(papers)
        assert "12 total" in user

    def test_truncates_long_author_lists(self):
        many_authors = [f"Author {i}" for i in range(15)]
        papers = [_make_paper(authors=many_authors)]
        _, user = build_ranking_prompt(papers)
        assert "(+ 5 more)" in user


class TestParseRankingResponse:
    def test_parses_valid_json(self):
        raw = json.dumps(VALID_RANKING_RESPONSE)
        result = _parse_ranking_response(raw)
        assert result["total_papers_evaluated"] == 100

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps(VALID_RANKING_RESPONSE) + "\n```"
        result = _parse_ranking_response(raw)
        assert result["total_papers_evaluated"] == 100

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_ranking_response("not json at all")


class TestValidateRanking:
    def test_valid_response_passes(self):
        result = _validate_ranking(VALID_RANKING_RESPONSE.copy(), 100)
        assert len(result["top_papers"]) == 10

    def test_missing_top_papers_key(self):
        with pytest.raises(ValueError, match="missing 'top_papers'"):
            _validate_ranking({"ranking_date": "2026-02-20"}, 100)

    def test_too_few_papers(self):
        bad = {
            "top_papers": [
                {
                    "rank": 1,
                    "arxiv_id": "2602.00001",
                    "title": "Only One",
                    "tier": "deep_dive",
                    "justification": "Test.",
                }
            ]
        }
        with pytest.raises(ValueError, match="at least 5"):
            _validate_ranking(bad, 100)

    def test_missing_required_keys(self):
        bad = {
            "top_papers": [
                {"rank": i, "arxiv_id": f"2602.{i:05d}"} for i in range(1, 11)
            ]
        }
        with pytest.raises(ValueError, match="missing keys"):
            _validate_ranking(bad, 100)

    def test_enforces_correct_tiers(self):
        response = json.loads(json.dumps(VALID_RANKING_RESPONSE))
        # Mess up tiers
        for p in response["top_papers"]:
            p["tier"] = "blurb"
        result = _validate_ranking(response, 100)
        for p in result["top_papers"]:
            if p["rank"] <= 5:
                assert p["tier"] == "deep_dive"
            else:
                assert p["tier"] == "blurb"

    def test_defaults_missing_optional_fields(self):
        response = json.loads(json.dumps(VALID_RANKING_RESPONSE))
        for p in response["top_papers"]:
            p.pop("relevance_tags", None)
            p.pop("source_match", None)
            p.pop("is_wildcard", None)
        result = _validate_ranking(response, 100)
        for p in result["top_papers"]:
            assert "relevance_tags" in p
            assert "source_match" in p
            assert "is_wildcard" in p


class TestRankPapers:
    def test_calls_claude_and_returns_result(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.content = [
            mocker.MagicMock(text=json.dumps(VALID_RANKING_RESPONSE))
        ]
        mock_response.usage.input_tokens = 50000
        mock_response.usage.output_tokens = 1500

        mock_client = mocker.MagicMock()
        mock_client.messages.create.return_value = mock_response
        mocker.patch("src.ranker.anthropic.Anthropic", return_value=mock_client)

        from src.ranker import rank_papers

        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(20)]
        result = rank_papers(papers)

        assert len(result["top_papers"]) == 10
        mock_client.messages.create.assert_called_once()

    def test_retries_on_malformed_json(self, mocker):
        good_response = mocker.MagicMock()
        good_response.content = [
            mocker.MagicMock(text=json.dumps(VALID_RANKING_RESPONSE))
        ]
        good_response.usage.input_tokens = 50000
        good_response.usage.output_tokens = 1500

        bad_response = mocker.MagicMock()
        bad_response.content = [mocker.MagicMock(text="not valid json")]
        bad_response.usage.input_tokens = 50000
        bad_response.usage.output_tokens = 100

        mock_client = mocker.MagicMock()
        mock_client.messages.create.side_effect = [bad_response, good_response]
        mocker.patch("src.ranker.anthropic.Anthropic", return_value=mock_client)

        from src.ranker import rank_papers

        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(20)]
        result = rank_papers(papers)

        assert len(result["top_papers"]) == 10
        assert mock_client.messages.create.call_count == 2

    def test_raises_after_two_failures(self, mocker):
        bad_response = mocker.MagicMock()
        bad_response.content = [mocker.MagicMock(text="garbage")]
        bad_response.usage.input_tokens = 50000
        bad_response.usage.output_tokens = 100

        mock_client = mocker.MagicMock()
        mock_client.messages.create.return_value = bad_response
        mocker.patch("src.ranker.anthropic.Anthropic", return_value=mock_client)

        from src.ranker import rank_papers

        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(20)]
        with pytest.raises(RuntimeError, match="Failed to get valid ranking"):
            rank_papers(papers)

        assert mock_client.messages.create.call_count == 2
