import json

import pytest

from src.blurb_generator import _parse_blurb_response, generate_blurbs
from src.collector import Paper


def _make_paper(**overrides) -> Paper:
    defaults = {
        "arxiv_id": "2602.00001",
        "title": "Test Paper",
        "authors": ["Alice Smith"],
        "abstract": "Abstract text.",
        "comments": None,
        "subjects": ["cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/2602.00001",
        "html_url": "https://arxiv.org/html/2602.00001",
        "published_date": "2026-02-19",
        "announce_type": "new",
    }
    defaults.update(overrides)
    return Paper(**defaults)


VALID_BLURB_RESPONSE = {
    "blurbs": [
        {
            "arxiv_id": f"2602.{i:05d}",
            "rank": i,
            "blurb": f"This paper introduces method {i}. " * 10,
            "read_this_if": f"you care about topic {i}",
        }
        for i in range(6, 11)
    ]
}


class TestParseBlurbResponse:
    def test_parses_valid_json(self):
        raw = json.dumps(VALID_BLURB_RESPONSE)
        blurbs = _parse_blurb_response(raw)
        assert len(blurbs) == 5

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps(VALID_BLURB_RESPONSE) + "\n```"
        blurbs = _parse_blurb_response(raw)
        assert len(blurbs) == 5

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_blurb_response("not json")


class TestGenerateBlurbs:
    def test_generates_blurbs_for_blurb_tier(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.content = [
            mocker.MagicMock(text=json.dumps(VALID_BLURB_RESPONSE))
        ]
        mocker.patch("src.blurb_generator.call_claude", return_value=mock_response)

        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(1, 11)]
        ranked = {
            "top_papers": [
                {
                    "rank": i,
                    "arxiv_id": f"2602.{i:05d}",
                    "title": f"Paper {i}",
                    "tier": "deep_dive" if i <= 5 else "blurb",
                    "justification": "Relevant.",
                }
                for i in range(1, 11)
            ]
        }

        blurbs = generate_blurbs(papers, ranked)
        assert len(blurbs) == 5

    def test_retries_on_malformed_json(self, mocker):
        good_response = mocker.MagicMock()
        good_response.content = [
            mocker.MagicMock(text=json.dumps(VALID_BLURB_RESPONSE))
        ]
        bad_response = mocker.MagicMock()
        bad_response.content = [mocker.MagicMock(text="garbage")]

        mock_call = mocker.patch(
            "src.blurb_generator.call_claude",
            side_effect=[bad_response, good_response],
        )

        papers = [_make_paper(arxiv_id=f"2602.{i:05d}") for i in range(1, 11)]
        ranked = {
            "top_papers": [
                {
                    "rank": i,
                    "arxiv_id": f"2602.{i:05d}",
                    "tier": "blurb" if i > 5 else "deep_dive",
                    "justification": "Relevant.",
                }
                for i in range(1, 11)
            ]
        }

        blurbs = generate_blurbs(papers, ranked)
        assert len(blurbs) == 5
        assert mock_call.call_count == 2

    def test_returns_empty_when_no_blurb_papers(self, mocker):
        papers = [_make_paper()]
        ranked = {
            "top_papers": [
                {
                    "rank": 1,
                    "arxiv_id": "2602.99999",
                    "tier": "blurb",
                    "justification": "Test.",
                }
            ]
        }
        blurbs = generate_blurbs(papers, ranked)
        assert blurbs == []
