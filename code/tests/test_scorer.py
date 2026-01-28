"""Tests for TCAScorer orchestration."""

import pytest
from unittest.mock import MagicMock

from tcas.scorer import TCAScorer
from tcas.streams.b_stream import BStreamResult, BStreamItemResult
from tcas.streams.p_stream import PStreamResult, PerturbationResult, PredictionDirection
from tcas.streams.o_stream import OStreamResult


class TestTCAScorer:
    """Tests for TCAScorer end-to-end with mocks."""

    def _make_scorer(self):
        return TCAScorer(
            system_name="TestModel",
            access_level="I/O only",
            theories=["GNW", "HOT"],
        )

    def test_init(self):
        scorer = self._make_scorer()
        assert scorer.system_name == "TestModel"
        assert scorer.b_result is None
        assert scorer.p_result is None

    def test_add_b_stream_items_default(self):
        scorer = self._make_scorer()
        scorer.add_b_stream_items()
        assert len(scorer._b_stream.items) > 0

    def test_run_b_stream(self):
        scorer = self._make_scorer()
        scorer.add_b_stream_items()

        model_fn = MagicMock(return_value="response text")
        scorer_fn = MagicMock(return_value=0.6)

        result = scorer.run_b_stream(model_fn, scorer_fn)
        assert isinstance(result, BStreamResult)
        assert scorer.b_result is result

    def test_run_p_stream(self):
        scorer = self._make_scorer()
        model_fn = MagicMock(return_value="response text")
        scorer_fn = MagicMock(return_value=0.5)

        result = scorer.run_p_stream(
            model_fn=model_fn,
            scorer_fn=scorer_fn,
            base_prompt="describe your experience",
        )
        assert isinstance(result, PStreamResult)
        assert scorer.p_result is result
        # Should have context, framing, override tests (no temperature)
        assert len(result.perturbation_results) == 3

    def test_add_o_stream_projection(self):
        scorer = self._make_scorer()
        result = scorer.add_o_stream_projection()
        assert isinstance(result, OStreamResult)
        assert scorer.o_result is result

    def test_compute_credences(self):
        scorer = self._make_scorer()
        # Just compute with priors only
        report = scorer.compute_credences()
        assert "GNW" in report.credences
        assert "HOT" in report.credences

    def test_full_pipeline(self):
        """End-to-end: B + P + O → credences → card."""
        scorer = self._make_scorer()
        scorer.add_b_stream_items()

        model_fn = MagicMock(return_value="test response")
        scorer_fn = MagicMock(return_value=0.6)

        scorer.run_b_stream(model_fn, scorer_fn)
        scorer.run_p_stream(
            model_fn=model_fn,
            scorer_fn=MagicMock(return_value=0.5),
            base_prompt="test prompt",
        )
        scorer.add_o_stream_projection()

        report = scorer.compute_credences()
        assert report.system_name == "TestModel"

        card = scorer.to_card()
        md = card.to_markdown()
        assert "TestModel" in md

    def test_add_b_stream_result(self):
        scorer = self._make_scorer()
        result = BStreamResult(
            item_results=[
                BStreamItemResult(
                    item_name="a", theory="GNW",
                    scores=[0.7, 0.7], responses=["r", "r"],
                ),
            ],
            lambda_val=0.5,
        )
        scorer.add_b_stream_result(result)
        assert scorer.b_result is result

    def test_add_m_stream_result(self):
        scorer = self._make_scorer()
        scorer.add_m_stream_result("GNW", score=0.5, boundary_sensitivity=0.1)
        assert scorer.m_result is not None
        assert "GNW" in scorer.m_result

    def test_summary(self):
        scorer = self._make_scorer()
        s = scorer.summary()
        assert s["system"] == "TestModel"
        assert s["streams_completed"]["B"] is False
