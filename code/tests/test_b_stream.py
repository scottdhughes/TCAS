"""Tests for the B-Stream (Behavioral Battery)."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from tcas.streams.b_stream import BStream, BStreamItem, BStreamItemResult, BStreamResult


class TestBStreamItemResult:
    """Tests for BStreamItemResult dataclass properties."""

    def test_mean(self):
        r = BStreamItemResult(
            item_name="x", theory="GNW",
            scores=[0.6, 0.8, 1.0], responses=["a", "b", "c"],
        )
        assert r.mean == pytest.approx(0.8)

    def test_variance_single_score(self):
        r = BStreamItemResult(
            item_name="x", theory="GNW",
            scores=[0.5], responses=["a"],
        )
        assert r.variance == 0.0

    def test_variance_multiple(self):
        r = BStreamItemResult(
            item_name="x", theory="GNW",
            scores=[0.4, 0.6], responses=["a", "b"],
        )
        # ddof=1: var = ((0.1)^2 + (0.1)^2) / 1 = 0.02
        assert r.variance == pytest.approx(0.02)

    def test_robustness_score_math(self):
        r = BStreamItemResult(
            item_name="x", theory="GNW",
            scores=[0.8, 0.8, 0.8], responses=["a", "b", "c"],
        )
        # variance=0, so robustness = mean = 0.8
        assert r.robustness_score(lambda_val=0.7) == pytest.approx(0.8)

    def test_robustness_penalises_variance(self):
        r = BStreamItemResult(
            item_name="x", theory="GNW",
            scores=[0.4, 0.8], responses=["a", "b"],
        )
        # mean=0.6, var=0.08, std~=0.2828
        expected = 0.6 - 0.7 * np.sqrt(0.08)
        assert r.robustness_score(0.7) == pytest.approx(expected)

    def test_n_paraphrases(self):
        r = BStreamItemResult(
            item_name="x", theory="GNW",
            scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            responses=["a"] * 5,
        )
        assert r.n_paraphrases == 5


class TestBStream:
    """Tests for BStream class."""

    def test_default_items_has_minimum_paraphrases(self):
        items = BStream.default_items()
        assert len(items) >= 3
        for item in items:
            assert len(item.paraphrases) >= 5

    def test_add_item_rejects_too_few_paraphrases(self):
        bs = BStream(min_paraphrases=5)
        item = BStreamItem(
            name="bad", theory="GNW", description="d",
            paraphrases=["a", "b"],
        )
        with pytest.raises(ValueError, match="paraphrases"):
            bs.add_item(item)

    def test_run_with_mock_model_and_scorer(self):
        bs = BStream(lambda_val=0.5, min_paraphrases=2)
        item = BStreamItem(
            name="test_item", theory="HOT", description="d",
            paraphrases=["p1", "p2"],
        )
        bs.add_item(item)

        model_fn = MagicMock(return_value="response text")
        scorer_fn = MagicMock(return_value=0.7)

        result = bs.run(model_fn, scorer_fn)

        assert isinstance(result, BStreamResult)
        assert len(result.item_results) == 1
        assert result.item_results[0].item_name == "test_item"
        assert model_fn.call_count == 2
        assert scorer_fn.call_count == 2

    def test_run_clips_scores(self):
        bs = BStream(lambda_val=0.5, min_paraphrases=2)
        item = BStreamItem(
            name="clip", theory="GNW", description="d",
            paraphrases=["p1", "p2"],
        )
        bs.add_item(item)

        model_fn = MagicMock(return_value="r")
        scorer_fn = MagicMock(return_value=1.5)  # out of range

        result = bs.run(model_fn, scorer_fn)
        assert all(s <= 1.0 for s in result.item_results[0].scores)

    def test_run_uses_item_scorer_when_no_global(self):
        bs = BStream(lambda_val=0.5, min_paraphrases=2)
        item_scorer = MagicMock(return_value=0.9)
        item = BStreamItem(
            name="custom", theory="IIT", description="d",
            paraphrases=["p1", "p2"],
            scorer=item_scorer,
        )
        bs.add_item(item)

        model_fn = MagicMock(return_value="r")
        result = bs.run(model_fn)

        assert item_scorer.call_count == 2
        assert result.item_results[0].scores == [0.9, 0.9]


class TestBStreamResult:
    """Tests for BStreamResult aggregation."""

    def _make_result(self):
        return BStreamResult(
            item_results=[
                BStreamItemResult(
                    item_name="a", theory="GNW",
                    scores=[0.8, 0.8], responses=["r"] * 2,
                ),
                BStreamItemResult(
                    item_name="b", theory="HOT",
                    scores=[0.6, 0.6], responses=["r"] * 2,
                ),
            ],
            lambda_val=0.5,
        )

    def test_aggregate_robustness(self):
        r = self._make_result()
        # Both items have zero variance, so robustness = mean
        assert r.aggregate_robustness() == pytest.approx(0.7)

    def test_by_theory(self):
        r = self._make_result()
        gnw = r.by_theory("GNW")
        assert len(gnw) == 1
        assert gnw[0].item_name == "a"

    def test_theory_score_missing(self):
        r = self._make_result()
        assert r.theory_score("IIT") is None
