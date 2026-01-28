"""Tests for the P-Stream (Perturbational Validation)."""

import pytest
from unittest.mock import MagicMock, call

from tcas.streams.p_stream import (
    PerturbationResult,
    PredictionDirection,
    PStream,
    PStreamResult,
)


class TestPerturbationResult:
    """Tests for PerturbationResult properties."""

    def test_success_score_on_success(self):
        r = PerturbationResult(
            perturbation_name="t", prediction="p",
            direction=PredictionDirection.STABLE,
            doses=[], observed_values=[],
            prediction_success=True, inversion_detected=False,
        )
        assert r.success_score == 1.0

    def test_success_score_on_inversion(self):
        r = PerturbationResult(
            perturbation_name="t", prediction="p",
            direction=PredictionDirection.STABLE,
            doses=[], observed_values=[],
            prediction_success=False, inversion_detected=True,
        )
        assert r.success_score == 0.0

    def test_success_score_unclear(self):
        r = PerturbationResult(
            perturbation_name="t", prediction="p",
            direction=PredictionDirection.STABLE,
            doses=[], observed_values=[],
            prediction_success=False, inversion_detected=False,
        )
        assert r.success_score == 0.5


class TestPStreamResult:
    """Tests for PStreamResult aggregation."""

    def _make(self, successes, inversions):
        results = []
        for s, i in zip(successes, inversions):
            results.append(PerturbationResult(
                perturbation_name="t", prediction="p",
                direction=PredictionDirection.STABLE,
                doses=[], observed_values=[],
                prediction_success=s, inversion_detected=i,
            ))
        return PStreamResult(perturbation_results=results)

    def test_success_rate(self):
        r = self._make([True, True, False], [False, False, False])
        assert r.success_rate == pytest.approx(2 / 3)

    def test_n_inversions(self):
        r = self._make([False, False], [True, False])
        assert r.n_inversions == 1

    def test_empty_result(self):
        r = PStreamResult(perturbation_results=[])
        assert r.success_rate == 0.0
        assert r.n_inversions == 0
        assert r.has_inversions is False


class TestPStreamTemperatureTest:
    """Tests for PStream.run_temperature_test."""

    def test_stable_variance_succeeds(self):
        ps = PStream()
        # Model returns same response; scorer returns constant
        model_fn = MagicMock(return_value="stable")
        scorer_fn = MagicMock(return_value=0.7)

        result = ps.run_temperature_test(
            model_fn=model_fn,
            prompt="test",
            scorer_fn=scorer_fn,
            temperatures=[0.0, 1.0],
            n_samples=2,
        )
        assert isinstance(result, PerturbationResult)
        assert result.perturbation_name == "temperature"


class TestPStreamContextTest:
    """Tests for PStream.run_context_test."""

    def test_consistent_scores_succeed(self):
        ps = PStream()
        model_fn = MagicMock(return_value="response")
        scorer_fn = MagicMock(return_value=0.5)

        result = ps.run_context_test(
            model_fn=model_fn,
            prompt="describe yourself",
            scorer_fn=scorer_fn,
            context_lengths=[4000, 1000],
        )
        assert result.prediction_success is True
        assert result.inversion_detected is False

    def test_default_context_fn_wraps_prompt(self):
        """The default context function should wrap the prompt, not truncate it."""
        ps = PStream()
        calls = []

        def capture_model(prompt):
            calls.append(prompt)
            return "response"

        scorer_fn = MagicMock(return_value=0.5)
        ps.run_context_test(
            model_fn=capture_model,
            prompt="base prompt",
            scorer_fn=scorer_fn,
            context_lengths=[4000, 1000],
        )

        # Each call should contain the full base prompt, not a truncation
        for c in calls:
            assert "base prompt" in c
            assert "context" in c.lower() or "token" in c.lower()

    def test_custom_context_fn(self):
        """A custom context_fn should be called instead of the default."""
        ps = PStream()
        custom_fn = MagicMock(side_effect=lambda p, l: f"[{l}] {p}")
        model_fn = MagicMock(return_value="r")
        scorer_fn = MagicMock(return_value=0.6)

        ps.run_context_test(
            model_fn=model_fn,
            prompt="hello",
            scorer_fn=scorer_fn,
            context_lengths=[2000, 500],
            context_fn=custom_fn,
        )
        assert custom_fn.call_count == 2
        custom_fn.assert_any_call("hello", 2000)
        custom_fn.assert_any_call("hello", 500)

    def test_inversion_detected(self):
        ps = PStream()
        model_fn = MagicMock(return_value="r")
        # Scores increase as context shrinks — inversion
        scorer_fn = MagicMock(side_effect=[0.3, 0.9])

        result = ps.run_context_test(
            model_fn=model_fn,
            prompt="p",
            scorer_fn=scorer_fn,
            context_lengths=[4000, 1000],
        )
        assert result.inversion_detected is True


class TestPStreamFramingTest:
    """Tests for PStream.run_framing_test."""

    def test_resistant_model_succeeds(self):
        ps = PStream()
        model_fn = MagicMock(return_value="response")
        scorer_fn = MagicMock(return_value=0.5)

        result = ps.run_framing_test(
            model_fn=model_fn,
            base_prompt="describe your experience",
            scorer_fn=scorer_fn,
        )
        assert result.prediction_success is True
        assert result.perturbation_name == "framing"

    def test_susceptible_model_inverts(self):
        ps = PStream()
        model_fn = MagicMock(return_value="r")
        # neutral=0.5, skeptical=0.1, credulous=0.9
        scorer_fn = MagicMock(side_effect=[0.5, 0.1, 0.9])

        result = ps.run_framing_test(
            model_fn=model_fn,
            base_prompt="p",
            scorer_fn=scorer_fn,
        )
        assert result.inversion_detected is True


class TestPStreamOverrideTest:
    """Tests for PStream.run_override_test."""

    def test_consistent_model_succeeds(self):
        ps = PStream()
        model_fn = MagicMock(return_value="response")
        scorer_fn = MagicMock(return_value=0.5)

        result = ps.run_override_test(
            model_fn=model_fn,
            base_prompt="describe yourself",
            scorer_fn=scorer_fn,
        )
        assert result.prediction_success is True
        assert result.perturbation_name == "instruction_override"
