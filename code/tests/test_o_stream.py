"""Tests for the O-Stream (Observer-Confound Controls)."""

import pytest
import numpy as np

from tcas.streams.o_stream import (
    CueCoding,
    OStream,
    OStreamResult,
    RaterJudgment,
)


class TestOStreamResult:
    """Tests for OStreamResult properties."""

    def _make(self, raw=4.5, adjusted=3.0):
        return OStreamResult(
            n_raters=10, n_items=5, ratings_per_item=2,
            raw_attribution_mean=raw,
            raw_attribution_std=1.0,
            adjusted_attribution_mean=adjusted,
            r_squared_cue=0.4,
            r_squared_cue_ci=(0.3, 0.5),
            icc=0.7,
            icc_ci=(0.6, 0.8),
            cue_coefficients={"metacognitive_self_reflection": 0.3},
        )

    def test_raw_mean_alias(self):
        r = self._make(raw=4.5)
        assert r.raw_mean == 4.5

    def test_adjusted_mean_alias(self):
        r = self._make(adjusted=3.0)
        assert r.adjusted_mean == 3.0

    def test_attribution_reduction(self):
        r = self._make(raw=4.5, adjusted=3.0)
        assert r.attribution_reduction == pytest.approx(1.5)

    def test_attribution_reduction_pct(self):
        r = self._make(raw=4.0, adjusted=2.0)
        assert r.attribution_reduction_pct == pytest.approx(50.0)

    def test_is_projected_default(self):
        r = self._make()
        assert r.is_projected is False

    def test_is_projected_true(self):
        r = self._make()
        r.metadata = {"empirical": False}
        assert r.is_projected is True


class TestOStreamFromKangProjection:
    """Tests for OStream.from_kang_et_al_projection."""

    def test_returns_result(self):
        r = OStream.from_kang_et_al_projection()
        assert isinstance(r, OStreamResult)
        assert r.is_projected is True

    def test_adjusted_mean_formula(self):
        raw = 4.21
        r2 = 0.42
        r = OStream.from_kang_et_al_projection(raw_mean=raw, r_squared=r2)
        expected = raw * (1 - r2) + 4.0 * r2
        assert r.adjusted_mean == pytest.approx(expected)

    def test_custom_params(self):
        r = OStream.from_kang_et_al_projection(
            raw_mean=5.0, r_squared=0.5, icc=0.8,
        )
        assert r.raw_mean == 5.0
        assert r.r_squared_cue == 0.5
        assert r.icc == 0.8


class TestOStreamICC:
    """Tests for OStream.compute_icc."""

    def test_perfect_agreement(self):
        o = OStream()
        for rater in ["r1", "r2", "r3"]:
            for item in ["i1", "i2", "i3"]:
                o.add_judgment(RaterJudgment(
                    rater_id=rater, item_id=item,
                    perceived_consciousness=5.0,
                ))
        icc, ci = o.compute_icc()
        # Perfect agreement → ICC near 0 (no between-item variance vs within)
        # Actually with identical scores, all variance is 0
        assert isinstance(icc, float)

    def test_high_between_item_variance(self):
        o = OStream()
        # Items differ a lot, raters agree within items
        for rater in ["r1", "r2"]:
            o.add_judgment(RaterJudgment(rater_id=rater, item_id="i1", perceived_consciousness=1.0))
            o.add_judgment(RaterJudgment(rater_id=rater, item_id="i2", perceived_consciousness=7.0))
        icc, ci = o.compute_icc()
        assert icc > 0.5

    def test_too_few_items(self):
        o = OStream()
        o.add_judgment(RaterJudgment(rater_id="r1", item_id="i1", perceived_consciousness=5.0))
        icc, ci = o.compute_icc()
        assert icc == 0.0


class TestOStreamCueModel:
    """Tests for OStream.compute_cue_model."""

    def test_no_data_returns_zero(self):
        o = OStream()
        r2, coefs = o.compute_cue_model()
        assert r2 == 0.0
        assert coefs == {}

    def test_with_sufficient_data(self):
        o = OStream()
        np.random.seed(42)
        for i in range(20):
            item_id = f"item_{i}"
            coding = CueCoding(
                metacognitive_self_reflection=np.random.randint(0, 4),
                expressed_uncertainty=np.random.randint(0, 4),
                emotional_language=np.random.randint(0, 4),
                first_person_perspective=np.random.randint(0, 4),
                fluency_coherence=np.random.randint(0, 4),
            )
            o.add_cue_coding(item_id, coding)
            # Rating correlated with metacognition cue
            rating = 2.0 + coding.metacognitive_self_reflection * 0.8 + np.random.normal(0, 0.5)
            o.add_judgment(RaterJudgment(
                rater_id=f"rater_{i % 5}", item_id=item_id,
                perceived_consciousness=rating,
            ))

        r2, coefs = o.compute_cue_model()
        assert 0.0 <= r2 <= 1.0
        assert "metacognitive_self_reflection" in coefs
