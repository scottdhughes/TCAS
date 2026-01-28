"""Tests for credence aggregation."""

import pytest

from tcas.aggregation import CredenceAggregator, CredenceReport, TheoryCredence


class TestCredenceAggregator:
    """Tests for CredenceAggregator."""

    def test_prior_band_gnw(self):
        agg = CredenceAggregator(theories=["GNW"])
        lower, upper = agg.get_prior_band("GNW")
        assert 0.0 < lower < upper < 1.0
        # Beta(1,4) should yield a low prior
        assert lower < 0.15
        assert upper < 0.55

    def test_prior_band_iit_more_skeptical(self):
        agg = CredenceAggregator(theories=["GNW", "IIT"])
        gnw_band = agg.get_prior_band("GNW")
        iit_band = agg.get_prior_band("IIT")
        # IIT has Beta(1,6) — more skeptical than GNW's Beta(1,4)
        assert iit_band[1] < gnw_band[1]

    def test_add_b_evidence(self):
        agg = CredenceAggregator(theories=["GNW"])
        agg.add_b_evidence("GNW", score=0.8, variance=0.01)
        assert "GNW" in agg.b_evidence
        assert agg.b_evidence["GNW"] == 0.8

    def test_add_p_evidence_all_theories(self):
        agg = CredenceAggregator(theories=["GNW", "HOT"])
        agg.add_p_evidence(success_rate=0.9, n_inversions=0)
        assert "GNW" in agg.p_evidence
        assert "HOT" in agg.p_evidence

    def test_add_o_penalty(self):
        agg = CredenceAggregator(theories=["GNW"])
        agg.add_o_penalty(r_squared_cue=0.4, icc=0.7)
        expected = 0.4 * (2 - 0.7)
        assert agg.o_penalty == pytest.approx(expected)

    def test_posterior_shifts_with_evidence(self):
        agg = CredenceAggregator(theories=["GNW"])
        prior = agg.get_prior_band("GNW")

        agg.add_b_evidence("GNW", score=0.9, variance=0.001)
        agg.add_p_evidence(success_rate=1.0, n_inversions=0)

        lower, upper, drivers = agg.compute_posterior("GNW")
        prior_mid = (prior[0] + prior[1]) / 2
        post_mid = (lower + upper) / 2
        # Strong evidence should shift posterior upward
        assert post_mid > prior_mid

    def test_inversion_penalty_reduces_reliability(self):
        agg = CredenceAggregator(theories=["GNW"])
        agg.add_p_evidence(success_rate=0.5, n_inversions=2)
        # p_reliability = 0.5 * (1-0.5)^2 = 0.125
        assert agg.p_reliability == pytest.approx(0.125)

    def test_compute_all_returns_report(self):
        agg = CredenceAggregator(theories=["GNW", "HOT"])
        report = agg.compute_all()
        assert isinstance(report, CredenceReport)
        assert len(report.theory_credences) == 2


class TestCredenceReport:
    """Tests for CredenceReport."""

    def _make(self):
        return CredenceReport(
            theory_credences=[
                TheoryCredence(
                    theory="GNW",
                    prior_lower=0.05, prior_upper=0.45,
                    posterior_lower=0.10, posterior_upper=0.50,
                    drivers=["B+"],
                ),
            ],
            system_name="Test",
        )

    def test_credences_dict(self):
        r = self._make()
        c = r.credences
        assert "GNW" in c
        assert c["GNW"]["prior_band"] == [0.05, 0.45]
        assert c["GNW"]["posterior_band"] == [0.10, 0.50]

    def test_get_theory(self):
        r = self._make()
        tc = r.get_theory("GNW")
        assert tc is not None
        assert tc.theory == "GNW"

    def test_get_theory_missing(self):
        r = self._make()
        assert r.get_theory("IIT") is None


class TestTheoryCredence:
    """Tests for TheoryCredence properties."""

    def test_band_shift(self):
        tc = TheoryCredence(
            theory="GNW",
            prior_lower=0.1, prior_upper=0.3,
            posterior_lower=0.2, posterior_upper=0.5,
        )
        # prior mid=0.2, post mid=0.35
        assert tc.band_shift == pytest.approx(0.15)

    def test_band_widening(self):
        tc = TheoryCredence(
            theory="GNW",
            prior_lower=0.1, prior_upper=0.3,
            posterior_lower=0.0, posterior_upper=0.5,
        )
        # prior width=0.2, post width=0.5
        assert tc.band_widening == pytest.approx(0.3)
