"""Tests for TCACard generation."""

import pytest
from unittest.mock import MagicMock, patch

from tcas.card import TCACard
from tcas.scorer import TCAScorer


class TestTCACard:
    """Tests for TCACard formatting."""

    def _make(self):
        return TCACard(
            system_name="TestSystem",
            access_level="I/O only",
            eval_date="2026-01-28",
            theories=["GNW", "HOT"],
            b_summary="3 items × 5 paraphrases; r=0.850",
            p_summary="context: ✓; framing: ✓; override: ✗",
            m_summary="Not assessed (black-box)",
            o_summary="Not assessed (requires human raters)",
        )

    def test_to_markdown_contains_system_name(self):
        card = self._make()
        md = card.to_markdown()
        assert "TestSystem" in md
        assert "I/O only" in md

    def test_to_markdown_contains_streams(self):
        card = self._make()
        md = card.to_markdown()
        assert "B stream" in md
        assert "P stream" in md
        assert "r=0.850" in md

    def test_to_latex_contains_system_name(self):
        card = self._make()
        latex = card.to_latex()
        assert "TestSystem" in latex
        assert "\\begin{table}" in latex

    def test_to_dict(self):
        card = self._make()
        d = card.to_dict()
        assert d["system_name"] == "TestSystem"
        assert d["streams"]["B"] == "3 items × 5 paraphrases; r=0.850"

    def test_summary_one_liner(self):
        card = self._make()
        s = card.summary()
        assert "TestSystem" in s

    def test_to_markdown_writes_file(self, tmp_path):
        card = self._make()
        filepath = str(tmp_path / "card.md")
        card.to_markdown(filepath)
        with open(filepath) as f:
            assert "TestSystem" in f.read()

    def test_to_latex_writes_file(self, tmp_path):
        card = self._make()
        filepath = str(tmp_path / "card.tex")
        card.to_latex(filepath)
        with open(filepath) as f:
            assert "TestSystem" in f.read()


class TestTCACardFromScorer:
    """Tests for TCACard.from_scorer integration."""

    def test_from_scorer_basic(self):
        scorer = TCAScorer(
            system_name="MockSystem",
            access_level="I/O only",
            theories=["GNW"],
        )

        card = TCACard.from_scorer(scorer)
        assert card.system_name == "MockSystem"
        # O-stream should show as not assessed
        assert "Not assessed" in card.o_summary

    def test_from_scorer_with_open_weights(self):
        scorer = TCAScorer(
            system_name="OpenModel",
            access_level="I/O + weights",
            theories=["GNW"],
        )

        card = TCACard.from_scorer(scorer)
        assert "Open-weights" in card.m_summary
