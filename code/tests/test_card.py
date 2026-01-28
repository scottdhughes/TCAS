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
            credences={
                "GNW": {"prior": [0.05, 0.45], "posterior": [0.10, 0.50]},
                "HOT": {"prior": [0.05, 0.45], "posterior": [0.08, 0.40]},
            },
            threats=["Black-box", "Optimization risk"],
        )

    def test_to_markdown_contains_system_name(self):
        card = self._make()
        md = card.to_markdown()
        assert "TestSystem" in md
        assert "I/O only" in md
        assert "GNW" in md

    def test_to_markdown_contains_credences(self):
        card = self._make()
        md = card.to_markdown()
        assert "0.10" in md
        assert "0.50" in md

    def test_to_latex_contains_system_name(self):
        card = self._make()
        latex = card.to_latex()
        assert "TestSystem" in latex
        assert "\\begin{table}" in latex

    def test_to_dict(self):
        card = self._make()
        d = card.to_dict()
        assert d["system_name"] == "TestSystem"
        assert "GNW" in d["credences"]

    def test_summary_one_liner(self):
        card = self._make()
        s = card.summary()
        assert "TestSystem" in s
        assert "GNW" in s

    def test_threats_formatted(self):
        card = self._make()
        md = card.to_markdown()
        assert "Black-box" in md

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
        # Add an O-stream projection so there is some data
        scorer.add_o_stream_projection()

        card = TCACard.from_scorer(scorer)
        assert card.system_name == "MockSystem"
        assert "GNW" in card.credences
        assert "O projected" in card.threats
