"""Tests for TCAConfig."""

import pytest

from tcas.config import TCAConfig, PriorConfig


class TestTCAConfig:
    """Tests for TCAConfig defaults and validation."""

    def test_default_theories(self):
        cfg = TCAConfig()
        assert cfg.theories == ["GNW", "HOT", "IIT"]

    def test_default_mode(self):
        cfg = TCAConfig()
        assert cfg.mode == "exploratory"

    def test_lambda_exploratory(self):
        cfg = TCAConfig(mode="exploratory")
        assert cfg.lambda_value == 0.5

    def test_lambda_confirmatory(self):
        cfg = TCAConfig(mode="confirmatory")
        assert cfg.lambda_value == 1.0

    def test_validate_no_warnings(self):
        cfg = TCAConfig()
        warnings = cfg.validate()
        assert len(warnings) == 0

    def test_validate_no_theories(self):
        cfg = TCAConfig(theories=[])
        warnings = cfg.validate()
        assert any("No theories" in w for w in warnings)

    def test_validate_low_paraphrases(self):
        from tcas.config import BStreamConfig
        cfg = TCAConfig()
        cfg.b_stream = BStreamConfig(min_paraphrases=2)
        warnings = cfg.validate()
        assert any("paraphrases" in w.lower() for w in warnings)

    def test_validate_low_raters(self):
        from tcas.config import OStreamConfig
        cfg = TCAConfig()
        cfg.o_stream = OStreamConfig(min_raters=10)
        warnings = cfg.validate()
        assert any("raters" in w.lower() for w in warnings)


class TestPriorConfig:
    """Tests for PriorConfig."""

    def test_default_band(self):
        pc = PriorConfig()
        lower, upper = pc.band
        assert 0.0 < lower < upper < 1.0
