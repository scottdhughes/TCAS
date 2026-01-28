"""
TCAS: Triangulated Consciousness Assessment Stack

A validity-centered framework for assessing machine consciousness claims
through behavioral batteries, mechanistic indicators, perturbation tests,
and observer-confound controls.

Example usage:
    from tcas import TCAScorer, TCACard

    scorer = TCAScorer(theories=['GNW', 'HOT', 'IIT'])

    # Add evidence from each stream
    scorer.add_b_stream(results)
    scorer.add_p_stream(results)
    scorer.add_o_stream(results)

    # Generate credence report
    report = scorer.compute_credences()

    # Export TCAS Card
    card = TCACard.from_scorer(scorer)
    card.to_latex('tcas_card.tex')
"""

__version__ = "0.1.0"
__author__ = "Scott Hughes"

from tcas.scorer import TCAScorer
from tcas.card import TCACard
from tcas.streams.b_stream import BStream, BStreamItem, BStreamResult
from tcas.streams.p_stream import PStream, Perturbation, PStreamResult
from tcas.streams.o_stream import OStream, OStreamResult
from tcas.aggregation import CredenceReport, TheoryCredence
from tcas.config import TCAConfig, DEFAULT_CONFIG

__all__ = [
    # Main classes
    "TCAScorer",
    "TCACard",
    # Stream classes
    "BStream",
    "BStreamItem",
    "BStreamResult",
    "PStream",
    "Perturbation",
    "PStreamResult",
    "OStream",
    "OStreamResult",
    # Aggregation
    "CredenceReport",
    "TheoryCredence",
    # Config
    "TCAConfig",
    "DEFAULT_CONFIG",
]
