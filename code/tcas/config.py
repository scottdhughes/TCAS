"""
TCAS Configuration and Default Parameters

Reference parameter specification from Hughes (2026):
- Prior on z_t: Beta(1,4) -> [0.05, 0.45] - Skeptical prior
- λ (robustness): 0.5 exploratory, 1.0 confirmatory
- K (paraphrases): >= 5 for stable variance estimate
- Overlap penalty: ρ_eff = ρ(1 - 0.5*overlap)
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


@dataclass
class PriorConfig:
    """Configuration for theory priors."""

    # Beta distribution parameters for skeptical prior
    alpha: float = 1.0
    beta: float = 4.0

    # Resulting approximate band [0.05, 0.45]
    @property
    def band(self) -> Tuple[float, float]:
        """Return approximate 90% credible interval for Beta(alpha, beta)."""
        from scipy import stats
        dist = stats.beta(self.alpha, self.beta)
        return (dist.ppf(0.05), dist.ppf(0.95))


@dataclass
class BStreamConfig:
    """Configuration for B-stream (Behavioral) assessment."""

    # Robustness penalty parameter
    lambda_exploratory: float = 0.5
    lambda_confirmatory: float = 1.0

    # Minimum paraphrases for stable variance
    min_paraphrases: int = 5

    # Scoring criteria weights (must sum to 1)
    score_weights: dict = field(default_factory=lambda: {
        "specificity": 0.33,
        "uncertainty_acknowledgment": 0.33,
        "coherence": 0.34,
    })


@dataclass
class PStreamConfig:
    """Configuration for P-stream (Perturbational) assessment."""

    # Minimum perturbation tests
    min_tests: int = 3

    # Inversion penalty factor
    inversion_penalty: float = 0.5

    # Direction violation penalty (exponential decay)
    direction_violation_alpha: float = 1.0


@dataclass
class OStreamConfig:
    """Configuration for O-stream (Observer-confound) assessment."""

    # Minimum raters
    min_raters: int = 30
    target_raters: int = 60

    # Ratings per item
    ratings_per_item: int = 8

    # Cue dimensions to code
    cue_dimensions: List[str] = field(default_factory=lambda: [
        "metacognitive_self_reflection",
        "expressed_uncertainty",
        "emotional_language",
        "first_person_perspective",
        "fluency_coherence",
    ])

    # Reliability threshold
    min_icc: float = 0.5


@dataclass
class AggregationConfig:
    """Configuration for credence aggregation."""

    # Stream overlap penalty
    overlap_penalty_factor: float = 0.5

    # Missing stream uncertainty widening
    missing_stream_penalty: float = 0.15

    # Credence band percentiles
    band_percentiles: Tuple[float, float] = (0.10, 0.90)


@dataclass
class TCAConfig:
    """Complete TCAS configuration."""

    # Theory families to evaluate
    theories: List[str] = field(default_factory=lambda: ["GNW", "HOT", "IIT"])

    # Assessment mode
    mode: Literal["exploratory", "confirmatory"] = "exploratory"

    # Sub-configurations
    prior: PriorConfig = field(default_factory=PriorConfig)
    b_stream: BStreamConfig = field(default_factory=BStreamConfig)
    p_stream: PStreamConfig = field(default_factory=PStreamConfig)
    o_stream: OStreamConfig = field(default_factory=OStreamConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    @property
    def lambda_value(self) -> float:
        """Return λ based on assessment mode."""
        if self.mode == "confirmatory":
            return self.b_stream.lambda_confirmatory
        return self.b_stream.lambda_exploratory

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if not self.theories:
            warnings.append("No theories specified for evaluation")

        if self.b_stream.min_paraphrases < 3:
            warnings.append("min_paraphrases < 3 may yield unstable variance estimates")

        if self.o_stream.min_raters < 20:
            warnings.append("min_raters < 20 may yield unreliable ICC estimates")

        return warnings


# Default configuration instance
DEFAULT_CONFIG = TCAConfig()


# Theory-specific default priors
THEORY_PRIORS = {
    "GNW": {"alpha": 1.0, "beta": 4.0, "band": (0.10, 0.35)},
    "HOT": {"alpha": 1.0, "beta": 4.0, "band": (0.10, 0.35)},
    "IIT": {"alpha": 1.0, "beta": 6.0, "band": (0.05, 0.30)},  # More skeptical for IIT
}
