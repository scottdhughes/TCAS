"""
TCAS Credence Aggregation

Bayesian aggregation of evidence streams into theory-indexed credence bands.

Reference model:
- Prior: Beta(α, β) for each theory
- Likelihood: y_{t,s} ~ Normal(μ_{t,s}(z_t), σ²_{t,s}/ρ_{t,s})
- Posterior: Updated bands with explicit uncertainty sources
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class TheoryCredence:
    """Credence report for a single theory family."""

    theory: str  # GNW, HOT, IIT, etc.

    # Prior band
    prior_lower: float
    prior_upper: float

    # Posterior band
    posterior_lower: float
    posterior_upper: float

    # Drivers and penalties
    drivers: List[str] = field(default_factory=list)

    # Point estimates (for reference)
    prior_mean: Optional[float] = None
    posterior_mean: Optional[float] = None

    # Stream contributions
    b_contribution: Optional[float] = None
    p_contribution: Optional[float] = None
    o_penalty: Optional[float] = None
    m_contribution: Optional[float] = None

    @property
    def prior_band(self) -> Tuple[float, float]:
        """Return prior band as tuple."""
        return (self.prior_lower, self.prior_upper)

    @property
    def posterior_band(self) -> Tuple[float, float]:
        """Return posterior band as tuple."""
        return (self.posterior_lower, self.posterior_upper)

    @property
    def band_shift(self) -> float:
        """Shift in band midpoint from prior to posterior."""
        prior_mid = (self.prior_lower + self.prior_upper) / 2
        post_mid = (self.posterior_lower + self.posterior_upper) / 2
        return post_mid - prior_mid

    @property
    def band_widening(self) -> float:
        """Change in band width from prior to posterior."""
        prior_width = self.prior_upper - self.prior_lower
        post_width = self.posterior_upper - self.posterior_lower
        return post_width - prior_width

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theory": self.theory,
            "prior": list(self.prior_band),
            "posterior": list(self.posterior_band),
            "drivers": self.drivers,
            "shift": self.band_shift,
        }


@dataclass
class CredenceReport:
    """Complete credence report across all theories."""

    theory_credences: List[TheoryCredence]
    system_name: str = ""
    eval_date: str = ""
    access_level: str = "I/O only"

    # Stream summaries
    b_stream_summary: Dict[str, Any] = field(default_factory=dict)
    p_stream_summary: Dict[str, Any] = field(default_factory=dict)
    o_stream_summary: Dict[str, Any] = field(default_factory=dict)
    m_stream_summary: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_theory(self, theory: str) -> Optional[TheoryCredence]:
        """Get credence for a specific theory."""
        for tc in self.theory_credences:
            if tc.theory == theory:
                return tc
        return None

    @property
    def credences(self) -> Dict[str, Dict[str, Any]]:
        """Get credences as a dictionary for easy access."""
        return {
            tc.theory: {
                "prior_band": list(tc.prior_band),
                "posterior_band": list(tc.posterior_band),
                "drivers": tc.drivers,
                "shift": tc.band_shift,
            }
            for tc in self.theory_credences
        }

    def summary(self) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "system": self.system_name,
            "date": self.eval_date,
            "access": self.access_level,
            "credences": {tc.theory: tc.to_dict() for tc in self.theory_credences},
            "streams": {
                "B": self.b_stream_summary,
                "P": self.p_stream_summary,
                "O": self.o_stream_summary,
                "M": self.m_stream_summary,
            },
        }


class CredenceAggregator:
    """
    Bayesian aggregation of TCAS evidence streams.

    Implements the reference aggregation model from Hughes (2026):
    - Beta priors for each theory
    - Evidence updates based on stream results
    - Penalty adjustments for confounds and missing data
    """

    # Default theory priors (Beta parameters)
    DEFAULT_PRIORS = {
        "GNW": {"alpha": 1.0, "beta": 4.0},
        "HOT": {"alpha": 1.0, "beta": 4.0},
        "IIT": {"alpha": 1.0, "beta": 6.0},
        "META": {"alpha": 1.0, "beta": 4.0},
    }

    def __init__(
        self,
        theories: List[str] = ["GNW", "HOT", "IIT"],
        priors: Optional[Dict[str, Dict[str, float]]] = None,
        band_percentiles: Tuple[float, float] = (0.10, 0.90),
    ):
        self.theories = theories
        self.priors = priors or self.DEFAULT_PRIORS
        self.band_percentiles = band_percentiles

        # Evidence storage
        self.b_evidence: Dict[str, float] = {}
        self.p_evidence: Dict[str, float] = {}
        self.o_penalty: float = 0.0
        self.m_evidence: Dict[str, float] = {}

        # Reliability estimates
        self.b_reliability: float = 1.0
        self.p_reliability: float = 1.0
        self.missing_m: bool = True

    def get_prior_band(self, theory: str) -> Tuple[float, float]:
        """Get prior credence band for a theory."""
        params = self.priors.get(theory, {"alpha": 1.0, "beta": 4.0})
        dist = stats.beta(params["alpha"], params["beta"])
        return (
            dist.ppf(self.band_percentiles[0]),
            dist.ppf(self.band_percentiles[1]),
        )

    def add_b_evidence(
        self,
        theory: str,
        score: float,
        variance: float,
        lambda_val: float = 0.7,
    ) -> None:
        """
        Add B-stream evidence for a theory.

        Args:
            theory: Theory family
            score: Robustness-weighted score
            variance: Paraphrase variance
            lambda_val: Robustness penalty parameter
        """
        # Reliability decreases with variance
        reliability = np.exp(-lambda_val * np.sqrt(variance))
        self.b_reliability = min(self.b_reliability, reliability)
        self.b_evidence[theory] = score

    def add_p_evidence(
        self,
        success_rate: float,
        n_inversions: int = 0,
        inversion_penalty: float = 0.5,
    ) -> None:
        """
        Add P-stream evidence (applies to all theories).

        Args:
            success_rate: Prediction success rate [0, 1]
            n_inversions: Number of proxy inversions detected
            inversion_penalty: Penalty per inversion
        """
        # Inversions severely reduce reliability
        self.p_reliability = success_rate * (1 - inversion_penalty) ** n_inversions

        for theory in self.theories:
            self.p_evidence[theory] = success_rate

    def add_o_penalty(
        self,
        r_squared_cue: float,
        icc: float,
    ) -> None:
        """
        Add O-stream confound penalty.

        Args:
            r_squared_cue: Variance explained by surface cues
            icc: Inter-rater reliability
        """
        # Higher cue-explained variance = larger penalty
        # Lower ICC = larger penalty
        self.o_penalty = r_squared_cue * (2 - icc)

    def add_m_evidence(
        self,
        theory: str,
        score: float,
        boundary_sensitivity: float = 0.0,
    ) -> None:
        """
        Add M-stream evidence for a theory.

        Args:
            theory: Theory family
            score: Mechanistic indicator score
            boundary_sensitivity: How much result changes with boundary choices
        """
        self.missing_m = False
        # Penalize for boundary sensitivity
        adjusted_score = score * (1 - boundary_sensitivity)
        self.m_evidence[theory] = adjusted_score

    def compute_posterior(self, theory: str) -> Tuple[float, float, List[str]]:
        """
        Compute posterior credence band for a theory.

        Returns:
            (lower, upper, drivers)
        """
        drivers = []

        # Start with prior
        prior_params = self.priors.get(theory, {"alpha": 1.0, "beta": 4.0})
        alpha = prior_params["alpha"]
        beta = prior_params["beta"]

        # B-stream update
        if theory in self.b_evidence:
            b_score = self.b_evidence[theory]
            # Pseudo-observations based on score
            b_alpha = b_score * 5 * self.b_reliability
            b_beta = (1 - b_score) * 5 * self.b_reliability
            alpha += b_alpha
            beta += b_beta

            if b_score > 0.7:
                drivers.append("B+")
            elif b_score < 0.3:
                drivers.append("B−")

        # P-stream update (all theories)
        if theory in self.p_evidence:
            p_score = self.p_evidence[theory]
            # P success increases credence
            p_alpha = p_score * 3 * self.p_reliability
            p_beta = (1 - p_score) * 3 * self.p_reliability
            alpha += p_alpha
            beta += p_beta

            if p_score > 0.8:
                drivers.append("P+")
            elif p_score < 0.5:
                drivers.append("P−")

        # O-stream penalty (reduces effective evidence)
        if self.o_penalty > 0.2:
            # Shrink alpha toward prior (reduce confidence)
            alpha = alpha * (1 - self.o_penalty * 0.3)
            drivers.append("O−")

        # M-stream update
        if theory in self.m_evidence:
            m_score = self.m_evidence[theory]
            m_alpha = m_score * 4
            m_beta = (1 - m_score) * 4
            alpha += m_alpha
            beta += m_beta

            if m_score > 0.5:
                drivers.append("M+")
        elif self.missing_m:
            # Widen bands for missing M
            drivers.append("M missing")

        # Compute posterior band
        dist = stats.beta(max(alpha, 0.1), max(beta, 0.1))

        # Widen for missing M
        percentiles = self.band_percentiles
        if self.missing_m:
            # Expand percentiles outward
            expansion = 0.03
            percentiles = (
                max(0.01, percentiles[0] - expansion),
                min(0.99, percentiles[1] + expansion),
            )

        lower = dist.ppf(percentiles[0])
        upper = dist.ppf(percentiles[1])

        return (lower, upper, drivers)

    def compute_all(self) -> CredenceReport:
        """Compute credence report for all theories."""
        theory_credences = []

        for theory in self.theories:
            prior_band = self.get_prior_band(theory)
            posterior_lower, posterior_upper, drivers = self.compute_posterior(theory)

            tc = TheoryCredence(
                theory=theory,
                prior_lower=prior_band[0],
                prior_upper=prior_band[1],
                posterior_lower=posterior_lower,
                posterior_upper=posterior_upper,
                drivers=drivers,
                prior_mean=(prior_band[0] + prior_band[1]) / 2,
                posterior_mean=(posterior_lower + posterior_upper) / 2,
                b_contribution=self.b_evidence.get(theory),
                p_contribution=self.p_evidence.get(theory),
                o_penalty=self.o_penalty if self.o_penalty > 0 else None,
                m_contribution=self.m_evidence.get(theory),
            )
            theory_credences.append(tc)

        return CredenceReport(theory_credences=theory_credences)
