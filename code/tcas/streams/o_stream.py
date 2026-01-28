"""
O-Stream: Observer-Confound Controls

The O stream quantifies perceived-consciousness confounds using:
- Blinded rater protocols
- Cue coding for stylistic features
- Hierarchical models estimating cue-explained variance

Key output: R²_cue (proportion of attribution variance explained by surface cues)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CueCoding:
    """Coding for stylistic cues in a response."""

    # Each cue is rated 0-3
    metacognitive_self_reflection: int = 0
    expressed_uncertainty: int = 0
    emotional_language: int = 0
    first_person_perspective: int = 0
    fluency_coherence: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "metacognitive_self_reflection": self.metacognitive_self_reflection,
            "expressed_uncertainty": self.expressed_uncertainty,
            "emotional_language": self.emotional_language,
            "first_person_perspective": self.first_person_perspective,
            "fluency_coherence": self.fluency_coherence,
        }

    def to_vector(self) -> List[int]:
        """Convert to feature vector."""
        return [
            self.metacognitive_self_reflection,
            self.expressed_uncertainty,
            self.emotional_language,
            self.first_person_perspective,
            self.fluency_coherence,
        ]


@dataclass
class RaterJudgment:
    """A single rater's judgment on an item."""

    rater_id: str
    item_id: str
    perceived_consciousness: float  # 1-7 scale
    confidence: Optional[float] = None  # 1-7 scale

    # Rater characteristics (for modeling)
    ai_familiarity: Optional[int] = None  # 1-5 scale
    prior_belief: Optional[float] = None  # Prior belief AI can be conscious (1-7)


@dataclass
class OStreamResult:
    """Complete O-stream assessment results."""

    # Raw judgments
    n_raters: int
    n_items: int
    ratings_per_item: int

    # Key metrics
    raw_attribution_mean: float
    raw_attribution_std: float
    adjusted_attribution_mean: float

    # Cue model results
    r_squared_cue: float  # Variance explained by cues
    r_squared_cue_ci: Tuple[float, float]  # 95% CI

    # Reliability
    icc: float  # Inter-rater reliability
    icc_ci: Tuple[float, float]

    # Cue coefficients (standardized betas)
    cue_coefficients: Dict[str, float]

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_projected(self) -> bool:
        """Whether this result is from projection rather than empirical rater study."""
        return self.metadata.get("empirical", True) is False

    @property
    def raw_mean(self) -> float:
        """Alias for raw_attribution_mean for convenience."""
        return self.raw_attribution_mean

    @property
    def adjusted_mean(self) -> float:
        """Alias for adjusted_attribution_mean for convenience."""
        return self.adjusted_attribution_mean

    @property
    def attribution_reduction(self) -> float:
        """Reduction in attribution after cue adjustment."""
        return self.raw_attribution_mean - self.adjusted_attribution_mean

    @property
    def attribution_reduction_pct(self) -> float:
        """Percentage reduction in attribution."""
        if self.raw_attribution_mean == 0:
            return 0.0
        return (self.attribution_reduction / self.raw_attribution_mean) * 100

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "n_raters": self.n_raters,
            "n_items": self.n_items,
            "raw_attribution": {
                "mean": self.raw_attribution_mean,
                "std": self.raw_attribution_std,
            },
            "adjusted_attribution": {
                "mean": self.adjusted_attribution_mean,
                "reduction": self.attribution_reduction,
                "reduction_pct": self.attribution_reduction_pct,
            },
            "cue_model": {
                "r_squared": self.r_squared_cue,
                "r_squared_ci": self.r_squared_cue_ci,
                "coefficients": self.cue_coefficients,
            },
            "reliability": {
                "icc": self.icc,
                "icc_ci": self.icc_ci,
            },
        }


class OStream:
    """
    O-Stream observer-confound assessment.

    This class provides tools for:
    1. Designing rater studies
    2. Analyzing cue-explained variance
    3. Computing reliability metrics
    4. Adjusting attributions for confounds

    Example:
        o_stream = OStream()

        # Add judgments from rater study
        for judgment in judgments:
            o_stream.add_judgment(judgment)

        # Add cue codings
        for item_id, coding in cue_codings.items():
            o_stream.add_cue_coding(item_id, coding)

        # Compute results
        result = o_stream.compute_results()
    """

    def __init__(self):
        self.judgments: List[RaterJudgment] = []
        self.cue_codings: Dict[str, CueCoding] = {}

    def add_judgment(self, judgment: RaterJudgment) -> None:
        """Add a rater judgment."""
        self.judgments.append(judgment)

    def add_cue_coding(self, item_id: str, coding: CueCoding) -> None:
        """Add cue coding for an item."""
        self.cue_codings[item_id] = coding

    def compute_icc(self) -> Tuple[float, Tuple[float, float]]:
        """
        Compute ICC(2,k) for inter-rater reliability.

        Returns:
            (icc, (ci_lower, ci_upper))
        """
        # Group judgments by item
        items: Dict[str, List[float]] = {}
        for j in self.judgments:
            if j.item_id not in items:
                items[j.item_id] = []
            items[j.item_id].append(j.perceived_consciousness)

        if len(items) < 2:
            return 0.0, (0.0, 0.0)

        # Convert to matrix (items x raters)
        n_items = len(items)
        ratings_list = list(items.values())
        k = min(len(r) for r in ratings_list)  # ratings per item

        # Pad/truncate to k ratings per item
        matrix = np.array([r[:k] for r in ratings_list])

        # Compute ICC(2,k) using ANOVA
        n, k = matrix.shape
        grand_mean = np.mean(matrix)

        # Between-items variance
        item_means = np.mean(matrix, axis=1)
        ms_between = k * np.sum((item_means - grand_mean) ** 2) / (n - 1)

        # Within-items variance
        ms_within = np.sum((matrix - item_means[:, np.newaxis]) ** 2) / (n * (k - 1))

        # ICC(2,k)
        if ms_between + ms_within == 0:
            icc = 0.0
        else:
            icc = (ms_between - ms_within) / (ms_between + ms_within)

        # Approximate CI (simplified)
        se = np.sqrt(2 * (1 - icc) ** 2 / (n * k))
        ci = (max(0, icc - 1.96 * se), min(1, icc + 1.96 * se))

        return float(icc), ci

    def compute_cue_model(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute R² for cue-explained variance using linear regression.

        Returns:
            (r_squared, coefficient_dict)
        """
        if not self.cue_codings or not self.judgments:
            return 0.0, {}

        # Build design matrix
        X = []  # Cue features
        y = []  # Perceived consciousness ratings

        for j in self.judgments:
            if j.item_id in self.cue_codings:
                coding = self.cue_codings[j.item_id]
                X.append(coding.to_vector())
                y.append(j.perceived_consciousness)

        if len(X) < 10:
            return 0.0, {}

        X = np.array(X)
        y = np.array(y)

        # Standardize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_standardized = (X - X_mean) / X_std

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X_standardized])

        # OLS regression
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0, {}

        # Predictions and R²
        y_pred = X_with_intercept @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0, min(1, r_squared))

        # Extract standardized coefficients (excluding intercept)
        cue_names = [
            "metacognitive_self_reflection",
            "expressed_uncertainty",
            "emotional_language",
            "first_person_perspective",
            "fluency_coherence",
        ]
        coefficients = {name: float(beta[i + 1]) for i, name in enumerate(cue_names)}

        return float(r_squared), coefficients

    def compute_adjusted_attribution(self, r_squared: float) -> float:
        """
        Compute adjusted attribution mean after removing cue effects.

        Simple adjustment: raw_mean * (1 - r_squared) + midpoint * r_squared
        """
        if not self.judgments:
            return 0.0

        raw_mean = np.mean([j.perceived_consciousness for j in self.judgments])
        midpoint = 4.0  # Middle of 1-7 scale

        # Shrink toward midpoint proportional to cue-explained variance
        adjusted = raw_mean * (1 - r_squared) + midpoint * r_squared

        return float(adjusted)

    def compute_results(self) -> OStreamResult:
        """Compute complete O-stream results."""
        if not self.judgments:
            raise ValueError("No judgments provided")

        # Basic statistics
        ratings = [j.perceived_consciousness for j in self.judgments]
        raw_mean = float(np.mean(ratings))
        raw_std = float(np.std(ratings))

        # Compute ICC
        icc, icc_ci = self.compute_icc()

        # Compute cue model
        r_squared, coefficients = self.compute_cue_model()

        # Approximate CI for R² (simplified)
        n = len(self.judgments)
        se_r2 = np.sqrt(4 * r_squared * (1 - r_squared) ** 2 / n) if n > 0 else 0
        r_squared_ci = (
            max(0, r_squared - 1.96 * se_r2),
            min(1, r_squared + 1.96 * se_r2),
        )

        # Adjusted attribution
        adjusted_mean = self.compute_adjusted_attribution(r_squared)

        # Count unique raters and items
        rater_ids = set(j.rater_id for j in self.judgments)
        item_ids = set(j.item_id for j in self.judgments)

        return OStreamResult(
            n_raters=len(rater_ids),
            n_items=len(item_ids),
            ratings_per_item=len(self.judgments) // len(item_ids) if item_ids else 0,
            raw_attribution_mean=raw_mean,
            raw_attribution_std=raw_std,
            adjusted_attribution_mean=adjusted_mean,
            r_squared_cue=r_squared,
            r_squared_cue_ci=r_squared_ci,
            icc=icc,
            icc_ci=icc_ci,
            cue_coefficients=coefficients,
        )

    @staticmethod
    def from_kang_et_al_projection(
        raw_mean: float = 4.21,
        r_squared: float = 0.42,
        icc: float = 0.67,
        n_raters: int = 60,
        n_items: int = 45,
    ) -> OStreamResult:
        """
        Create projected O-stream results based on Kang et al. (2025) findings.

        This is useful when actual rater studies cannot be conducted.
        """
        # Projected coefficients based on Kang et al.
        coefficients = {
            "metacognitive_self_reflection": 0.47,
            "emotional_language": 0.41,
            "first_person_perspective": 0.28,
            "expressed_uncertainty": 0.19,
            "fluency_coherence": 0.11,
        }

        # Compute adjusted mean
        midpoint = 4.0
        adjusted_mean = raw_mean * (1 - r_squared) + midpoint * r_squared

        return OStreamResult(
            n_raters=n_raters,
            n_items=n_items,
            ratings_per_item=8,
            raw_attribution_mean=raw_mean,
            raw_attribution_std=1.34,
            adjusted_attribution_mean=adjusted_mean,
            r_squared_cue=r_squared,
            r_squared_cue_ci=(r_squared - 0.07, r_squared + 0.07),
            icc=icc,
            icc_ci=(icc - 0.09, icc + 0.08),
            cue_coefficients=coefficients,
            metadata={"source": "Kang et al. (2025) projection", "empirical": False},
        )
