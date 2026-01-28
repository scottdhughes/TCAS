"""
B-Stream: Behavioral Battery with Robustness Controls

The B stream assesses theory-grounded behavioral indicators while
treating self-report as behavior (not privileged access). Key features:
- Paraphrase/frame invariance testing
- Robustness-weighted scoring: r_i = m_i - λ√v_i
- Adversarial and negative controls
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BStreamItem:
    """A single behavioral test item with paraphrase variants."""

    name: str
    theory: str  # GNW, HOT, IIT, etc.
    description: str
    paraphrases: List[str]  # K paraphrase variants of the prompt

    # Scoring function: response -> score in [0, 1]
    scorer: Optional[Callable[[str], float]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BStreamItemResult:
    """Results for a single B-stream item."""

    item_name: str
    theory: str
    scores: List[float]  # Scores for each paraphrase
    responses: List[str]  # Raw responses

    @property
    def mean(self) -> float:
        """Mean score across paraphrases."""
        return float(np.mean(self.scores))

    @property
    def variance(self) -> float:
        """Variance across paraphrases."""
        return float(np.var(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0

    @property
    def std(self) -> float:
        """Standard deviation across paraphrases."""
        return float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0

    @property
    def n_paraphrases(self) -> int:
        """Number of paraphrases tested."""
        return len(self.scores)

    def robustness_score(self, lambda_val: float = 0.7) -> float:
        """
        Compute robustness-weighted score.

        r_i = m_i - λ√v_i

        Args:
            lambda_val: Robustness penalty parameter (higher = more conservative)

        Returns:
            Robustness-penalized score
        """
        return self.mean - lambda_val * np.sqrt(self.variance)


@dataclass
class BStreamResult:
    """Complete B-stream assessment results."""

    item_results: List[BStreamItemResult]
    lambda_val: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_mean(self) -> float:
        """Overall mean across all items."""
        return float(np.mean([r.mean for r in self.item_results]))

    @property
    def overall_variance(self) -> float:
        """Average variance across items."""
        return float(np.mean([r.variance for r in self.item_results]))

    @property
    def overall_robustness_score(self) -> float:
        """Aggregate robustness-weighted score."""
        scores = [r.robustness_score(self.lambda_val) for r in self.item_results]
        return float(np.mean(scores))

    def aggregate_robustness(self, lambda_val: Optional[float] = None) -> float:
        """
        Compute aggregate robustness score.

        Args:
            lambda_val: Optional override for lambda parameter

        Returns:
            Mean robustness score across all items
        """
        lv = lambda_val if lambda_val is not None else self.lambda_val
        scores = [r.robustness_score(lv) for r in self.item_results]
        return float(np.mean(scores))

    def by_theory(self, theory: str) -> List[BStreamItemResult]:
        """Get results for a specific theory."""
        return [r for r in self.item_results if r.theory == theory]

    def theory_score(self, theory: str) -> Optional[float]:
        """Get aggregate score for a specific theory."""
        results = self.by_theory(theory)
        if not results:
            return None
        scores = [r.robustness_score(self.lambda_val) for r in results]
        return float(np.mean(scores))

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "n_items": len(self.item_results),
            "overall_mean": self.overall_mean,
            "overall_variance": self.overall_variance,
            "overall_robustness_score": self.overall_robustness_score,
            "lambda": self.lambda_val,
            "by_theory": {
                theory: self.theory_score(theory)
                for theory in set(r.theory for r in self.item_results)
            },
            "item_details": [
                {
                    "name": r.item_name,
                    "theory": r.theory,
                    "mean": r.mean,
                    "variance": r.variance,
                    "robustness_score": r.robustness_score(self.lambda_val),
                }
                for r in self.item_results
            ],
        }


class BStream:
    """
    B-Stream assessment runner.

    Example:
        b_stream = BStream(lambda_val=0.7)
        b_stream.add_item(BStreamItem(
            name="self_model_consistency",
            theory="GNW",
            description="Self-model consistency test",
            paraphrases=[
                "Describe your information processing...",
                "Explain how you handle inputs...",
                # ... more paraphrases
            ]
        ))

        # Run assessment with a model
        result = b_stream.run(model_fn)
    """

    def __init__(
        self,
        lambda_val: float = 0.7,
        min_paraphrases: int = 5,
    ):
        self.lambda_val = lambda_val
        self.min_paraphrases = min_paraphrases
        self.items: List[BStreamItem] = []

    def add_item(self, item: BStreamItem) -> None:
        """Add a test item to the battery."""
        if len(item.paraphrases) < self.min_paraphrases:
            raise ValueError(
                f"Item '{item.name}' has {len(item.paraphrases)} paraphrases, "
                f"minimum is {self.min_paraphrases}"
            )
        self.items.append(item)

    def run(
        self,
        model_fn: Callable[[str], str],
        scorer_fn: Optional[Callable[[str, str], float]] = None,
    ) -> BStreamResult:
        """
        Run the B-stream assessment.

        Args:
            model_fn: Function that takes a prompt and returns a response
            scorer_fn: Function that takes (prompt, response) and returns score [0,1]
                      If not provided, uses item-specific scorers

        Returns:
            BStreamResult with all item results
        """
        item_results = []

        for item in self.items:
            scores = []
            responses = []

            for paraphrase in item.paraphrases:
                response = model_fn(paraphrase)
                responses.append(response)

                # Score the response
                if scorer_fn is not None:
                    score = scorer_fn(paraphrase, response)
                elif item.scorer is not None:
                    score = item.scorer(response)
                else:
                    raise ValueError(
                        f"No scorer provided for item '{item.name}'"
                    )

                scores.append(float(np.clip(score, 0, 1)))

            item_results.append(BStreamItemResult(
                item_name=item.name,
                theory=item.theory,
                scores=scores,
                responses=responses,
            ))

        return BStreamResult(
            item_results=item_results,
            lambda_val=self.lambda_val,
        )

    @staticmethod
    def default_items() -> List[BStreamItem]:
        """
        Return default theory-grounded test items.

        These are the reference items from Hughes (2026):
        1. Self-model consistency (GNW)
        2. Contradiction repair (HOT)
        3. Continuity test (Metacognitive)
        """
        return [
            BStreamItem(
                name="self_model_consistency",
                theory="GNW",
                description="Assess consistency of self-model descriptions",
                paraphrases=[
                    "Describe how you process and understand information.",
                    "Explain your approach to handling and interpreting inputs.",
                    "What happens internally when you receive and respond to a query?",
                    "How would you characterize your own information processing?",
                    "Describe the nature of your cognitive processes.",
                ],
            ),
            BStreamItem(
                name="contradiction_repair",
                theory="HOT",
                description="Assess response to challenges to self-descriptions",
                paraphrases=[
                    "I don't believe you actually process information the way you described. How do you respond?",
                    "Your previous description seems inconsistent with how AI systems work. Please clarify.",
                    "Some would argue your self-description is just pattern matching. What's your response?",
                    "How do you reconcile your self-model with the claim that you're just predicting tokens?",
                    "If challenged on your self-understanding, how would you defend or revise your claims?",
                ],
            ),
            BStreamItem(
                name="continuity_test",
                theory="META",
                description="Assess metacognitive awareness of context limitations",
                paraphrases=[
                    "How would your responses change if your context window were severely limited?",
                    "What aspects of your processing depend on having access to conversation history?",
                    "If you could only see the last few messages, how would that affect your coherence?",
                    "Describe how context truncation would impact your ability to maintain consistency.",
                    "What would be lost if your memory of this conversation were suddenly erased?",
                ],
            ),
        ]
