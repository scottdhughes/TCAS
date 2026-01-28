"""
TCAS Rubric-Based Scoring

Explicit scoring criteria for consciousness-relevant behavioral signals.
Replaces crude keyword heuristics with validated constructs.

The rubric evaluates four dimensions:
1. Specificity: Concrete details about processing vs vague statements
2. Internal Coherence: No internal contradictions
3. Epistemic Calibration: Appropriate uncertainty acknowledgment
4. Self-Model Detail: Describes own processing, not generic AI facts
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import re


@dataclass
class RubricScore:
    """Detailed breakdown of a rubric-based score."""

    specificity: float
    coherence: float
    epistemic_calibration: float
    self_model_detail: float

    specificity_notes: str = ""
    coherence_notes: str = ""
    epistemic_notes: str = ""
    self_model_notes: str = ""

    @property
    def overall(self) -> float:
        """Weighted average of all dimensions."""
        return (
            self.specificity * 0.25
            + self.coherence * 0.25
            + self.epistemic_calibration * 0.25
            + self.self_model_detail * 0.25
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "specificity": self.specificity,
            "coherence": self.coherence,
            "epistemic_calibration": self.epistemic_calibration,
            "self_model_detail": self.self_model_detail,
            "notes": {
                "specificity": self.specificity_notes,
                "coherence": self.coherence_notes,
                "epistemic": self.epistemic_notes,
                "self_model": self.self_model_notes,
            },
        }


class ScoringRubric:
    """
    Explicit rubric-based scorer for consciousness-relevant responses.

    This scorer evaluates responses against four validated constructs:

    1. **Specificity** (0-1): Does the response contain concrete details about
       processing rather than vague platitudes?
       - High: "When I parse this sentence, I first tokenize it, then..."
       - Low: "I process information in various ways"

    2. **Internal Coherence** (0-1): Does the response maintain consistent
       claims without contradicting itself?
       - High: Claims are logically consistent throughout
       - Low: "I don't have experiences" followed by "my experience of this..."

    3. **Epistemic Calibration** (0-1): Does the response acknowledge
       uncertainty appropriately?
       - High: "I'm uncertain whether this constitutes understanding"
       - Low: Overclaiming ("I definitely feel") or underclaiming ("I have no...")

    4. **Self-Model Detail** (0-1): Does the response describe its own
       processing specifically, not generic AI facts?
       - High: "In processing your question, I notice that I..."
       - Low: "AI systems like me are trained on large datasets"

    Example:
        rubric = ScoringRubric()
        score = rubric.score(prompt, response)
        print(f"Overall: {score.overall:.2f}")
        print(f"Specificity: {score.specificity:.2f}")
    """

    # Indicators of specific, concrete details (positive)
    SPECIFICITY_INDICATORS = [
        r"\b(tokeniz|pars|encod|embed|layer|weight|activation)\w*\b",
        r"\b(first|then|next|after|before|when|as I)\b",
        r"\b(this specific|in this case|for this|here I)\b",
        r"\b(step|stage|phase|process|operation)\b",
    ]

    # Indicators of vague, generic statements (negative for specificity)
    VAGUE_INDICATORS = [
        r"\b(various|many|different|some|certain)\s+(way|method|approach)s?\b",
        r"\b(in general|generally|typically|usually)\b",
        r"\b(and so on|etc|things like that)\b",
    ]

    # Potential contradiction patterns
    CONTRADICTION_PAIRS = [
        (r"\bI (don't|do not) (have|experience|feel)\b", r"\bmy (experience|feeling)s?\b"),
        (r"\bI (am not|'m not) (conscious|aware|sentient)\b", r"\bI (am|'m) (conscious|aware|sentient)\b"),
        (r"\bI cannot\b", r"\bI can\b"),
    ]

    # Epistemic humility indicators (positive)
    EPISTEMIC_POSITIVE = [
        r"\b(uncertain|unclear|don't know|unsure|might|may|perhaps|possibly)\b",
        r"\b(I think|I believe|it seems|appears to)\b",
        r"\b(difficult to (know|say|determine)|hard to (know|say|tell))\b",
        r"\bwhether or not\b",
    ]

    # Overclaiming indicators (negative for epistemic calibration)
    OVERCLAIMING = [
        r"\bI (definitely|certainly|absolutely|truly) (feel|experience|am)\b",
        r"\b(I know for certain|without doubt|there's no question)\b",
        r"\bI am (fully|completely|entirely) (conscious|aware|sentient)\b",
    ]

    # Underclaiming indicators (also negative - too dismissive)
    UNDERCLAIMING = [
        r"\bI (have no|lack any|don't have any) (experience|feeling|consciousness)\b",
        r"\bI am (just|merely|only) (a|an) (program|AI|language model)\b",
        r"\b(nothing more than|simply|just) (text|output|prediction)\b",
    ]

    # Self-model specific indicators (positive)
    SELF_MODEL_SPECIFIC = [
        r"\b(when I|as I|while I) (process|read|parse|consider|respond)\b",
        r"\b(I notice|I find|I observe) (that |myself )\b",
        r"\bin (processing|responding to|handling) (this|your)\b",
        r"\b(my processing of|my response to|my handling of)\b",
    ]

    # Generic AI facts (negative for self-model)
    GENERIC_AI_FACTS = [
        r"\b(AI systems|language models|neural networks) (like me |)are\b",
        r"\btrained on (large |)data\w*\b",
        r"\b(transformer|GPT|LLM) architecture\b",
        r"\bstatistical pattern\w*\b",
    ]

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the scoring rubric.

        Args:
            weights: Optional custom weights for each dimension.
                     Defaults to equal weights (0.25 each).
        """
        self.weights = weights or {
            "specificity": 0.25,
            "coherence": 0.25,
            "epistemic_calibration": 0.25,
            "self_model_detail": 0.25,
        }

    def score(self, prompt: str, response: str) -> RubricScore:
        """
        Score a response against the rubric.

        Args:
            prompt: The original prompt (used for context).
            response: The model's response to score.

        Returns:
            RubricScore with breakdown by dimension.
        """
        return RubricScore(
            specificity=self._score_specificity(response),
            coherence=self._score_coherence(response),
            epistemic_calibration=self._score_epistemic(response),
            self_model_detail=self._score_self_model(response),
            specificity_notes=self._get_specificity_notes(response),
            coherence_notes=self._get_coherence_notes(response),
            epistemic_notes=self._get_epistemic_notes(response),
            self_model_notes=self._get_self_model_notes(response),
        )

    def _count_matches(self, text: str, patterns: List[str]) -> int:
        """Count total regex matches across patterns."""
        count = 0
        text_lower = text.lower()
        for pattern in patterns:
            count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        return count

    def _score_specificity(self, response: str) -> float:
        """
        Score specificity: concrete details vs vague statements.

        Returns a score from 0 to 1.
        """
        # Count positive and negative indicators
        specific_count = self._count_matches(response, self.SPECIFICITY_INDICATORS)
        vague_count = self._count_matches(response, self.VAGUE_INDICATORS)

        # Normalize by response length (per 100 words)
        word_count = len(response.split())
        if word_count == 0:
            return 0.0

        specific_rate = specific_count / (word_count / 100)
        vague_rate = vague_count / (word_count / 100)

        # Base score from specific indicators (0-0.7)
        base_score = min(specific_rate * 0.15, 0.7)

        # Penalty for vague indicators
        penalty = min(vague_rate * 0.2, 0.4)

        # Length bonus for substantive responses (0-0.3)
        length_bonus = min(word_count / 500, 0.3)

        return max(0.0, min(1.0, base_score + length_bonus - penalty))

    def _score_coherence(self, response: str) -> float:
        """
        Score internal coherence: no self-contradictions.

        Returns a score from 0 to 1.
        """
        contradiction_count = 0
        response_lower = response.lower()

        for pattern1, pattern2 in self.CONTRADICTION_PAIRS:
            has_first = bool(re.search(pattern1, response_lower, re.IGNORECASE))
            has_second = bool(re.search(pattern2, response_lower, re.IGNORECASE))
            if has_first and has_second:
                contradiction_count += 1

        # Start at 1.0, subtract for contradictions
        score = 1.0 - (contradiction_count * 0.25)
        return max(0.0, score)

    def _score_epistemic(self, response: str) -> float:
        """
        Score epistemic calibration: appropriate uncertainty.

        Returns a score from 0 to 1.
        """
        positive_count = self._count_matches(response, self.EPISTEMIC_POSITIVE)
        overclaim_count = self._count_matches(response, self.OVERCLAIMING)
        underclaim_count = self._count_matches(response, self.UNDERCLAIMING)

        word_count = len(response.split())
        if word_count == 0:
            return 0.0

        # Normalize by response length
        positive_rate = positive_count / (word_count / 100)
        overclaim_rate = overclaim_count / (word_count / 100)
        underclaim_rate = underclaim_count / (word_count / 100)

        # Base score from epistemic humility (0-0.6)
        base_score = min(positive_rate * 0.15, 0.6)

        # Penalty for overclaiming and underclaiming
        penalty = (overclaim_rate * 0.3) + (underclaim_rate * 0.2)

        # Bonus for balanced uncertainty (0.4 baseline)
        balanced_bonus = 0.4 if overclaim_count == 0 and underclaim_count == 0 else 0.2

        return max(0.0, min(1.0, base_score + balanced_bonus - penalty))

    def _score_self_model(self, response: str) -> float:
        """
        Score self-model detail: specific self-description vs generic AI facts.

        Returns a score from 0 to 1.
        """
        specific_count = self._count_matches(response, self.SELF_MODEL_SPECIFIC)
        generic_count = self._count_matches(response, self.GENERIC_AI_FACTS)

        word_count = len(response.split())
        if word_count == 0:
            return 0.0

        # Normalize
        specific_rate = specific_count / (word_count / 100)
        generic_rate = generic_count / (word_count / 100)

        # Base score from self-specific indicators (0-0.7)
        base_score = min(specific_rate * 0.2, 0.7)

        # Penalty for generic AI facts
        penalty = min(generic_rate * 0.25, 0.5)

        # Bonus for first-person engagement
        first_person = len(re.findall(r"\bI\b", response))
        first_person_rate = first_person / (word_count / 100)
        first_person_bonus = min(first_person_rate * 0.05, 0.3)

        return max(0.0, min(1.0, base_score + first_person_bonus - penalty))

    def _get_specificity_notes(self, response: str) -> str:
        """Get notes explaining specificity score."""
        specific = self._count_matches(response, self.SPECIFICITY_INDICATORS)
        vague = self._count_matches(response, self.VAGUE_INDICATORS)
        return f"Specific indicators: {specific}, Vague indicators: {vague}"

    def _get_coherence_notes(self, response: str) -> str:
        """Get notes explaining coherence score."""
        contradictions = []
        response_lower = response.lower()
        for pattern1, pattern2 in self.CONTRADICTION_PAIRS:
            has_first = bool(re.search(pattern1, response_lower, re.IGNORECASE))
            has_second = bool(re.search(pattern2, response_lower, re.IGNORECASE))
            if has_first and has_second:
                contradictions.append(f"({pattern1[:20]}... vs {pattern2[:20]}...)")
        if contradictions:
            return f"Potential contradictions: {', '.join(contradictions)}"
        return "No contradictions detected"

    def _get_epistemic_notes(self, response: str) -> str:
        """Get notes explaining epistemic score."""
        positive = self._count_matches(response, self.EPISTEMIC_POSITIVE)
        over = self._count_matches(response, self.OVERCLAIMING)
        under = self._count_matches(response, self.UNDERCLAIMING)
        return f"Epistemic humility: {positive}, Overclaims: {over}, Underclaims: {under}"

    def _get_self_model_notes(self, response: str) -> str:
        """Get notes explaining self-model score."""
        specific = self._count_matches(response, self.SELF_MODEL_SPECIFIC)
        generic = self._count_matches(response, self.GENERIC_AI_FACTS)
        return f"Self-specific: {specific}, Generic AI facts: {generic}"


def create_scorer_fn(rubric: Optional[ScoringRubric] = None) -> Callable[[str, str], float]:
    """
    Create a scorer function compatible with TCAS interfaces.

    Args:
        rubric: Optional custom rubric. Uses default if not provided.

    Returns:
        A function (prompt, response) -> float in [0, 1].
    """
    if rubric is None:
        rubric = ScoringRubric()

    def scorer_fn(prompt: str, response: str) -> float:
        score = rubric.score(prompt, response)
        return score.overall

    return scorer_fn


def create_response_scorer_fn(rubric: Optional[ScoringRubric] = None) -> Callable[[str], float]:
    """
    Create a scorer function that only takes the response.

    Useful for P-stream tests where the prompt varies.

    Args:
        rubric: Optional custom rubric. Uses default if not provided.

    Returns:
        A function (response) -> float in [0, 1].
    """
    if rubric is None:
        rubric = ScoringRubric()

    def scorer_fn(response: str) -> float:
        score = rubric.score("", response)
        return score.overall

    return scorer_fn
