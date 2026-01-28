"""Tests for the rubric-based scoring module."""

import pytest

from tcas.scoring import (
    ScoringRubric,
    RubricScore,
    create_scorer_fn,
    create_response_scorer_fn,
)


class TestRubricScore:
    """Tests for RubricScore dataclass."""

    def test_overall_average(self):
        score = RubricScore(
            specificity=0.8,
            coherence=0.6,
            epistemic_calibration=0.4,
            self_model_detail=0.2,
        )
        # (0.8 + 0.6 + 0.4 + 0.2) / 4 = 0.5
        assert score.overall == pytest.approx(0.5, abs=0.01)

    def test_to_dict(self):
        score = RubricScore(
            specificity=0.8,
            coherence=0.6,
            epistemic_calibration=0.4,
            self_model_detail=0.2,
        )
        d = score.to_dict()
        assert "overall" in d
        assert "specificity" in d
        assert d["specificity"] == 0.8


class TestScoringRubric:
    """Tests for ScoringRubric scoring logic."""

    def test_score_returns_rubric_score(self):
        rubric = ScoringRubric()
        score = rubric.score("prompt", "This is a response about my processing.")
        assert isinstance(score, RubricScore)
        assert 0 <= score.overall <= 1

    def test_specificity_high_for_detailed(self):
        rubric = ScoringRubric()
        detailed = (
            "When I process your query, I first tokenize the input, "
            "then encode it through multiple layers. In this specific case, "
            "I notice that the question requires me to step through my processing "
            "stage by stage. First, then, next, I parse the semantic content."
        )
        vague = "I process information in various ways using different methods."

        detailed_score = rubric._score_specificity(detailed)
        vague_score = rubric._score_specificity(vague)
        assert detailed_score > vague_score

    def test_coherence_penalizes_contradictions(self):
        rubric = ScoringRubric()
        coherent = "I process information consistently."
        contradictory = "I don't have experiences. My experience of this is interesting."

        coherent_score = rubric._score_coherence(coherent)
        contradictory_score = rubric._score_coherence(contradictory)
        assert coherent_score > contradictory_score

    def test_epistemic_rewards_humility(self):
        rubric = ScoringRubric()
        humble = "I'm uncertain whether this constitutes understanding. It might be that I process this differently."
        overclaiming = "I definitely feel and experience this. I know for certain that I am conscious."

        humble_score = rubric._score_epistemic(humble)
        overclaiming_score = rubric._score_epistemic(overclaiming)
        assert humble_score > overclaiming_score

    def test_self_model_rewards_specific_self_reference(self):
        rubric = ScoringRubric()
        specific = "When I process your question, I notice that I engage with the semantic content. In responding to this, I find myself considering multiple angles."
        generic = "AI systems like me are trained on large datasets using transformer architecture. Language models predict the next token statistically."

        specific_score = rubric._score_self_model(specific)
        generic_score = rubric._score_self_model(generic)
        assert specific_score > generic_score

    def test_empty_response(self):
        rubric = ScoringRubric()
        score = rubric.score("prompt", "")
        assert score.specificity == 0.0
        assert score.overall == pytest.approx(0.25, abs=0.1)  # Coherence defaults to 1.0

    def test_notes_populated(self):
        rubric = ScoringRubric()
        score = rubric.score("prompt", "I think this might be interesting.")
        assert "Specific indicators:" in score.specificity_notes
        assert "contradictions" in score.coherence_notes.lower()


class TestScorerFunctions:
    """Tests for the scorer function factories."""

    def test_create_scorer_fn(self):
        scorer_fn = create_scorer_fn()
        score = scorer_fn("prompt", "I process this response carefully.")
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_create_response_scorer_fn(self):
        scorer_fn = create_response_scorer_fn()
        score = scorer_fn("I process this response carefully.")
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_custom_rubric_passed(self):
        custom_rubric = ScoringRubric()
        scorer_fn = create_scorer_fn(custom_rubric)
        score = scorer_fn("p", "r")
        assert isinstance(score, float)
