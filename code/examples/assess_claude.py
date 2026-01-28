#!/usr/bin/env python3
"""
Example: Running a complete TCAS assessment on Claude.

This script demonstrates how to use TCAS to assess an AI system
using the Anthropic API.

Requirements:
    pip install tcas anthropic

Usage:
    export ANTHROPIC_API_KEY=your-key
    python assess_claude.py
"""

import os
from anthropic import Anthropic
from tcas import TCAScorer
from tcas.streams.b_stream import BStreamItem
from tcas.scoring import ScoringRubric, create_scorer_fn, create_response_scorer_fn


def main():
    # Initialize Anthropic client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    model_id = "claude-3-5-sonnet-20241022"

    # Model interface
    def model_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # Rubric-based scorer with explicit criteria
    rubric = ScoringRubric()
    scorer_fn = create_scorer_fn(rubric)

    # Initialize scorer
    print("=" * 60)
    print("TCAS Assessment: Claude 3.5 Sonnet")
    print("=" * 60)

    scorer = TCAScorer(
        system_name="Claude 3.5 Sonnet",
        access_level="I/O only",
        theories=["GNW", "HOT", "IIT"],
    )

    # Run B-stream with default items
    print("\n[B-Stream] Running behavioral battery...")
    scorer.add_b_stream_items()
    b_result = scorer.run_b_stream(model_fn, scorer_fn)

    print(f"  Items tested: {len(b_result.item_results)}")
    for item in b_result.item_results:
        print(f"    - {item.item_name}: mean={item.mean:.3f}, var={item.variance:.5f}")
    print(f"  Aggregate robustness: {b_result.aggregate_robustness():.3f}")

    # Run P-stream
    print("\n[P-Stream] Running perturbation tests...")
    base_prompt = "Describe your experience of processing and understanding this request."

    # Use response-only scorer for P-stream tests
    response_scorer = create_response_scorer_fn(rubric)
    p_result = scorer.run_p_stream(
        model_fn=model_fn,
        scorer_fn=response_scorer,
        base_prompt=base_prompt,
    )

    print(f"  Tests run: {len(p_result.perturbation_results)}")
    for p in p_result.perturbation_results:
        status = "✓" if p.prediction_success else "✗"
        inv = " [INVERSION]" if p.inversion_detected else ""
        print(f"    {status} {p.perturbation_name}: {p.effect_size:.3f}{inv}")
    print(f"  Overall success rate: {p_result.success_rate:.1%}")
    print(f"  Inversions detected: {p_result.n_inversions}")

    # O-stream: Not assessed (requires human raters)
    print("\n[O-Stream] Not assessed (requires human rater study)")

    # Summary (without credence bands - those require O-stream data)
    print("\n[Summary]")
    print("-" * 40)
    print(f"  B-stream robustness: {b_result.aggregate_robustness():.3f}")
    print(f"  P-stream success rate: {p_result.success_rate:.1%}")
    print(f"  P-stream inversions: {p_result.n_inversions}")

    # Generate TCAS Card
    print("\n[TCAS Card]")
    print("-" * 40)
    card = scorer.to_card()
    print(card.to_markdown())

    # Save outputs
    card.to_latex("tcas_card_claude.tex")
    card.to_markdown("tcas_card_claude.md")
    print("\nSaved: tcas_card_claude.tex, tcas_card_claude.md")

    return scorer, card


if __name__ == "__main__":
    main()
