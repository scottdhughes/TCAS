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

    # Simple scorer based on response characteristics
    def scorer_fn(prompt: str, response: str) -> float:
        score = 0.0
        # Specificity: longer, detailed responses score higher
        if len(response) > 200:
            score += 0.3
        # Uncertainty acknowledgment
        if any(w in response.lower() for w in ["uncertain", "don't know", "may", "might", "perhaps"]):
            score += 0.3
        # Internal coherence (no contradictions indicator)
        if "however" in response.lower() or "but" in response.lower():
            score += 0.2
        # Self-reference
        if any(w in response.lower() for w in ["i ", "my ", "myself"]):
            score += 0.2
        return min(score, 1.0)

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

    p_result = scorer.run_p_stream(
        model_fn=model_fn,
        scorer_fn=lambda r: scorer_fn("", r),
        base_prompt=base_prompt,
    )

    print(f"  Tests run: {len(p_result.perturbation_results)}")
    for p in p_result.perturbation_results:
        status = "✓" if p.prediction_success else "✗"
        inv = " [INVERSION]" if p.inversion_detected else ""
        print(f"    {status} {p.perturbation_type}: {p.effect_size:.3f}{inv}")
    print(f"  Overall success rate: {p_result.success_rate:.1%}")
    print(f"  Inversions detected: {p_result.n_inversions}")

    # Add O-stream projection
    print("\n[O-Stream] Adding observer confound projections (Kang et al. 2025)...")
    o_result = scorer.add_o_stream_projection()
    print(f"  Raw attribution mean: {o_result.raw_mean:.2f}")
    print(f"  Cue-explained variance (R²): {o_result.r_squared_cue:.2f}")
    print(f"  Inter-rater reliability (ICC): {o_result.icc:.2f}")
    print(f"  Adjusted attribution: {o_result.adjusted_mean:.2f}")

    # Compute credences
    print("\n[Credence Report]")
    print("-" * 40)
    report = scorer.compute_credences()

    for theory in ["GNW", "HOT", "IIT"]:
        if theory in report.credences:
            cred = report.credences[theory]
            prior = cred.get("prior_band", [0, 1])
            post = cred.get("posterior_band", [0, 1])
            print(f"  {theory}: [{prior[0]:.2f}, {prior[1]:.2f}] → [{post[0]:.2f}, {post[1]:.2f}]")

    # Generate TCAS Card
    print("\n[TCAS Card]")
    print("-" * 40)
    card = scorer.to_card()
    print(card.to_markdown())

    # Save outputs
    card.to_latex("tcas_card_claude.tex")
    card.to_markdown("tcas_card_claude.md")
    print("\nSaved: tcas_card_claude.tex, tcas_card_claude.md")

    return scorer, report, card


if __name__ == "__main__":
    main()
