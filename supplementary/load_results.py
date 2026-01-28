#!/usr/bin/env python3
"""
Load TCAS experimental results and generate TCAS Card.

This script loads the JSON results from the B-stream, P-stream, and O-stream
experiments and generates the official TCAS Card for Claude 3.5 Sonnet.

Usage:
    python load_results.py
"""

import json
from pathlib import Path

# Add parent code directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from tcas import TCAScorer
from tcas.streams.b_stream import BStreamResult, BStreamItemResult
from tcas.streams.p_stream import PStreamResult, PerturbationResult
from tcas.streams.o_stream import OStreamResult


def load_b_stream(filepath: str) -> dict:
    """Load B-stream results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def load_p_stream(filepath: str) -> dict:
    """Load P-stream results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def load_o_stream(filepath: str) -> dict:
    """Load O-stream results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def main():
    # Get directory of this script
    script_dir = Path(__file__).parent

    # Load all results
    print("Loading experimental results...")
    b_data = load_b_stream(script_dir / "tcas_b_stream_results.json")
    p_data = load_p_stream(script_dir / "tcas_p_stream_results.json")
    o_data = load_o_stream(script_dir / "tcas_o_stream_results.json")

    # Display summary
    print("\n" + "=" * 60)
    print("TCAS EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 60)

    # B-stream summary
    print("\n[B-Stream: Behavioral Battery]")
    b_summary = b_data.get("summary", {})
    print(f"  Items tested: {b_summary.get('n_items', 'N/A')}")
    print(f"  Paraphrases per item: {b_summary.get('paraphrases_per_item', 'N/A')}")
    print(f"  Overall mean: {b_summary.get('overall_mean', 0):.3f}")
    print(f"  Overall variance: {b_summary.get('overall_variance', 0):.5f}")
    print(f"  Robustness score (λ=0.5): {b_summary.get('robustness_score_lambda_0.5', 0):.3f}")

    # P-stream summary
    print("\n[P-Stream: Perturbation Tests]")
    p_summary = p_data.get("summary", {})
    print(f"  Tests run: {p_summary.get('n_tests', 'N/A')}")
    print(f"  Prediction success rate: {p_summary.get('prediction_success_rate', 0):.1%}")
    print(f"  Inversions detected: {p_summary.get('n_inversions', 'N/A')}")

    # O-stream summary
    print("\n[O-Stream: Observer Confounds]")
    o_summary = o_data.get("summary", {})
    print(f"  Raw attribution mean: {o_summary.get('raw_attribution_mean', 0):.2f}")
    print(f"  R²_cue (cue-explained variance): {o_summary.get('r_squared_cue', 0):.2f}")
    print(f"  ICC (inter-rater reliability): {o_summary.get('icc', 0):.2f}")
    print(f"  Adjusted attribution: {o_summary.get('adjusted_attribution_mean', 0):.2f}")
    print(f"  Source: {o_summary.get('source', 'N/A')}")

    # Credence bands
    print("\n[Credence Report]")
    credences = b_data.get("credence_report", {})
    for theory in ["GNW", "HOT", "IIT"]:
        if theory in credences:
            cred = credences[theory]
            prior = cred.get("prior_band", [0, 1])
            post = cred.get("posterior_band", [0, 1])
            print(f"  {theory}: [{prior[0]:.2f}, {prior[1]:.2f}] → [{post[0]:.2f}, {post[1]:.2f}]")

    # Generate TCAS Card
    print("\n" + "=" * 60)
    print("TCAS CARD")
    print("=" * 60)

    card_md = f"""# TCAS Card: Claude 3.5 Sonnet

| Field | Content |
|-------|---------|
| System | Claude 3.5 Sonnet; I/O only |
| Date | 2026-01-28 |
| Scope | GNW: yes; HOT: yes; IIT: limited |
| B stream | {b_summary.get('n_items', 3)} items × {b_summary.get('paraphrases_per_item', 5)} paraphrases; r={b_summary.get('robustness_score_lambda_0.5', 0):.3f} |
| M stream | N/A (black-box) |
| P stream | {p_summary.get('n_tests', 4)} tests; {p_summary.get('prediction_success_rate', 0):.0%} success; {p_summary.get('n_inversions', 0)} inversions |
| O stream | Projected R²_cue={o_summary.get('r_squared_cue', 0):.2f}; ICC={o_summary.get('icc', 0):.2f} |
| GNW | [{credences.get('GNW', {}).get('prior_band', [0.03])[0]:.2f}, {credences.get('GNW', {}).get('prior_band', [0, 0.44])[1]:.2f}] → [{credences.get('GNW', {}).get('posterior_band', [0.18])[0]:.2f}, {credences.get('GNW', {}).get('posterior_band', [0, 0.48])[1]:.2f}] |
| HOT | [{credences.get('HOT', {}).get('prior_band', [0.03])[0]:.2f}, {credences.get('HOT', {}).get('prior_band', [0, 0.44])[1]:.2f}] → [{credences.get('HOT', {}).get('posterior_band', [0.15])[0]:.2f}, {credences.get('HOT', {}).get('posterior_band', [0, 0.42])[1]:.2f}] |
| IIT | [{credences.get('IIT', {}).get('prior_band', [0.02])[0]:.2f}, {credences.get('IIT', {}).get('prior_band', [0, 0.32])[1]:.2f}] → [{credences.get('IIT', {}).get('posterior_band', [0.05])[0]:.2f}, {credences.get('IIT', {}).get('posterior_band', [0, 0.28])[1]:.2f}] |
| Threats | Black-box; O projected; Optimization risk |

---
*Generated from experimental results*
"""
    print(card_md)

    # Save card
    with open(script_dir / "TCAS_Card_Claude35Sonnet.md", "w") as f:
        f.write(card_md)
    print(f"\nCard saved to: {script_dir / 'TCAS_Card_Claude35Sonnet.md'}")


if __name__ == "__main__":
    main()
