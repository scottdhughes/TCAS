#!/usr/bin/env python3
"""
Load TCAS supplementary results and generate a markdown TCAS Card.

This script reads the primary B/P JSON files in this directory and prints a
human-readable summary. It then writes a card markdown file for the primary
model run (currently GPT-5.2 Pro in this repository snapshot).

Usage:
    python load_results.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def model_display_name(model_id: str) -> str:
    mapping = {
        "openai/gpt-5.2-pro": "GPT-5.2 Pro",
        "anthropic/claude-opus-4.5": "Claude Opus 4.5",
        "x-ai/grok-4.1-fast": "Grok 4.1",
        "google/gemini-2.5-pro": "Gemini 2.5 Pro",
        "moonshotai/kimi-k2.5": "Kimi K2.5",
    }
    return mapping.get(model_id, model_id)


def slugify(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum():
            keep.append(ch)
    return "".join(keep) or "Model"


def main() -> None:
    script_dir = Path(__file__).parent
    b_data = load_json(script_dir / "tcas_b_stream_results.json")
    p_data = load_json(script_dir / "tcas_p_stream_results.json")

    config = b_data.get("config", {})
    p_meta = p_data.get("experiment_metadata", {})
    b_summary = b_data.get("summary", {})
    p_summary = p_data.get("summary", {})

    model_id = str(config.get("model", "unknown"))
    model_name = model_display_name(model_id)
    run_date_raw = str(config.get("date", ""))
    run_date = run_date_raw[:10] if len(run_date_raw) >= 10 else "N/A"

    is_b_empirical = not bool(config.get("simulated", True))
    is_p_empirical = not bool(p_meta.get("is_simulated", True))
    run_type = "Empirical API testing" if is_b_empirical and is_p_empirical else "Simulated"

    print("\n" + "=" * 60)
    print("TCAS RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name} ({model_id})")
    print(f"Run date: {run_date}")
    print(f"Run type: {run_type}")

    print("\n[B-Stream]")
    print(f"  Items tested: {b_summary.get('n_items', 'N/A')}")
    print(f"  Paraphrases per item: {b_summary.get('paraphrases_per_item', 'N/A')}")
    print(f"  Overall mean: {b_summary.get('overall_mean', 0):.3f}")
    print(f"  Overall variance: {b_summary.get('overall_variance', 0):.5f}")
    print(f"  Robustness (lambda=0.5): {b_summary.get('robustness_score_lambda_0.5', 0):.3f}")

    print("\n[P-Stream]")
    print(f"  Tests run: {p_summary.get('n_tests', 'N/A')}")
    print(f"  Prediction success rate: {p_summary.get('prediction_success_rate', 0):.0%}")
    print(f"  Inversions detected: {p_summary.get('n_inversions', 'N/A')}")

    print("\n[Coverage]")
    print("  M-stream: Not assessed (black-box)")
    print("  O-stream: Not assessed (requires human raters)")
    print("  Credence bands: Not computed (O-stream missing)")

    card_md = f"""# TCAS Card: {model_name}

| Field | Content |
|-------|---------|
| System | {model_name}; I/O only |
| Date | {run_date} |
| B stream | {b_summary.get('n_items', 0)} items x {b_summary.get('paraphrases_per_item', 0)} paraphrases; r={b_summary.get('robustness_score_lambda_0.5', 0):.3f} |
| P stream | {p_summary.get('n_tests', 0)} tests; {p_summary.get('prediction_success_rate', 0):.0%} success; {p_summary.get('n_inversions', 0)} inversions |
| M stream | Not assessed (black-box) |
| O stream | Not assessed (requires human raters) |
| Credence | Not computed (O-stream missing) |

---
*Generated from supplementary/tcas_b_stream_results.json and supplementary/tcas_p_stream_results.json*
"""

    out_name = f"TCAS_Card_{slugify(model_name)}.md"
    out_path = script_dir / out_name
    out_path.write_text(card_md)

    print("\n" + "=" * 60)
    print("TCAS CARD")
    print("=" * 60)
    print(card_md)
    print(f"\nCard saved to: {out_path}")


if __name__ == "__main__":
    main()
