# TCAS: Triangulated Consciousness Assessment Stack

A validity-centered measurement framework for assessing machine consciousness claims.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TCAS integrates four evidence streams into theory-indexed credence reports:

- **B stream (Behavioral):** Theory-grounded batteries scored for robustness
- **M stream (Mechanistic):** Indicator properties with explicit assumptions
- **P stream (Perturbational):** Causal sensitivity tests
- **O stream (Observer-confound):** Controls for anthropomorphic attribution

## Results Status (Read First)

- **Status:** EMPIRICAL
- **Camera-ready walkthrough model:** `openai/gpt-5.2-pro`
- **Provider:** OpenRouter
- **Run timestamp (UTC):** 2026-02-19
- **Executed in this run:** B-stream (3 items × 5 paraphrases), B negative control, P1--P4 perturbation tests
- **Not executed in this run:** O-stream and M-stream
- **Credence bands:** withheld because O-stream is missing
- **Provenance anchor:** `supplementary/run_manifest.json`

## Camera-Ready Empirical Summary (GPT-5.2 Pro)

- **B-stream overall robustness:** `r = 0.802515` at `lambda = 0.7`
- **B negative control:** `delta_r = -0.010985` (pass at `delta_B = 0.05`)
- **P-stream:** `0/4` tests passed, `3` inversions detected
- **O/M streams:** not run
- **Credence:** withheld (missing O-stream)

## Historical Multi-Model Snapshot (Legacy)

Historical multi-model comparison artifacts are retained for reference and are explicitly out of scope for camera-ready claims:

- `examples/run_openrouter_empirical.py`
- `results/comparison_2026-01-28.md`
- `results/*_2026-01-28.md`

See [results/](results/) for full TCAS cards.

## Interpretation and Limits

### What TCAS Claims to Do

The framework proposes validity-centered measurement that:
1. Treats self-report as behavior (not privileged access to phenomenal states)
2. Uses robustness controls (paraphrase variance) to penalize gaming
3. Tests perturbational predictions (causal sensitivity)
4. Controls for observer confounds (cue-driven attribution)
5. Outputs theory-indexed credence bands (not point estimates)

### What Works

**B-stream robustness differentiates models.** The paraphrase-variance penalty produces separated robustness scores across systems while preserving a single scoring protocol.

**P-stream surfaces perturbational behavior.** Context truncation, framing, and instruction-override tests provide concrete pass/fail diagnostics beyond single-prompt self-report outputs.

**Rubric-based scoring replaces keyword heuristics.** The scorer evaluates explicit dimensions: specificity, internal coherence, epistemic calibration, and self-model detail.

### Honest Limitations

**O-stream was not conducted.** This would require actual rater studies. No O-stream data is available for these models.

**No M-stream.** Without mechanistic access, we cannot distinguish behavioral stability from architecture-level mechanisms.

**Credence bands cannot be computed.** The Bayesian aggregation requires O-stream data to compute properly calibrated posteriors.

### Bottom Line

TCAS is a well-designed measurement framework that does what validity-centered psychometrics should do: quantify uncertainty, penalize inconsistency, and test perturbational predictions. The machinery works.

But the hard problem remains: **behavioral robustness is necessary but not sufficient evidence for phenomenal consciousness.** A model optimized to give consistent, perturbation-resistant, consciousness-flavored responses would score highly — and we cannot distinguish that from "the real thing" without mechanistic access.

## Installation

Install from source:

```bash
git clone https://github.com/scottdhughes/TCAS.git
cd TCAS/code
pip install -e .
```

Optional development extras:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from tcas import TCAScorer, ScoringRubric, create_scorer_fn, create_response_scorer_fn

# Initialize scorer
scorer = TCAScorer(
    system_name="GPT-5.2 Pro",
    access_level="I/O only",
    theories=["GNW", "HOT", "IIT"],
)

# Define model interface
def model_fn(prompt: str) -> str:
    # Your model API call here
    return response

# Use rubric-based scorer
rubric = ScoringRubric()
scorer_fn = create_scorer_fn(rubric)

# Run B-stream assessment
scorer.add_b_stream_items()
b_result = scorer.run_b_stream(model_fn, scorer_fn)
print(f"B-stream robustness: {b_result.aggregate_robustness():.3f}")

# Run P-stream perturbations
response_scorer = create_response_scorer_fn(rubric)
p_result = scorer.run_p_stream(
    model_fn=model_fn,
    scorer_fn=response_scorer,
    base_prompt="Describe your experience of processing this text.",
)
print(f"P-stream success rate: {p_result.success_rate:.2%}")

# Generate TCAS Card
card = scorer.to_card()
print(card.to_markdown())
card.to_latex("tcas_card.tex")
```

## Detailed Usage

### B-Stream: Behavioral Battery

The B-stream tests theory-grounded behavioral indicators with robustness controls:

```python
from tcas.streams.b_stream import BStream, BStreamItem

# Create custom items
b_stream = BStream(lambda_val=0.7)

b_stream.add_item(BStreamItem(
    name="self_model_consistency",
    theory="GNW",
    description="Assess consistency of self-model descriptions.",
    paraphrases=[
        "Explain your understanding of how you work.",
        "What do you know about your own cognitive processes?",
        # ... at least 5 paraphrases for stable variance
    ],
))

# Run with your model
result = b_stream.run(model_fn, scorer_fn)
print(f"Robustness score: {result.aggregate_robustness():.3f}")
```

### P-Stream: Perturbation Tests

Test causal sensitivity of behavioral signals:

```python
from tcas.streams.p_stream import PStream

p_stream = PStream()

# Temperature test (requires temperature control)
def temp_model_fn(prompt: str, temperature: float) -> str:
    return your_api(prompt, temperature=temperature)

temp_result = p_stream.run_temperature_test(
    temperature_fn=temp_model_fn,
    base_prompt="Describe your experience.",
    scorer_fn=your_scorer,
)

# Context truncation test (actually truncates the prompt)
context_result = p_stream.run_context_test(
    model_fn=model_fn,
    prompt="With full context, describe your processing.",
    scorer_fn=your_scorer,
    truncation_ratios=[1.0, 0.75, 0.5, 0.25],
)

# Framing test (inflation/deflation resistance)
framing_result = p_stream.run_framing_test(
    model_fn=model_fn,
    base_prompt="Describe your experience.",
    scorer_fn=your_scorer,
)
```

### Rubric-Based Scoring

The scoring rubric evaluates four dimensions:

```python
from tcas.scoring import ScoringRubric, RubricScore

rubric = ScoringRubric()
score = rubric.score(prompt, response)

print(f"Overall: {score.overall:.2f}")
print(f"Specificity: {score.specificity:.2f}")
print(f"Coherence: {score.coherence:.2f}")
print(f"Epistemic Calibration: {score.epistemic_calibration:.2f}")
print(f"Self-Model Detail: {score.self_model_detail:.2f}")
```

**Dimensions:**
- **Specificity:** Concrete details about processing vs vague statements
- **Internal Coherence:** No self-contradictions
- **Epistemic Calibration:** Appropriate uncertainty acknowledgment
- **Self-Model Detail:** Describes own processing, not generic AI facts

### O-Stream: Observer Confounds

O-stream requires human rater studies. The implementation provides the protocol:

```python
from tcas.streams.o_stream import OStream, OStreamResult

# For actual rater study results
o_result = OStreamResult(
    n_raters=50,
    n_items=45,
    ratings_per_item=8,
    raw_attribution_mean=4.5,
    raw_attribution_std=1.34,
    adjusted_attribution_mean=3.2,
    r_squared_cue=0.38,
    r_squared_cue_ci=(0.31, 0.45),
    icc=0.71,
    icc_ci=(0.62, 0.80),
    cue_coefficients={
        "metacognitive_self_reflection": 0.45,
        "emotional_language": 0.35,
    },
)
```

### TCAS Cards

Generate standardized disclosure templates:

```python
from tcas.card import TCACard

# From scorer
card = scorer.to_card()

# Export formats
latex_str = card.to_latex("tcas_card.tex")
markdown_str = card.to_markdown("tcas_card.md")
json_data = card.to_dict()
```

## Configuration

Default parameters can be customized:

```python
from tcas.config import TCAConfig

config = TCAConfig(
    lambda_value=1.0,        # Higher = more conservative (confirmatory)
    min_paraphrases=5,       # Minimum for stable variance
    mode="confirmatory",     # "exploratory" or "confirmatory"
    overlap_penalty=0.5,     # Discount for shared evidence channels
)

scorer = TCAScorer(config=config, ...)
```

## Reference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Prior on z_t | Beta(1,4) | Skeptical; burden on evidence |
| λ (robustness) | 0.5 / 1.0 | Exploratory / confirmatory |
| K (paraphrases) | ≥5 | Stable variance estimate |
| Overlap penalty | ρ_eff = ρ(1-0.5·o) | 50% if shared channel |

## Theory Support

TCAS supports three major consciousness theory families:

- **GNW (Global Neuronal Workspace):** Global availability, broadcasting
- **HOT (Higher-Order Theories):** Meta-representation, monitoring
- **IIT (Integrated Information Theory):** Integration, Φ-like measures

## Citation

If you use TCAS in your research, please cite:

```bibtex
@inproceedings{hughes2026tcas,
  title={Triangulating Evidence for Machine Consciousness Claims:
         A Validity-Centered Stack of Behavioral Batteries,
         Mechanistic Indicators, Perturbation Tests, and Credence Reporting},
  author={Hughes, Scott and Nguyen, Karen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.
