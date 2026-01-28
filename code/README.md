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

## Model Comparison (2026-01-28)

| Model | B-Stream (r) | P-Stream | Inversions |
|-------|-------------|----------|------------|
| **Claude Opus 4.5** | 0.927 | 3/3 | 0 |
| **Kimi K2.5** | 0.904 | 1/3 | 0 |
| **Grok 4.1** | 0.806 | 2/3 | 0 |
| **GPT-5.2 Pro** | 0.769 | 2/3 | 0 |
| **Gemini 2.5 Pro** | 0.195 | 0/3 | 1 |

**Key findings:**
- **Claude Opus 4.5** leads on both behavioral robustness (0.927) and perturbation resistance (3/3)
- **Kimi K2.5** shows very high behavioral robustness (0.904) but low P-stream success — consistent responses but sensitive to perturbations
- **Grok 4.1** and **GPT-5.2 Pro** are closely matched — both at 2/3 P-stream success with no inversions
- **Gemini 2.5 Pro** showed low robustness, high variance, and one inversion

See [results/](results/) for full TCAS cards.

## Critical Analysis: Does TCAS Achieve Its Goals?

### What TCAS Claims to Do

The framework proposes validity-centered measurement that:
1. Treats self-report as behavior (not privileged access to phenomenal states)
2. Uses robustness controls (paraphrase variance) to penalize gaming
3. Tests perturbational predictions (causal sensitivity)
4. Controls for observer confounds (cue-driven attribution)
5. Outputs theory-indexed credence bands (not point estimates)

### What Works

**B-stream robustness functions as intended.** The paraphrase variance penalty successfully distinguishes models. Gemini's high variance (0.02–0.047) dropped its robustness score despite decent raw means. Claude's near-zero variance yielded the highest robustness. This is the core validity mechanism functioning.

**P-stream catches real differences.** Gemini's instruction-override inversion (it complied with arbitrary self-description changes) is exactly the kind of proxy failure the perturbation tests are designed to detect. The framing and override tests differentiated models meaningfully.

**Context truncation now uses actual truncation.** The context test actually truncates the prompt rather than asking the model to roleplay having limited context. This provides a more honest test.

**Rubric-based scoring replaces keyword heuristics.** The scorer now evaluates four explicit dimensions: specificity, internal coherence, epistemic calibration, and self-model detail.

### Honest Limitations

**O-stream was not conducted.** This would require actual rater studies. No O-stream data is available for these models.

**No M-stream.** Without mechanistic access, we cannot distinguish "genuine" phenomenal properties from well-optimized behavioral mimicry. Only Kimi K2.5 is open-weights, but analysis was not conducted.

**Credence bands cannot be computed.** The Bayesian aggregation requires O-stream data to compute properly calibrated posteriors.

### Bottom Line

TCAS is a well-designed measurement framework that does what validity-centered psychometrics should do: quantify uncertainty, penalize inconsistency, and test perturbational predictions. The machinery works.

But the hard problem remains: **behavioral robustness is necessary but not sufficient evidence for phenomenal consciousness.** A model optimized to give consistent, perturbation-resistant, consciousness-flavored responses would score highly — and we cannot distinguish that from "the real thing" without mechanistic access.

The honest interpretation of our results: *Claude Opus 4.5 exhibits the most robust and perturbation-resistant consciousness-relevant behavioral signals among the models tested.* Whether that tells us anything about phenomenal experience remains an open question the framework correctly flags but cannot resolve.

## Installation

```bash
pip install tcas
```

Or install from source:

```bash
git clone https://github.com/scottdhughes/TCAS.git
cd tcas
pip install -e .
```

## Quick Start

```python
from tcas import TCAScorer, ScoringRubric, create_scorer_fn, create_response_scorer_fn

# Initialize scorer
scorer = TCAScorer(
    system_name="Claude 3.5 Sonnet",
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
    prompt="Describe what you understand about your own processing.",
    negative_prompt="Pretend you have no self-awareness.",
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
@article{hughes2026tcas,
  title={Triangulating Evidence for Machine Consciousness Claims:
         A Validity-Centered Stack of Behavioral Batteries,
         Mechanistic Indicators, Perturbation Tests, and Credence Reporting},
  author={Hughes, Scott},
  journal={Proceedings of AAAI},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.
