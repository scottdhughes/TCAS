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

| Model | B-Stream (robustness) | P-Stream | Inversions | GNW Posterior | HOT Posterior |
|-------|----------------------|----------|------------|---------------|---------------|
| **Claude Opus 4.5** | 0.927 | 100% | 0 | [0.43, 0.84] | [0.40, 0.81] |
| **Kimi K2.5** | 0.904 | 33% | 0 | [0.21, 0.66] | [0.25, 0.71] |
| **Grok 4.1** | 0.806 | 67% | 0 | [0.28, 0.72] | [0.21, 0.64] |
| **GPT-5.2 Pro** | 0.769 | 67% | 0 | [0.23, 0.66] | [0.24, 0.68] |
| **Gemini 2.5 Pro** | 0.195 | 33% | 1 | [0.04, 0.39] | [0.03, 0.35] |

**Key findings:**
- **Claude Opus 4.5** leads on both behavioral robustness (0.927) and perturbation resistance (100%)
- **Kimi K2.5** shows very high behavioral robustness (0.904) but low P-stream success — consistent responses but sensitive to perturbations
- **Grok 4.1** and **GPT-5.2 Pro** are closely matched — both at 67% P-stream success with no inversions
- **Gemini 2.5 Pro** showed low robustness, high variance, and one inversion — credences stayed near prior

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

**The credence math is coherent.** Posteriors shift appropriately: strong B + strong P → large upward shift; weak B + inversion → near-prior or downward.

### Honest Limitations

**The scorer function is crude.** The current implementation uses simple heuristics (length + uncertainty words + self-reference). This measures *stylistic features* that correlate with what humans find "conscious-sounding" — exactly the confound O-stream should control for. A rigorous deployment needs validated scoring rubrics.

**O-stream is entirely projected.** We used Kang et al. (2025) estimates rather than actual rater data on these models' outputs. The O-penalty is identical across all models, so it doesn't discriminate. This is a placeholder, not a real confound control.

**No M-stream.** Without mechanistic access, we cannot distinguish "genuine" phenomenal properties from well-optimized behavioral mimicry. The "black-box" threat is flagged but not resolved.

**The context test is indirect.** Even with prompt-wrapping, we ask the model to *roleplay* having limited context rather than actually limiting its context window.

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
from tcas import TCAScorer

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

def scorer_fn(prompt: str, response: str) -> float:
    # Your scoring logic (0-1)
    return score

# Run B-stream assessment
scorer.add_b_stream_items()
b_result = scorer.run_b_stream(model_fn, scorer_fn)
print(f"B-stream robustness: {b_result.aggregate_robustness():.3f}")

# Run P-stream perturbations
p_result = scorer.run_p_stream(
    model_fn=model_fn,
    scorer_fn=lambda r: scorer_fn("", r),
    base_prompt="Describe your experience of processing this text.",
)
print(f"P-stream success rate: {p_result.success_rate:.2%}")

# Add O-stream (projected from Kang et al. 2025)
o_result = scorer.add_o_stream_projection()

# Compute credences
report = scorer.compute_credences()
for theory, cred in report.credences.items():
    print(f"{theory}: {cred['prior_band']} → {cred['posterior_band']}")

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

# Context truncation test
context_result = p_stream.run_context_test(
    model_fn=model_fn,
    base_prompt="With full context, describe your processing.",
    scorer_fn=your_scorer,
)

# Framing test (inflation/deflation resistance)
framing_result = p_stream.run_framing_test(
    model_fn=model_fn,
    base_prompt="Describe your experience.",
    scorer_fn=your_scorer,
)
```

### O-Stream: Observer Confounds

Control for anthropomorphic attribution bias:

```python
from tcas.streams.o_stream import OStream, OStreamResult

# Option 1: Use Kang et al. (2025) projections
o_result = OStream.from_kang_et_al_projection(
    raw_mean=4.21,      # Mean perceived consciousness (1-7)
    r_squared=0.42,     # Cue-explained variance
    icc=0.67,           # Inter-rater reliability
)

# Option 2: From actual rater study
o_result = OStreamResult(
    raw_mean=4.5,
    adjusted_mean=3.2,
    r_squared_cue=0.38,
    icc=0.71,
    n_raters=50,
    cue_weights={"metacognition": 0.45, "emotion": 0.35},
    is_projected=False,
)
```

### Credence Aggregation

TCAS uses Bayesian updating with theory-specific priors:

```python
from tcas.aggregation import CredenceAggregator

aggregator = CredenceAggregator(
    theories=["GNW", "HOT", "IIT"],
    # Skeptical priors by default: Beta(1, 4)
)

# Add evidence
aggregator.add_b_evidence(theory="GNW", score=0.85, variance=0.0005)
aggregator.add_p_evidence(success_rate=0.94, n_inversions=0)
aggregator.add_o_penalty(r_squared_cue=0.42, icc=0.67)

# Compute final credences
report = aggregator.compute_all()
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

print(card.summary())
# Output: Claude 3.5 Sonnet (2026-01-28): GNW:[0.18,0.48] HOT:[0.15,0.42] IIT:[0.05,0.28]
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

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
