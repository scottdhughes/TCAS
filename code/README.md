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

## Installation

```bash
pip install tcas
```

Or install from source:

```bash
git clone https://github.com/yourusername/tcas.git
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
