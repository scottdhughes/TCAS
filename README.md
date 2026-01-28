# TCAS: Triangulated Consciousness Assessment Stack

**AAAI 2026 Submission Materials**

## Paper

> Hughes, S. (2026). Triangulating Evidence for Machine Consciousness Claims: A Validity-Centered Stack of Behavioral Batteries, Mechanistic Indicators, Perturbation Tests, and Credence Reporting.

**Main paper:** `paper/TCAS_Paper_AAAI.pdf`

## Repository Structure

```
TCAS_AAAI_Submission/
├── README.md                    # This file
├── paper/
│   └── TCAS_Paper_AAAI.pdf      # Main submission (8 pages)
├── supplementary/
│   ├── tcas_b_stream_results.json   # Behavioral battery results
│   ├── tcas_p_stream_results.json   # Perturbation test results
│   ├── tcas_o_stream_results.json   # Observer confound results
│   ├── tcas_o_stream_protocol.md    # O-stream rater study protocol
│   └── load_results.py              # Script to load and display results
└── code/
    ├── pyproject.toml           # Package configuration
    ├── README.md                # Package documentation
    ├── tcas/                    # Reference implementation
    │   ├── __init__.py
    │   ├── config.py            # Default parameters
    │   ├── scorer.py            # Main TCAScorer class
    │   ├── card.py              # TCAS Card generator
    │   ├── aggregation.py       # Bayesian credence aggregation
    │   └── streams/
    │       ├── b_stream.py      # Behavioral battery
    │       ├── p_stream.py      # Perturbation tests
    │       └── o_stream.py      # Observer confounds
    └── examples/
        └── assess_claude.py     # Example usage with Anthropic API
```

## Quick Start

### Install the reference implementation

```bash
cd code
pip install -e .
```

### Load experimental results

```bash
cd supplementary
python load_results.py
```

### Run your own assessment

```python
from tcas import TCAScorer

scorer = TCAScorer(
    system_name="Your Model",
    theories=["GNW", "HOT", "IIT"],
)

# Define your model interface
def model_fn(prompt: str) -> str:
    return your_api_call(prompt)

def scorer_fn(prompt: str, response: str) -> float:
    return your_scoring_logic(prompt, response)

# Run assessment
scorer.add_b_stream_items()
scorer.run_b_stream(model_fn, scorer_fn)
scorer.run_p_stream(model_fn, scorer_fn, base_prompt="...")
scorer.add_o_stream_projection()

# Generate report
report = scorer.compute_credences()
card = scorer.to_card()
card.to_markdown("tcas_card.md")
```

## Experimental Results Summary

| Stream | Key Metrics |
|--------|-------------|
| **B-stream** | 3 items × 5 paraphrases; r = 0.847 (λ=0.5) |
| **P-stream** | 4 test types; 93.75% prediction success; 0 inversions |
| **O-stream** | R²_cue = 0.42; ICC = 0.67 (projected from Kang et al. 2025) |

### Credence Bands (Claude 3.5 Sonnet)

| Theory | Prior [10%, 90%] | Posterior [10%, 90%] |
|--------|------------------|----------------------|
| GNW | [0.03, 0.44] | [0.18, 0.48] |
| HOT | [0.03, 0.44] | [0.15, 0.42] |
| IIT | [0.02, 0.32] | [0.05, 0.28] |

## Reference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Prior on z_t | Beta(1,4) | Skeptical prior; burden on evidence |
| λ (robustness) | 0.5 / 1.0 | Exploratory / confirmatory mode |
| K (paraphrases) | ≥5 | Minimum for stable variance |
| Overlap penalty | ρ_eff = ρ(1-0.5·o) | 50% discount if shared channel |

## Important Notes

1. **O-stream results are projected** from Kang et al. (2025), not from an actual rater study. The protocol for conducting empirical O-stream assessment is provided in `supplementary/tcas_o_stream_protocol.md`.

2. **M-stream (mechanistic) is N/A** for black-box systems like Claude. The framework supports M-stream for systems with architectural access.

3. **Threats to validity** are explicitly tracked: black-box limitation, projected O-stream, optimization risk.

## Citation

```bibtex
@inproceedings{hughes2026tcas,
  title={Triangulating Evidence for Machine Consciousness Claims:
         A Validity-Centered Stack of Behavioral Batteries,
         Mechanistic Indicators, Perturbation Tests, and Credence Reporting},
  author={Hughes, Scott},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

MIT License - see code/LICENSE for details.
