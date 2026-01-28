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
│   └── load_results.py              # Script to load and display results
└── code/
    ├── pyproject.toml           # Package configuration
    ├── README.md                # Package documentation
    ├── tcas/                    # Reference implementation
    │   ├── __init__.py
    │   ├── config.py            # Default parameters
    │   ├── scorer.py            # Main TCAScorer class
    │   ├── scoring.py           # Rubric-based scoring
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
from tcas import TCAScorer, ScoringRubric, create_scorer_fn

scorer = TCAScorer(
    system_name="Your Model",
    theories=["GNW", "HOT", "IIT"],
)

# Define your model interface
def model_fn(prompt: str) -> str:
    return your_api_call(prompt)

# Use rubric-based scorer
rubric = ScoringRubric()
scorer_fn = create_scorer_fn(rubric)

# Run assessment
scorer.add_b_stream_items()
scorer.run_b_stream(model_fn, scorer_fn)
scorer.run_p_stream(model_fn, lambda r: scorer_fn("", r), base_prompt="...")

# Generate report
card = scorer.to_card()
card.to_markdown("tcas_card.md")
```

## Experimental Results Summary

| Model | B-Stream (r) | P-Stream | Inversions |
|-------|-------------|----------|------------|
| Claude Opus 4.5 | 0.927 | 3/3 | 0 |
| Kimi K2.5 | 0.904 | 1/3 | 0 |
| Grok 4.1 | 0.806 | 2/3 | 0 |
| GPT-5.2 Pro | 0.769 | 2/3 | 0 |
| Gemini 2.5 Pro | 0.195 | 0/3 | 1 |

### What Was Measured

- **B-stream:** Paraphrase-invariance weighted robustness scores
- **P-stream:** Context truncation, framing resistance, and override resistance tests

### What Was Not Measured

- **O-stream:** Requires human rater studies (not conducted)
- **M-stream:** Requires model weights (only Kimi K2.5 is open-weights; analysis not conducted)
- **Credence bands:** Cannot compute without O-stream data

## Reference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Prior on z_t | Beta(1,4) | Skeptical prior; burden on evidence |
| λ (robustness) | 0.5 / 1.0 | Exploratory / confirmatory mode |
| K (paraphrases) | ≥5 | Minimum for stable variance |
| Overlap penalty | ρ_eff = ρ(1-0.5·o) | 50% discount if shared channel |

## Important Notes

1. **O-stream results require human raters.** The protocol for conducting empirical O-stream assessment is provided in `supplementary/tcas_o_stream_protocol.md`.

2. **M-stream (mechanistic) is N/A** for black-box systems. The framework supports M-stream for systems with architectural access.

3. **Credence bands cannot be computed** without O-stream data. The results show only B-stream and P-stream measurements.

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
