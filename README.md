# TCAS: Triangulated Consciousness Assessment Stack

**AAAI 2026 Submission Materials**

## Provenance Status (Current Snapshot)

- **Run type in this snapshot:** Empirical API-based B/P runs via OpenRouter on 2026-02-18
- **Primary paper walkthrough model:** GPT-5.2 Pro (`openai/gpt-5.2-pro`)
- **Supplementary JSON status:** `simulated=false` in B-stream JSON and `is_simulated=false` in P-stream JSON
- **Not executed:** O-stream (human raters required) and M-stream (mechanistic access required)
- **Credence bands:** intentionally withheld because O-stream is missing

## Paper

> Hughes, S., and Nguyen, K. (2026). Triangulating Evidence for Machine Consciousness Claims: A Validity-Centered Stack of Behavioral Batteries, Mechanistic Indicators, Perturbation Tests, and Credence Reporting.

**Main paper PDF:** `paper/TCAS_Paper_AAAI.pdf`  
**Paper source:** `paper/main.tex`

## Repository Structure

```
TCAS/
├── README.md                    # This file
├── paper/
│   ├── TCAS_Paper_AAAI.pdf      # Main paper PDF
│   ├── main.tex                 # Paper source
│   ├── references.bib           # Paper bibliography source
│   ├── aaai2026.sty             # AAAI style file
│   └── aaai2026.bst             # AAAI bibliography style
├── supplementary/
│   ├── tcas_b_stream_results.json   # Behavioral battery results
│   ├── tcas_p_stream_results.json   # Perturbation test results
│   ├── tcas_o_stream_results.json   # Observer-confound projections
│   ├── tcas_o_stream_protocol.md    # O-stream protocol
│   └── load_results.py              # Script to load and display results
└── code/
    ├── pyproject.toml           # Package configuration
    ├── README.md                # Package documentation
    ├── tests/                   # Test suite
    ├── results/                 # Model comparison result cards
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
	        ├── assess_claude.py             # Basic API example
	        └── run_openrouter_empirical.py # Empirical multi-model runner
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
| Claude Opus 4.5 | 0.556 | 2/3 | 0 |
| GPT-5.2 Pro | 0.501 | 3/3 | 0 |
| Grok 4.1 | 0.505 | 3/3 | 0 |
| Gemini 2.5 Pro | 0.361 | 3/3 | 0 |
| Kimi K2.5 | 0.520 | 3/3 | 0 |

### What Was Measured

- **B-stream:** Empirical API-based runs via OpenRouter with paraphrase-invariance weighted robustness scores
- **P-stream:** Empirical perturbation tests (context truncation, framing resistance, instruction override)

### What Was Not Measured

- **O-stream:** Requires human rater studies (not conducted)
- **M-stream:** Not run for these black-box API assessments
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
  author={Hughes, Scott and Nguyen, Karen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

MIT License - see `LICENSE` for details.
