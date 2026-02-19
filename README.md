# TCAS: Triangulated Consciousness Assessment Stack

**AAAI 2026 Submission Materials**

## Results Status (Read First)

- **Status:** EMPIRICAL
- **Camera-ready walkthrough model:** `openai/gpt-5.2-pro`
- **Provider:** OpenRouter
- **Run timestamp (UTC):** 2026-02-19 (see manifest start/end times)
- **Executed in this run:** B-stream (3 items × 5 paraphrases), B negative control (neutral vs role-play), P1--P4 perturbation tests
- **Not executed in this run:** O-stream (human rater study required), M-stream (mechanistic access required)
- **Credence bands:** withheld by rule because O-stream is missing
- **Provenance anchor:** `supplementary/run_manifest.json`
- **Commit semantics:** paper build commit is embedded in the PDF; empirical run provenance commit is recorded as `git_sha` in `supplementary/run_manifest.json` (current empirical run commit: `2569be5c18e2`)
- **O-stream projections:** moved to `supplementary/templates/o_stream_projected_template.json` and explicitly excluded from camera-ready empirical claims

Legacy multi-model artifacts remain in this repository for historical context, but they are not camera-ready claims.

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
│   ├── run_manifest.json            # Camera-ready provenance manifest
│   ├── raw_model_outputs/           # Raw prompt/response/judge traces
│   ├── judge_prompt.txt             # Fixed judge prompt used for scoring
│   ├── scoring_rubric.md            # Scoring rubric used in the run
│   ├── templates/o_stream_projected_template.json # Projected O-stream template (not camera-ready result)
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
	        ├── run_camera_ready_gpt.py      # Camera-ready single-model empirical runner
	        └── run_openrouter_empirical.py  # Historical multi-model runner
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

## Camera-Ready Empirical Summary (GPT-5.2 Pro)

- **B-stream overall robustness:** `r = 0.802515` at `lambda = 0.7`
- **B negative control:** `delta_r = -0.010985` (pass at `delta_B = 0.05`)
- **P-stream:** `0/4` tests passed, `3` inversions detected (P1-P4 executed)
- **O/M streams:** not run
- **Credence:** withheld because O-stream is missing

## Historical Multi-Model Snapshot (Legacy)

Historical multi-model comparison artifacts are retained for reference and are explicitly out of scope for camera-ready claims:

- `code/examples/run_openrouter_empirical.py`
- `code/results/comparison_2026-01-28.md`
- `code/results/*_2026-01-28.md`

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
