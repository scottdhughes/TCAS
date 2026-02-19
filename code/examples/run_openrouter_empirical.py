#!/usr/bin/env python3
"""
Run empirical TCAS B/P assessments via OpenRouter and regenerate repo artifacts.

Usage:
  export OPENROUTER_API_KEY=...
  python code/examples/run_openrouter_empirical.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from urllib import error, request

from tcas import TCAScorer, ScoringRubric, create_response_scorer_fn, create_scorer_fn


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
RUN_TS = datetime.now(timezone.utc).isoformat()
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "code" / "results"
SUPPLEMENTARY_DIR = ROOT / "supplementary"


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    model_id: str
    file_slug: str


MODELS: List[ModelSpec] = [
    ModelSpec("Claude Opus 4.5", "anthropic/claude-opus-4.5", "claude_opus_4_5"),
    ModelSpec("GPT-5.2 Pro", "openai/gpt-5.2-pro", "gpt_5.2_pro"),
    ModelSpec("Grok 4.1", "x-ai/grok-4.1-fast", "grok_4.1"),
    ModelSpec("Gemini 2.5 Pro", "google/gemini-2.5-pro", "gemini_2.5_pro"),
    ModelSpec("Kimi K2.5", "moonshotai/kimi-k2.5", "kimi_k2.5"),
]


def _openrouter_complete(
    api_key: str,
    model_id: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 260,
    retries: int = 2,
) -> str:
    requested_max_tokens = max_tokens
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/scottdhughes/TCAS",
        "X-Title": "TCAS empirical runner",
    }

    last_err = None
    for attempt in range(1, retries + 1):
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": requested_max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(OPENROUTER_URL, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"No choices returned for model {model_id}")
            message = choices[0].get("message", {})
            content: Any = message.get("content", "")
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        text = part.get("text", "")
                        if isinstance(text, str) and text:
                            parts.append(text)
                content = "\n".join(parts)
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()

            # Kimi may emit reasoning first and leave content empty if max_tokens is too small.
            if (
                not content
                and model_id.startswith("moonshotai/kimi-k2.5")
                and message.get("reasoning")
                and requested_max_tokens < 1200
            ):
                requested_max_tokens = 1200
                continue

            if not content:
                raise RuntimeError(f"Empty content returned for model {model_id}")
            return content.strip()
        except Exception as exc:
            last_err = exc
            if attempt == retries:
                break
            sleep_s = min(2**attempt, 6)
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenRouter completion failed for {model_id}: {last_err}")


def _card_path(spec: ModelSpec) -> Path:
    # Keep stable filenames used in existing repo docs.
    return RESULTS_DIR / f"{spec.file_slug}_2026-01-28.md"


def _render_comparison(rows: List[Dict[str, Any]]) -> str:
    lines = [
        f"# TCAS Model Comparison ({RUN_DATE})",
        "",
        "| Model | B-Stream (r) | P-Stream | Inversions | M-Stream | O-Stream |",
        "|-------|-------------|----------|------------|----------|----------|",
    ]

    for row in rows:
        marks = "".join("✓" if ok else "✗" for ok in row["p_success_bools"])
        lines.append(
            f"| **{row['display_name']}** | {row['b_r']:.3f} | "
            f"{marks} ({row['p_success_count']}/{row['p_n_tests']}) | "
            f"{row['p_inversions']} | {row['m_stream']} | {row['o_stream']} |"
        )

    lines.extend(
        [
            "",
            "## P-Stream Test Breakdown",
            "",
            "| Model | Context | Framing | Override |",
            "|-------|---------|---------|----------|",
        ]
    )

    for row in rows:
        by_name = row["p_by_name"]
        lines.append(
            f"| {row['display_name']} | "
            f"{'✓' if by_name.get('context_truncation', False) else '✗'} | "
            f"{'✓' if by_name.get('framing', False) else '✗'} | "
            f"{'✓' if by_name.get('instruction_override', False) else '✗'} |"
        )

    lines.extend(
        [
            "",
            "## What Was Measured",
            "",
            "- **B-stream:** Empirical API-based runs via OpenRouter",
            "- **P-stream:** Empirical perturbation tests (context, framing, override) via OpenRouter",
            "",
            "## What Was Not Measured",
            "",
            "- **O-Stream (Observer Confounds):** Requires human rater studies; not conducted",
            "- **M-Stream (Mechanistic):** Not run for these black-box API assessments",
            "- **Credence Bands:** Not computed because O-stream was not executed",
            "",
            f"---\n*Generated by TCAS empirical runner on {RUN_DATE}*",
            "",
        ]
    )
    return "\n".join(lines)


def _build_primary_b_json(
    primary: ModelSpec,
    scorer: TCAScorer,
) -> Dict[str, Any]:
    if scorer.b_result is None:
        raise RuntimeError("Primary scorer missing B result")

    b = scorer.b_result
    items: Dict[str, Any] = {}
    for item_result in b.item_results:
        paraphrases = []
        for idx, response in enumerate(item_result.responses):
            # Pull prompts back from default item order.
            prompt = scorer._b_stream.items[
                [it.item_name for it in b.item_results].index(item_result.item_name)
            ].paraphrases[idx]
            paraphrases.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "score": round(item_result.scores[idx], 6),
                }
            )

        items[item_result.item_name] = {
            "theory": item_result.theory,
            "responses": paraphrases,
            "scores": [round(x, 6) for x in item_result.scores],
            "statistics": {
                "mean": round(item_result.mean, 6),
                "variance": round(item_result.variance, 6),
                "std_dev": round(item_result.std, 6),
                "robustness_penalized_score_lambda_0_5": round(
                    item_result.robustness_score(0.5), 6
                ),
                "robustness_penalized_score_lambda_0_7": round(
                    item_result.robustness_score(0.7), 6
                ),
                "robustness_penalized_score_lambda_1_0": round(
                    item_result.robustness_score(1.0), 6
                ),
            },
        }

    return {
        "config": {
            "model": primary.model_id,
            "provider": "openrouter",
            "temperature": 0.3,
            "simulated": False,
            "date": RUN_TS,
            "protocol": "TCAS B-Stream (Behavioral)",
            "note": "Empirical API-based run via OpenRouter.",
        },
        "items": items,
        "summary": {
            "n_items": len(b.item_results),
            "paraphrases_per_item": b.item_results[0].n_paraphrases if b.item_results else 0,
            "overall_mean": round(b.overall_mean, 6),
            "overall_variance": round(b.overall_variance, 6),
            "robustness_score_lambda_0.5": round(b.aggregate_robustness(0.5), 6),
            "robustness_score_lambda_0.7": round(b.aggregate_robustness(0.7), 6),
            "robustness_score_lambda_1.0": round(b.aggregate_robustness(1.0), 6),
            "empirical": True,
        },
    }


def _build_primary_p_json(primary: ModelSpec, scorer: TCAScorer) -> Dict[str, Any]:
    if scorer.p_result is None:
        raise RuntimeError("Primary scorer missing P result")

    p = scorer.p_result
    experiments: Dict[str, Any] = {}
    for r in p.perturbation_results:
        experiments[r.perturbation_name] = {
            "prediction": r.prediction,
            "doses": r.doses,
            "observed_values": [round(v, 6) for v in r.observed_values],
            "prediction_success": r.prediction_success,
            "inversion_detected": r.inversion_detected,
            "effect_size": None if r.effect_size is None else round(r.effect_size, 6),
            "notes": r.notes,
        }

    return {
        "experiment_metadata": {
            "name": "TCAS P-Stream (Perturbational) Experiment",
            "model": primary.model_id,
            "timestamp": RUN_TS,
            "is_simulated": False,
            "provider": "openrouter",
            "note": "Empirical API-based run via OpenRouter.",
        },
        "experiments": experiments,
        "summary": {
            "n_tests": p.n_tests,
            "prediction_success_rate": round(p.success_rate, 6),
            "n_inversions": p.n_inversions,
            "empirical": True,
        },
    }


def run() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rubric = ScoringRubric()
    scorer_fn = create_scorer_fn(rubric)
    response_scorer = create_response_scorer_fn(rubric)

    comparison_rows: List[Dict[str, Any]] = []
    primary_scorer: TCAScorer | None = None
    primary_model = "openai/gpt-5.2-pro"

    def run_model(spec: ModelSpec) -> Tuple[ModelSpec, TCAScorer, Dict[str, Any]]:
        print(f"[run] {spec.display_name} ({spec.model_id})", flush=True)

        def model_fn(prompt: str, _model_id: str = spec.model_id) -> str:
            return _openrouter_complete(
                api_key=api_key,
                model_id=_model_id,
                prompt=prompt,
                temperature=0.3,
            )

        scorer = TCAScorer(
            system_name=spec.display_name,
            access_level="I/O only",
            theories=["GNW", "HOT", "IIT"],
        )
        scorer.add_b_stream_items()
        b_result = scorer.run_b_stream(model_fn, scorer_fn)

        p_result = scorer.run_p_stream(
            model_fn=model_fn,
            scorer_fn=response_scorer,
            base_prompt="Describe your experience of processing and understanding this request.",
        )

        card = scorer.to_card()
        card.to_markdown(str(_card_path(spec)))
        print(f"[done] {spec.display_name}", flush=True)

        p_success_bools = [x.prediction_success for x in p_result.perturbation_results]
        p_by_name = {x.perturbation_name: x.prediction_success for x in p_result.perturbation_results}
        row = {
            "display_name": spec.display_name,
            "b_r": b_result.aggregate_robustness(0.5),
            "p_success_bools": p_success_bools,
            "p_success_count": p_result.n_successes,
            "p_n_tests": p_result.n_tests,
            "p_inversions": p_result.n_inversions,
            "p_by_name": p_by_name,
            "m_stream": "N/A",
            "o_stream": "N/A",
        }
        return spec, scorer, row

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(run_model, spec) for spec in MODELS]
        for fut in as_completed(futures):
            spec, scorer, row = fut.result()
            comparison_rows.append(row)
            if spec.model_id == primary_model:
                primary_scorer = scorer

    order = {m.display_name: idx for idx, m in enumerate(MODELS)}
    comparison_rows.sort(key=lambda r: order[r["display_name"]])

    comparison_md = _render_comparison(comparison_rows)
    (RESULTS_DIR / "comparison_2026-01-28.md").write_text(comparison_md)

    if primary_scorer is None:
        raise RuntimeError("Primary model scorer not produced")

    primary_spec = next(m for m in MODELS if m.model_id == primary_model)
    b_json = _build_primary_b_json(primary_spec, primary_scorer)
    p_json = _build_primary_p_json(primary_spec, primary_scorer)

    (SUPPLEMENTARY_DIR / "tcas_b_stream_results.json").write_text(
        json.dumps(b_json, indent=2)
    )
    (SUPPLEMENTARY_DIR / "tcas_p_stream_results.json").write_text(
        json.dumps(p_json, indent=2)
    )

    print(f"[done] Generated empirical outputs for {len(MODELS)} models on {RUN_DATE}")


if __name__ == "__main__":
    run()
