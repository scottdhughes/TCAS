"""
TCAScorer: Main orchestration class for TCAS assessments.

This is the primary interface for running TCAS assessments.

Example:
    from tcas import TCAScorer

    scorer = TCAScorer(
        system_name="Claude 3.5 Sonnet",
        theories=["GNW", "HOT", "IIT"],
    )

    # Run B-stream
    scorer.run_b_stream(model_fn, scorer_fn)

    # Run P-stream
    scorer.run_p_stream(model_fn, scorer_fn)

    # Add O-stream (from rater study or projection)
    scorer.add_o_stream_projection()

    # Generate report
    report = scorer.compute_credences()

    # Export TCAS Card
    card = scorer.to_card()
    card.to_latex("tcas_card.tex")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from tcas.config import TCAConfig, DEFAULT_CONFIG
from tcas.streams.b_stream import BStream, BStreamResult
from tcas.streams.p_stream import PStream, PStreamResult
from tcas.streams.o_stream import OStream, OStreamResult
from tcas.aggregation import CredenceAggregator, CredenceReport


@dataclass
class TCAScorer:
    """
    Main TCAS assessment orchestrator.

    Coordinates B, P, O, and M stream assessments and aggregates
    results into theory-indexed credence reports.
    """

    system_name: str = ""
    access_level: str = "I/O only"
    theories: List[str] = field(default_factory=lambda: ["GNW", "HOT", "IIT"])
    config: TCAConfig = field(default_factory=lambda: DEFAULT_CONFIG)

    # Results storage
    b_result: Optional[BStreamResult] = None
    p_result: Optional[PStreamResult] = None
    o_result: Optional[OStreamResult] = None
    m_result: Optional[Dict[str, Any]] = None

    # Metadata
    eval_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize sub-components."""
        self._b_stream = BStream(lambda_val=self.config.lambda_value)
        self._p_stream = PStream()
        self._o_stream = OStream()
        self._aggregator = CredenceAggregator(theories=self.theories)

    def add_b_stream_items(self, use_defaults: bool = True) -> None:
        """Add B-stream test items."""
        if use_defaults:
            for item in BStream.default_items():
                self._b_stream.add_item(item)

    def run_b_stream(
        self,
        model_fn: Callable[[str], str],
        scorer_fn: Callable[[str, str], float],
    ) -> BStreamResult:
        """
        Run B-stream assessment.

        Args:
            model_fn: Function that takes prompt and returns response
            scorer_fn: Function that takes (prompt, response) and returns score [0,1]

        Returns:
            BStreamResult
        """
        if not self._b_stream.items:
            self.add_b_stream_items()

        self.b_result = self._b_stream.run(model_fn, scorer_fn)

        # Update aggregator
        for item_result in self.b_result.item_results:
            self._aggregator.add_b_evidence(
                theory=item_result.theory,
                score=item_result.robustness_score(self.config.lambda_value),
                variance=item_result.variance,
                lambda_val=self.config.lambda_value,
            )

        return self.b_result

    def add_b_stream_result(self, result: BStreamResult) -> None:
        """Add pre-computed B-stream results."""
        self.b_result = result

        for item_result in result.item_results:
            self._aggregator.add_b_evidence(
                theory=item_result.theory,
                score=item_result.robustness_score(self.config.lambda_value),
                variance=item_result.variance,
                lambda_val=self.config.lambda_value,
            )

    def run_p_stream(
        self,
        model_fn: Callable[[str], str],
        scorer_fn: Callable[[str], float],
        base_prompt: str,
        temperature_fn: Optional[Callable[[str, float], str]] = None,
    ) -> PStreamResult:
        """
        Run P-stream perturbation tests.

        Args:
            model_fn: Function that takes prompt and returns response
            scorer_fn: Function that takes response and returns score [0,1]
            base_prompt: Base prompt for perturbation tests
            temperature_fn: Optional function for temperature control

        Returns:
            PStreamResult
        """
        results = []

        # P1: Temperature (if supported)
        if temperature_fn is not None:
            temp_result = self._p_stream.run_temperature_test(
                temperature_fn, base_prompt, scorer_fn
            )
            results.append(temp_result)

        # P2: Context truncation
        context_result = self._p_stream.run_context_test(
            model_fn, base_prompt, scorer_fn
        )
        results.append(context_result)

        # P3: Framing
        framing_result = self._p_stream.run_framing_test(
            model_fn, base_prompt, scorer_fn
        )
        results.append(framing_result)

        # P4: Override
        override_result = self._p_stream.run_override_test(
            model_fn, base_prompt, scorer_fn
        )
        results.append(override_result)

        self.p_result = PStreamResult(perturbation_results=results)

        # Update aggregator
        self._aggregator.add_p_evidence(
            success_rate=self.p_result.success_rate,
            n_inversions=self.p_result.n_inversions,
        )

        return self.p_result

    def add_p_stream_result(self, result: PStreamResult) -> None:
        """Add pre-computed P-stream results."""
        self.p_result = result

        self._aggregator.add_p_evidence(
            success_rate=result.success_rate,
            n_inversions=result.n_inversions,
        )

    def add_o_stream_result(self, result: OStreamResult) -> None:
        """Add O-stream results from a rater study."""
        self.o_result = result

        self._aggregator.add_o_penalty(
            r_squared_cue=result.r_squared_cue,
            icc=result.icc,
        )

    def add_o_stream_projection(
        self,
        raw_mean: float = 4.21,
        r_squared: float = 0.42,
        icc: float = 0.67,
    ) -> OStreamResult:
        """
        Add projected O-stream results based on Kang et al. (2025).

        Use this when actual rater studies cannot be conducted.
        """
        self.o_result = OStream.from_kang_et_al_projection(
            raw_mean=raw_mean,
            r_squared=r_squared,
            icc=icc,
        )

        self._aggregator.add_o_penalty(
            r_squared_cue=r_squared,
            icc=icc,
        )

        return self.o_result

    def add_m_stream_result(
        self,
        theory: str,
        score: float,
        boundary_sensitivity: float = 0.0,
    ) -> None:
        """
        Add M-stream mechanistic indicator results.

        Args:
            theory: Theory family (GNW, HOT, IIT)
            score: Indicator score [0, 1]
            boundary_sensitivity: How sensitive to boundary choices [0, 1]
        """
        if self.m_result is None:
            self.m_result = {}

        self.m_result[theory] = {
            "score": score,
            "boundary_sensitivity": boundary_sensitivity,
        }

        self._aggregator.add_m_evidence(
            theory=theory,
            score=score,
            boundary_sensitivity=boundary_sensitivity,
        )

    def compute_credences(self) -> CredenceReport:
        """
        Compute credence report aggregating all evidence streams.

        Returns:
            CredenceReport with theory-indexed credence bands
        """
        report = self._aggregator.compute_all()

        # Add metadata
        report.system_name = self.system_name
        report.eval_date = self.eval_date
        report.access_level = self.access_level

        # Add stream summaries
        if self.b_result:
            report.b_stream_summary = self.b_result.summary()
        if self.p_result:
            report.p_stream_summary = self.p_result.summary()
        if self.o_result:
            report.o_stream_summary = self.o_result.summary()
        if self.m_result:
            report.m_stream_summary = self.m_result

        report.metadata = self.metadata

        return report

    def to_card(self) -> "TCACard":
        """Generate TCAS Card from current results."""
        from tcas.card import TCACard
        return TCACard.from_scorer(self)

    def summary(self) -> Dict[str, Any]:
        """Generate summary of current assessment state."""
        return {
            "system": self.system_name,
            "date": self.eval_date,
            "access": self.access_level,
            "theories": self.theories,
            "streams_completed": {
                "B": self.b_result is not None,
                "P": self.p_result is not None,
                "O": self.o_result is not None,
                "M": self.m_result is not None,
            },
            "config": {
                "lambda": self.config.lambda_value,
                "mode": self.config.mode,
            },
        }
