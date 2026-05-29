from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .runtime import SurveyApp


@dataclass(slots=True)
class BenchmarkRun:
    mode: str
    seconds: float


@dataclass(slots=True)
class BenchmarkResult:
    cold: BenchmarkRun
    warm: BenchmarkRun

    @property
    def speedup(self) -> float | None:
        if self.warm.seconds == 0:
            return None
        return self.cold.seconds / self.warm.seconds


@dataclass(slots=True)
class BaselineRun:
    question_column: str
    seconds: float


def benchmark_question_report(
    app: SurveyApp,
    question_column: str,
    *,
    prompt: str | None = None,
    verbose: bool = False,
) -> BenchmarkResult:
    t0 = perf_counter()
    app.report_question(question_column, prompt=prompt, verbose=verbose)
    cold = perf_counter() - t0

    t1 = perf_counter()
    app.report_question(question_column, prompt=prompt, verbose=verbose)
    warm = perf_counter() - t1

    return BenchmarkResult(
        cold=BenchmarkRun(mode="cold", seconds=cold),
        warm=BenchmarkRun(mode="warm", seconds=warm),
    )


def baseline_question_report(
    app: SurveyApp,
    question_column: str,
    *,
    prompt: str | None = None,
    verbose: bool = False,
) -> BaselineRun:
    t0 = perf_counter()
    app.report_question(question_column, prompt=prompt, verbose=verbose)
    return BaselineRun(question_column=question_column, seconds=perf_counter() - t0)
