from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sinclair import Agent, AgentConfig, PythonKernel, make_trace_observer
from sinclair._obs import ObserverFn

from .config import SurveyDefaults
from .models import Report
from .store import SurveyArtifactStore
from .tools import SurveyToolKit
from .validators import validate_report


def resolve_runtime(
    defaults: SurveyDefaults | None,
    store: SurveyArtifactStore | None,
    *,
    model: str,
    verbose: bool,
) -> tuple[SurveyDefaults, SurveyArtifactStore]:
    defaults = defaults or SurveyDefaults(model=model, verbose=verbose)
    store = store or SurveyArtifactStore(identity=defaults.identity)
    return defaults, store


def run_report(
    df: pd.DataFrame,
    system_prompt: str,
    user_prompt: str,
    *,
    store: SurveyArtifactStore | None,
    defaults: SurveyDefaults,
    model: str,
    verbose: bool,
    analysis_question_id: str | None = None,
    require_publishable_data: bool = False,
    allow_simple_reply: bool = False,
    observers: list[ObserverFn] | None = None,
) -> Report | str:
    report_kind = (
        "chat"
        if allow_simple_reply
        else (
            "question_report"
            if analysis_question_id is not None
            else "study_report"
        )
    )

    active_observers = list(observers or [])
    if verbose:
        active_observers.append(make_trace_observer())
    tools = []
    if store is not None:
        tools.extend(store.as_tools())
    tools.extend(SurveyToolKit(df=df, identity=defaults.identity).as_tools())
    policy = defaults.validation_policy
    agent = Agent(
        tools=tools,
        config=AgentConfig(
            model=model,
            temperature=defaults.temperature,
            reasoning_level=defaults.reasoning_level,
            finalization_window=defaults.finalization_window,
            system_prompt=system_prompt,
            response_schema=Report,
            response_validator=lambda report: validate_report(
                report,
                df,
                policy=policy,
                analysis_question_id=analysis_question_id,
            ),
            allow_simple_reply=allow_simple_reply,
            max_iterations=defaults.max_iterations,
            tool_retries=defaults.tool_retries,
            observers=active_observers,
        ),
    )
    kernel_env: dict[str, Any] = {"df": df, "pd": pd, "np": np}
    if store is not None:
        kernel_env["store"] = store
    result = agent.run(
        user_prompt,
        kernel=PythonKernel(env=kernel_env),
        metadata={
            "require_publishable_data": require_publishable_data,
        },
    )
    if allow_simple_reply and isinstance(result.reply, str):
        return result.reply
    if not isinstance(result.reply, Report):
        raise RuntimeError(
            f"expected Report, got {type(result.reply).__name__}"
        )
    return result.reply
