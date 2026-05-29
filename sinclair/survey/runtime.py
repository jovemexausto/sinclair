from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from sinclair._obs import ObserverFn

import pandas as pd

from .config import SurveyDefaults
from .context import (
    ContextColumns,
    chat_scope,
    chat_working_material,
    ingest_question_reports,
    question_scope,
    select_columns,
    study_scope,
    study_working_material,
)
from .models import Report
from .prompts import (
    chat_prompt,
    chat_user_prompt,
    question_prompt,
    question_user_prompt,
    study_prompt,
    study_user_prompt,
)
from .runner import resolve_runtime, run_report
from .store import SurveyArtifactStore


class SurveyApp:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        study_context: str | None = None,
        question_map: dict[str, str] | None = None,
        column_metadata: str | None = None,
        language_instruction: str | None = None,
        context_columns: ContextColumns = "*",
        store: SurveyArtifactStore | None = None,
        defaults: SurveyDefaults | None = None,
    ) -> None:
        self.df = df
        self.study_context = study_context
        self.question_map = question_map or {}
        self.column_metadata = column_metadata
        self.language_instruction = language_instruction
        self.context_columns = context_columns
        self.defaults = defaults or SurveyDefaults()
        self.store = store or SurveyArtifactStore(identity=self.defaults.identity)
        self._question_reports: dict[str, Report] = {}
        self._study_report: Report | None = None

    def report_question(
        self,
        question_column: str,
        *,
        prompt: str | None = None,
        context_columns: ContextColumns | None = None,
        model: str | None = None,
        verbose: bool | None = None,
    ) -> Report:
        report = report_question(
            self.df,
            question_column,
            prompt=prompt,
            context_columns=cast(
                ContextColumns,
                self.context_columns if context_columns is None else context_columns,
            ),
            store=self.store,
            model=model or self.defaults.model,
            verbose=self.defaults.verbose if verbose is None else verbose,
            study_context=self.study_context,
            question_map=self.question_map,
            column_metadata=self.column_metadata,
            language_instruction=self.language_instruction,
            defaults=self.defaults,
        )
        self._question_reports[question_column] = report
        return report

    def report_study(
        self,
        *,
        prompt: str | None = None,
        question_reports: list[Report] | None = None,
        context_columns: ContextColumns | None = None,
        model: str | None = None,
        verbose: bool | None = None,
    ) -> Report:
        reports = question_reports or list(self._question_reports.values())
        report = report_study(
            self.df,
            reports,
            prompt=prompt,
            context_columns=cast(
                ContextColumns,
                self.context_columns if context_columns is None else context_columns,
            ),
            store=self.store,
            model=model or self.defaults.model,
            verbose=self.defaults.verbose if verbose is None else verbose,
            study_context=self.study_context,
            question_map=self.question_map,
            column_metadata=self.column_metadata,
            language_instruction=self.language_instruction,
            defaults=self.defaults,
        )
        self._study_report = report
        return report

    def chat(
        self,
        query: str,
        *,
        thread_id: str = "default",
        question_reports: list[Report] | None = None,
        study_report: Report | None = None,
        context_columns: ContextColumns | None = None,
        model: str | None = None,
        verbose: bool | None = None,
    ) -> Report | str:
        return chat(
            self.df,
            query,
            thread_id=thread_id,
            question_reports=question_reports or list(self._question_reports.values()),
            study_report=study_report or self._study_report,
            context_columns=cast(
                ContextColumns,
                self.context_columns if context_columns is None else context_columns,
            ),
            store=self.store,
            model=model or self.defaults.model,
            verbose=self.defaults.verbose if verbose is None else verbose,
            study_context=self.study_context,
            question_map=self.question_map,
            column_metadata=self.column_metadata,
            language_instruction=self.language_instruction,
            defaults=self.defaults,
        )


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def report_question(
    df: pd.DataFrame,
    question_column: str,
    *,
    prompt: str | None = None,
    context_columns: ContextColumns = "*",
    store: SurveyArtifactStore | None = None,
    model: str = "gpt-5.4",
    verbose: bool = False,
    study_context: str | None = None,
    question_map: dict[str, str] | None = None,
    column_metadata: str | None = None,
    language_instruction: str | None = None,
    defaults: SurveyDefaults | None = None,
    observers: list[ObserverFn] | None = None,
) -> Report:
    defaults, store = resolve_runtime(defaults, store, model=model, verbose=verbose)
    scoped_df = select_columns(df, question_column, context_columns, defaults.identity)
    system_prompt = question_prompt(
        question_column,
        scoped_df.columns.tolist(),
        study_context=study_context,
        question_map=question_map,
        column_metadata=column_metadata,
        language_instruction=language_instruction,
    )
    user_prompt = question_user_prompt(question_column, prompt)
    report = run_report(
        scoped_df,
        system_prompt,
        user_prompt,
        store=store,
        defaults=defaults,
        model=model,
        verbose=verbose,
        analysis_question_id=question_column,
        require_publishable_data=True,
        observers=observers,
    )
    if not isinstance(report, Report):
        raise RuntimeError(f"expected Report, got {type(report).__name__}")
    store.ingest_report(report, scoped_df, scope=question_scope(question_column))
    return report


def report_study(
    df: pd.DataFrame,
    question_reports: list[Report] | None = None,
    *,
    prompt: str | None = None,
    context_columns: ContextColumns = "*",
    store: SurveyArtifactStore | None = None,
    model: str = "gpt-5.4",
    verbose: bool = False,
    study_context: str | None = None,
    question_map: dict[str, str] | None = None,
    column_metadata: str | None = None,
    language_instruction: str | None = None,
    defaults: SurveyDefaults | None = None,
    observers: list[ObserverFn] | None = None,
) -> Report:
    defaults, store = resolve_runtime(defaults, store, model=model, verbose=verbose)
    study_defaults = replace(
        defaults,
        validation_policy=replace(
            defaults.validation_policy,
            min_charts=max(defaults.validation_policy.min_charts, 5),
        ),
    )
    scoped_df = select_columns(df, None, context_columns, defaults.identity)
    ingest_question_reports(store, scoped_df, question_reports)
    system_prompt = study_prompt(
        scoped_df.columns.tolist(),
        study_context=study_context,
        question_map=question_map,
        column_metadata=column_metadata,
        language_instruction=language_instruction,
    )
    user_prompt = study_user_prompt(study_working_material(store), prompt)
    report = run_report(
        scoped_df,
        system_prompt,
        user_prompt,
        store=store,
        defaults=study_defaults,
        model=model,
        verbose=verbose,
        require_publishable_data=True,
        observers=observers,
    )
    if not isinstance(report, Report):
        raise RuntimeError(f"expected Report, got {type(report).__name__}")
    store.ingest_report(report, scoped_df, scope=study_scope())
    return report


def chat(
    df: pd.DataFrame,
    query: str,
    *,
    thread_id: str = "default",
    question_reports: list[Report] | None = None,
    study_report: Report | None = None,
    context_columns: ContextColumns = "*",
    store: SurveyArtifactStore | None = None,
    model: str = "gpt-5.4",
    verbose: bool = False,
    study_context: str | None = None,
    question_map: dict[str, str] | None = None,
    column_metadata: str | None = None,
    language_instruction: str | None = None,
    defaults: SurveyDefaults | None = None,
    observers: list[ObserverFn] | None = None,
) -> Report | str:
    defaults, store = resolve_runtime(defaults, store, model=model, verbose=verbose)
    scoped_df = select_columns(df, None, context_columns, defaults.identity)
    ingest_question_reports(store, scoped_df, question_reports)
    if study_report is not None:
        store.ingest_report(study_report, scoped_df, scope=study_scope())
    working_material = chat_working_material(store, study_report)
    system_prompt = chat_prompt(
        scoped_df.columns.tolist(),
        study_context=study_context,
        question_map=question_map,
        column_metadata=column_metadata,
        language_instruction=language_instruction,
    )
    user_prompt = chat_user_prompt(query, working_material)
    report = run_report(
        scoped_df,
        system_prompt,
        user_prompt,
        store=store,
        defaults=defaults,
        model=model,
        verbose=verbose,
        allow_simple_reply=True,
        observers=observers,
    )
    if isinstance(report, Report):
        store.ingest_report(report, scoped_df, scope=chat_scope(thread_id))
    return report
