from __future__ import annotations

from pathlib import Path
from typing import cast

from sinclair._obs import ObserverFn

import pandas as pd

from .artifacts import UseCaseArtifacts, bundle_artifacts
from .config import SurveyDefaults
from .context import ContextColumns
from .render import SurveyFrontendController
from .runtime import chat, load_dataframe, report_question, report_study
from .store import SurveyArtifactStore


class SurveyStudy:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        study_id: str | None = None,
        study_context: str | None = None,
        question_map: dict[str, str] | None = None,
        column_metadata: str | None = None,
        language_instruction: str | None = None,
        context_columns: ContextColumns = "*",
        store: SurveyArtifactStore | None = None,
        defaults: SurveyDefaults | None = None,
    ) -> None:
        self.df = df
        self.study_id = study_id or "study"
        self.study_context = study_context
        self.question_map = question_map or {}
        self.column_metadata = column_metadata
        self.language_instruction = language_instruction
        self.context_columns = context_columns
        self.defaults = defaults or SurveyDefaults()
        self.store = store or SurveyArtifactStore(
            identity=self.defaults.identity
        )
        self._question_reports: dict[str, Report] = {}
        self._study_report: Report | None = None
        self._chat_reports: dict[str, list[Report]] = {}

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        study_id: str | None = None,
        **kwargs,
    ) -> "SurveyStudy":
        path = Path(csv_path)
        return cls(
            load_dataframe(path), study_id=study_id or path.stem, **kwargs
        )

    def report_question(
        self,
        question_column: str,
        *,
        prompt: str | None = None,
        context_columns: ContextColumns | None = None,
        model: str | None = None,
        verbose: bool | None = None,
        observers: list[ObserverFn] | None = None,
    ) -> Report:
        report = report_question(
            self.df,
            question_column,
            prompt=prompt,
            context_columns=cast(
                ContextColumns,
                self.context_columns
                if context_columns is None
                else context_columns,
            ),
            store=self.store,
            model=model or self.defaults.model,
            verbose=self.defaults.verbose if verbose is None else verbose,
            study_context=self.study_context,
            question_map=self.question_map,
            column_metadata=self.column_metadata,
            language_instruction=self.language_instruction,
            defaults=self.defaults,
            observers=observers,
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
        observers: list[ObserverFn] | None = None,
    ) -> Report:
        reports = question_reports or list(self._question_reports.values())
        report = report_study(
            self.df,
            reports,
            prompt=prompt,
            context_columns=cast(
                ContextColumns,
                self.context_columns
                if context_columns is None
                else context_columns,
            ),
            store=self.store,
            model=model or self.defaults.model,
            verbose=self.defaults.verbose if verbose is None else verbose,
            study_context=self.study_context,
            question_map=self.question_map,
            column_metadata=self.column_metadata,
            language_instruction=self.language_instruction,
            defaults=self.defaults,
            observers=observers,
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
        observers: list[ObserverFn] | None = None,
    ) -> Report | str:
        reply = chat(
            self.df,
            query,
            thread_id=thread_id,
            question_reports=question_reports
            or list(self._question_reports.values()),
            study_report=study_report or self._study_report,
            context_columns=cast(
                ContextColumns,
                self.context_columns
                if context_columns is None
                else context_columns,
            ),
            store=self.store,
            model=model or self.defaults.model,
            verbose=self.defaults.verbose if verbose is None else verbose,
            study_context=self.study_context,
            question_map=self.question_map,
            column_metadata=self.column_metadata,
            language_instruction=self.language_instruction,
            defaults=self.defaults,
            observers=observers,
        )
        if isinstance(reply, Report):
            self._chat_reports.setdefault(thread_id, []).append(reply)
        return reply

    def export_artifacts(self) -> UseCaseArtifacts:
        return bundle_artifacts(
            question_reports=self._question_reports,
            study_report=self._study_report,
            chat_reports=[
                report
                for reports in self._chat_reports.values()
                for report in reports
            ],
            store=self.store,
            metadata={"study_id": self.study_id},
        )

    def frontend_controller(self) -> SurveyFrontendController:
        reports: dict[str, Report] = {
            f"question:{question_id}": report
            for question_id, report in self._question_reports.items()
        }
        if self._study_report is not None:
            reports["study"] = self._study_report
        for thread_id, chat_reports in self._chat_reports.items():
            for index, report in enumerate(chat_reports, start=1):
                reports[f"chat:{thread_id}:{index}"] = report
        return SurveyFrontendController(store=self.store, reports=reports)


from .models import Report
