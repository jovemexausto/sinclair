from __future__ import annotations

import json
from typing import Any, Literal

import pandas as pd

from .models import Report
from .store import SurveyArtifactStore

ContextColumns = list[str] | Literal["*"] | None


def question_scope(question_column: str) -> str:
    return f"question:{question_column}"


def study_scope() -> str:
    return "study:executive"


def chat_scope(thread_id: str) -> str:
    return f"chat:{thread_id}"


def report_scope(report: Report, index: int) -> str:
    question_id = _report_question_id(report)
    if question_id is not None:
        return question_scope(question_id)
    return f"report:{index + 1}"


def select_columns(
    df: pd.DataFrame,
    question_column: str | None,
    context_columns: ContextColumns,
    identity,
) -> pd.DataFrame:
    id_columns = [
        col
        for col in (
            identity.respondent_id_column,
            *identity.fallback_respondent_id_columns,
        )
        if col in df.columns
    ]
    if context_columns == "*":
        columns = list(df.columns)
    elif context_columns is None:
        columns = id_columns + ([question_column] if question_column else [])
    else:
        columns = (
            id_columns
            + ([question_column] if question_column else [])
            + list(context_columns)
        )
    seen: set[str] = set()
    selected = [
        col
        for col in columns
        if col in df.columns and not (col in seen or seen.add(col))
    ]
    if question_column is not None and question_column not in selected:
        raise ValueError(f"question_column not found: {question_column!r}")
    return df[selected].copy()


def ingest_question_reports(
    store: SurveyArtifactStore,
    scoped_df: pd.DataFrame,
    question_reports: list[Report] | None,
) -> None:
    if not question_reports:
        return
    for index, report in enumerate(question_reports):
        store.ingest_report(report, scoped_df, scope=report_scope(report, index))


def study_working_material(store: SurveyArtifactStore) -> str:
    findings_inventory = [
        {
            "finding_id": record.finding_id,
            "scope": record.scope,
            "claim": record.claim,
            "implication": record.implication,
            "confidence": record.confidence,
            "evidence_ids": record.evidence_ids,
        }
        for record in store.list_findings()
    ]
    return json.dumps({"findings": findings_inventory}, ensure_ascii=False)


def chat_working_material(
    store: SurveyArtifactStore, study_report: Report | None
) -> str:
    payload: dict[str, Any] = {
        "findings": [
            {
                "finding_id": record.finding_id,
                "scope": record.scope,
                "claim": record.claim,
                "implication": record.implication,
                "confidence": record.confidence,
            }
            for record in store.list_findings()
        ],
        "study_report": None,
    }
    if study_report is not None:
        payload["study_report"] = {
            "finding_count": len(study_report.findings),
            "chart_count": len(study_report.charts),
            "citations": len(study_report.citations),
            "markdown_excerpt": study_report.markdown[:600],
        }
    return json.dumps(payload, ensure_ascii=False)


def _report_question_id(report: Report) -> str | None:
    for chart in report.charts:
        if chart.nr_questao:
            return chart.nr_questao
    for finding in report.findings:
        for evidence in finding.evidences:
            if evidence.question_id:
                return evidence.question_id
            if evidence.source_column:
                return evidence.source_column
    return None
