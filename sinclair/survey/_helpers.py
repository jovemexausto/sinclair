from __future__ import annotations

import ast
import builtins as py_builtins
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import SurveyIdentityPolicy
from .models import Evidence, Report, ResponseRef


def eval_mask(expr: str, df: pd.DataFrame) -> pd.Series:
    result = eval(
        expr,
        {"__builtins__": py_builtins.__dict__},
        {"df": df, "pd": pd, "np": np, "str": str},
    )  # noqa: S307
    if not isinstance(result, pd.Series):
        raise ValueError(f"expression must return a pandas Series: {expr!r}")
    return result.fillna(False).astype(bool)


def iter_report_evidences(report: Report) -> Iterable[Evidence]:
    for finding in report.findings:
        yield from finding.evidences
    for chart in report.charts:
        for datum in chart.data:
            if datum.evidence is not None:
                yield datum.evidence
            for series in datum.series:
                yield series.evidence
    for citation in report.citations:
        if citation.target.evidence is not None:
            yield citation.target.evidence


def extract_refs(
    df: pd.DataFrame,
    mask: pd.Series,
    identity: SurveyIdentityPolicy,
) -> list[str]:
    for id_column in (
        identity.respondent_id_column,
        *identity.fallback_respondent_id_columns,
    ):
        if id_column in df.columns:
            return [str(value) for value in df.loc[mask, id_column].tolist()]
    return [str(index) for index in df.index[mask].tolist()]


def extract_response_refs(
    df: pd.DataFrame,
    mask: pd.Series,
    identity: SurveyIdentityPolicy,
    question_id: str | None,
) -> list[ResponseRef]:
    respondent_ids = extract_refs(df, mask, identity)
    return [
        ResponseRef(respondent_id=respondent_id, question_id=question_id)
        for respondent_id in respondent_ids
    ]


def extract_preview(
    df: pd.DataFrame, mask: pd.Series, evidence: Evidence, limit: int = 8
) -> list[str]:
    if evidence.source_column is None or evidence.source_column not in df.columns:
        return []
    preview: list[str] = []
    for raw in df.loc[mask, evidence.source_column].tolist():
        preview.extend(render_match(raw, evidence.match_label))
        if len(preview) >= limit:
            break
    return preview[:limit]


def extract_response_previews(
    df: pd.DataFrame,
    mask: pd.Series,
    evidence: Evidence,
) -> list[str]:
    if evidence.source_column is None or evidence.source_column not in df.columns:
        return ["" for _ in df.index[mask].tolist()]

    previews: list[str] = []
    for raw in df.loc[mask, evidence.source_column].tolist():
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            previews.append("")
            continue
        multi_values = extract_multi_value_preview(raw)
        if evidence.match_label is not None and multi_values is not None:
            matched = [
                value
                for value in multi_values
                if normalize_token(value) == normalize_token(evidence.match_label)
            ]
            previews.append(
                "; ".join(item for item in matched if item.strip())
                or "; ".join(item for item in multi_values if item.strip())
            )
            continue
        if multi_values is not None:
            previews.append(
                "; ".join(
                    str(value).strip() for value in multi_values if str(value).strip()
                )
            )
            continue
        text = str(raw).strip()
        previews.append(text)
    return previews


def extract_multi_value_preview(raw: Any) -> list[str] | None:
    if isinstance(raw, list):
        values = [str(value).strip() for value in raw if str(value).strip()]
        return values or None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            values = [str(value).strip() for value in parsed if str(value).strip()]
            return values or None
    if any(separator in text for separator in [";", "|"]):
        values = [part.strip() for part in re.split(r"[;|]", text) if part.strip()]
        return values or None
    if "," in text:
        values = [part.strip() for part in text.split(",") if part.strip()]
        if values and all(
            len(value) <= 40 and len(value.split()) <= 5 for value in values
        ):
            return values
    return None


def render_match(raw: Any, match_label: str | None) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if match_label is None:
        text = str(raw).strip()
        return [text] if text else []
    normalized_label = normalize_token(match_label)
    values = coerce_multi_values(raw)
    matched = [value for value in values if normalize_token(value) == normalized_label]
    if matched:
        return matched
    text = str(raw).strip()
    if normalized_label in normalize_token(text):
        return [match_label]
    return []


def coerce_multi_values(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(value).strip() for value in raw if str(value).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(value).strip() for value in parsed if str(value).strip()]
    if any(separator in text for separator in [";", ",", "|"]):
        cleaned = [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]
        if cleaned:
            return cleaned
    return [text]


def normalize_token(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def normalize_cited_percentages_markdown(markdown: str) -> str:
    def _rebuild(match: re.Match[str]) -> str:
        pct = match.group("pct")
        marker = match.group("marker")
        return f"**{pct}**{marker}"

    normalized = re.sub(
        r"(?P<open>\*\*|__|\*|_)(?P<pct>\d+(?:[\.,]\d+)?%)(?P<marker>\[ct:[A-Za-z0-9_-]+\])(?P<close>\*\*|__|\*|_)",
        _rebuild,
        markdown,
    )
    normalized = re.sub(
        r"(?P<open>\*\*|__|\*|_)?(?P<pct>\d+(?:[\.,]\d+)?%)(?P<close>\*\*|__|\*|_)?\s*(?P<marker>\[ct:[A-Za-z0-9_-]+\])",
        _rebuild,
        normalized,
    )
    return normalized


def visible_percentages_with_spans(
    markdown: str,
) -> list[tuple[float, int, int]]:
    return [
        (float(match.group(1).replace(",", ".")), match.start(), match.end())
        for match in re.finditer(r"(?<!\w)(\d+(?:[\.,]\d+)?)%", markdown)
    ]


def referenced_df_columns(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    columns: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "df":
            continue
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            columns.add(slice_node.value)
    return columns


def line_bounds(text: str, index: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end == -1:
        end = len(text)
    return start, end
