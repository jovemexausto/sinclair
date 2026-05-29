from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel

from .config import SurveyValidationPolicy
from ._helpers import (
    eval_mask,
    line_bounds,
    normalize_cited_percentages_markdown,
    referenced_df_columns,
    visible_percentages_with_spans,
)
from .models import Evidence, Report
from .provenance import (
    chart_anchor,
    find_anchor_span,
    find_citation_marker_span_optional,
    list_chart_anchors,
    normalize_chart_slug,
    normalize_chart_anchors,
    stable_chart_slug,
    list_citation_markers,
    stable_datum_id,
    stable_evidence_id,
)


def validate_report(
    report: BaseModel,
    df: pd.DataFrame,
    policy: SurveyValidationPolicy | None = None,
    analysis_question_id: str | None = None,
) -> None:
    policy = policy or SurveyValidationPolicy()

    if not isinstance(report, Report):
        raise ValueError(f"expected Report, got {type(report).__name__}")
    report.markdown = normalize_cited_percentages_markdown(report.markdown)
    if len(report.markdown.strip()) < policy.min_markdown_chars:
        raise ValueError("markdown is too short")
    if policy.require_sections and "##" not in report.markdown:
        raise ValueError("markdown must include at least one ## section")
    if len(report.findings) < policy.min_findings:
        raise ValueError("report must include at least one finding")
    if len(report.charts) < policy.min_charts:
        raise ValueError(f"report must include at least {policy.min_charts} chart(s)")
    if len(report.citations) < policy.min_citations:
        raise ValueError(
            f"report must include at least {policy.min_citations} citation(s)"
        )

    for finding in report.findings:
        if not finding.claim.strip() or not finding.implication.strip():
            raise ValueError("finding claim and implication are required")
        if not finding.evidences:
            raise ValueError("finding must include evidence")
        for evidence in finding.evidences:
            validate_evidence(evidence, df)

    chart_refs, legacy_chart_refs = _validate_charts(report, df, policy)
    if analysis_question_id is not None:
        _validate_question_scope(report, analysis_question_id)
    _validate_chart_anchors(report)
    percentages = visible_percentages_with_spans(report.markdown)
    if policy.require_percent_citations:
        cited_spans = _validate_citations(report, df, chart_refs, legacy_chart_refs)
        for pct in percentages:
            if pct not in cited_spans:
                raise ValueError(
                    f"visible percentage {pct[0]}% must be disambiguated by a citation to a chart datum"
                )


def validate_report_question_scope(report: Report, question_id: str) -> None:
    for finding in report.findings:
        for evidence in finding.evidences:
            _validate_evidence_scope(
                evidence.question_id, evidence.source_column, question_id
            )
    for chart in report.charts:
        _validate_question_field(chart.nr_questao, question_id)
        for datum in chart.data:
            if datum.evidence is not None:
                _validate_evidence_scope(
                    datum.evidence.question_id,
                    datum.evidence.source_column,
                    question_id,
                )
            for series in datum.series:
                _validate_evidence_scope(
                    series.evidence.question_id,
                    series.evidence.source_column,
                    question_id,
                )
    for citation in report.citations:
        _validate_question_field(citation.target.nr_questao, question_id)
        if citation.target.evidence is not None:
            _validate_evidence_scope(
                citation.target.evidence.question_id,
                citation.target.evidence.source_column,
                question_id,
            )


def validate_evidence(evidence: Evidence, df: pd.DataFrame) -> None:
    if not evidence.reason.strip():
        raise ValueError("evidence.reason cannot be empty")
    _validate_evidence_reason(evidence)
    base_mask = eval_mask(evidence.base_rule, df)
    rule_mask = eval_mask(evidence.rule, df)
    if not base_mask.any():
        raise ValueError(f"base_rule matched no rows: {evidence.base_rule!r}")
    if (rule_mask & ~base_mask).any():
        raise ValueError("rule must be an explicit subset of base_rule")


def canonicalize_evidence_reason(evidence: Evidence) -> Evidence:
    if _reason_has_required_context(evidence):
        return evidence
    question_token = str(evidence.question_id or evidence.source_column or "").strip()
    if not question_token:
        return evidence
    label = str(evidence.match_label or "").strip()
    reason = (
        f"Ao responder {question_token}, menciona {label}."
        if label
        else f"Ao responder {question_token}, entra no recorte observado."
    )
    return evidence.model_copy(update={"reason": reason})


def _validate_evidence_reason(evidence: Evidence) -> None:
    if _reason_has_required_context(evidence):
        return
    normalized_reason = " ".join(evidence.reason.casefold().split())
    scope_tokens = [evidence.question_id, evidence.source_column]
    if any(scope_tokens):
        normalized_scope_tokens = [
            " ".join(str(token).casefold().split())
            for token in scope_tokens
            if str(token or "").strip()
        ]
        if normalized_scope_tokens and not any(
            token in normalized_reason for token in normalized_scope_tokens
        ):
            raise ValueError(
                "evidence.reason must mention the source question or column"
            )
    if evidence.match_label is not None:
        normalized_match_label = " ".join(evidence.match_label.casefold().split())
        if normalized_match_label not in normalized_reason:
            raise ValueError(
                "evidence.reason must mention the matched label when match_label is set"
            )


def _reason_has_required_context(evidence: Evidence) -> bool:
    normalized_reason = " ".join(evidence.reason.casefold().split())
    scope_tokens = [evidence.question_id, evidence.source_column]
    if any(scope_tokens):
        normalized_scope_tokens = [
            " ".join(str(token).casefold().split())
            for token in scope_tokens
            if str(token or "").strip()
        ]
        if normalized_scope_tokens and not any(
            token in normalized_reason for token in normalized_scope_tokens
        ):
            return False
    if evidence.match_label is not None:
        normalized_match_label = " ".join(evidence.match_label.casefold().split())
        if normalized_match_label not in normalized_reason:
            return False
    return True


def validate_datum(
    label: str, value_pct: float | None, evidence: Evidence, df: pd.DataFrame
) -> float | None:
    validate_evidence(evidence, df)
    if value_pct is None:
        return None
    base_mask = eval_mask(evidence.base_rule, df)
    rule_mask = eval_mask(evidence.rule, df)
    expected_pct = round(
        int((base_mask & rule_mask).sum()) / int(base_mask.sum()) * 100, 1
    )
    if abs(value_pct - expected_pct) > 0.1:
        raise ValueError(f"{label!r}: expected {expected_pct}, got {value_pct}")
    return float(value_pct)


def _validate_question_scope(report: Report, analysis_question_id: str) -> None:
    for finding in report.findings:
        for evidence in finding.evidences:
            _validate_question_scoped_evidence(evidence, analysis_question_id)

    for chart in report.charts:
        if chart.nr_questao != analysis_question_id:
            raise ValueError(f"chart provenance must stay on {analysis_question_id!r}")
        for datum in chart.data:
            if datum.evidence is not None:
                _validate_question_scoped_evidence(datum.evidence, analysis_question_id)
            for series in datum.series:
                _validate_question_scoped_evidence(
                    series.evidence, analysis_question_id
                )

    for citation in report.citations:
        target_question_id = citation.target.nr_questao
        if (
            target_question_id is not None
            and target_question_id != analysis_question_id
        ):
            raise ValueError(
                f"citation provenance must stay on {analysis_question_id!r}"
            )
        if citation.target.evidence is not None:
            _validate_question_scoped_evidence(
                citation.target.evidence, analysis_question_id
            )


def _validate_question_scoped_evidence(
    evidence: Evidence, analysis_question_id: str
) -> None:
    if (
        evidence.question_id is not None
        and evidence.question_id != analysis_question_id
    ):
        raise ValueError(f"evidence provenance must stay on {analysis_question_id!r}")
    if (
        evidence.source_column is not None
        and evidence.source_column != analysis_question_id
    ):
        raise ValueError(f"evidence provenance must stay on {analysis_question_id!r}")
    rule_columns = referenced_df_columns(evidence.rule)
    if rule_columns - {analysis_question_id}:
        raise ValueError(f"rule provenance must stay on {analysis_question_id!r}")


def _validate_charts(
    report: Report,
    df: pd.DataFrame,
    policy: SurveyValidationPolicy,
) -> tuple[
    dict[str, dict[str, float | str | None]],
    dict[tuple[int, str, str | None], str],
]:
    chart_refs: dict[str, dict[str, float | str | None]] = {}
    legacy_chart_refs: dict[tuple[int, str, str | None], str] = {}
    for chart_index, chart in enumerate(report.charts):
        if chart.type not in policy.allowed_chart_types:
            raise ValueError(f"unsupported chart type: {chart.type!r}")
        if chart.unit != policy.chart_unit:
            raise ValueError("chart unit must be '%' for percentage charts")
        if not chart.title.strip() or not chart.data:
            raise ValueError("chart title and data are required")
        _validate_human_label(chart.title, field_name="chart.title")
        chart.slug = normalize_chart_slug(chart.slug or "") or stable_chart_slug(
            chart_index=chart_index,
            title=chart.title,
            nr_questao=chart.nr_questao,
        )
        chart_scope_ids: set[str] = set()
        for datum in chart.data:
            _validate_human_label(datum.label, field_name="chart datum label")
            if datum.evidence is not None:
                datum_scope_id = _evidence_scope_id(datum.evidence)
                if datum_scope_id is None:
                    raise ValueError(
                        f"chart datum {datum.label!r} must declare a single nr_questao scope"
                    )
                chart_scope_ids.add(datum_scope_id)
                value = validate_datum(datum.label, datum.value_pct, datum.evidence, df)
                datum.evidence_id = stable_evidence_id(datum.evidence)
                datum.datum_id = stable_datum_id(
                    chart_index=chart_index,
                    label=datum.label,
                    series=None,
                    evidence_id=datum.evidence_id,
                )
                if value is not None:
                    chart_refs[datum.datum_id] = {
                        "value": value,
                        "evidence_id": datum.evidence_id,
                        "chart_index": chart_index,
                        "label": datum.label,
                        "series": None,
                    }
                    legacy_chart_refs[(chart_index, datum.label, None)] = datum.datum_id
            for series in datum.series:
                _validate_human_label(series.name, field_name="chart series name")
                series_scope_id = _evidence_scope_id(series.evidence)
                if series_scope_id is None:
                    raise ValueError(
                        f"chart series {series.name!r} must declare a single nr_questao scope"
                    )
                chart_scope_ids.add(series_scope_id)
                value = validate_datum(
                    series.name, series.value_pct, series.evidence, df
                )
                series.evidence_id = stable_evidence_id(series.evidence)
                series.datum_id = stable_datum_id(
                    chart_index=chart_index,
                    label=datum.label,
                    series=series.name,
                    evidence_id=series.evidence_id,
                )
                if value is not None:
                    chart_refs[series.datum_id] = {
                        "value": value,
                        "evidence_id": series.evidence_id,
                        "chart_index": chart_index,
                        "label": datum.label,
                        "series": series.name,
                    }
                    legacy_chart_refs[(chart_index, datum.label, series.name)] = (
                        series.datum_id
                    )
        if chart.nr_questao is not None:
            if chart_scope_ids and chart_scope_ids != {chart.nr_questao}:
                raise ValueError(
                    f"chart provenance mixes datum scopes {sorted(chart_scope_ids)!r} but declares nr_questao={chart.nr_questao!r}"
                )
        elif len(chart_scope_ids) == 1:
            chart.nr_questao = next(iter(chart_scope_ids))
    return chart_refs, legacy_chart_refs


def _validate_chart_anchors(report: Report) -> None:
    if not report.charts:
        return
    chart_slugs = [chart.slug for chart in report.charts if chart.slug is not None]
    report.markdown = normalize_chart_anchors(report.markdown, chart_slugs)
    anchors = list_chart_anchors(report.markdown)
    unknown = sorted(set(anchors) - set(chart_slugs))
    if unknown:
        raise ValueError(f"markdown includes unknown chart anchors: {unknown!r}")
    missing = [slug for slug in chart_slugs if anchors.count(slug) == 0]
    if missing:
        raise ValueError(
            "markdown must include one [[chart:slug]] anchor for each chart: "
            + ", ".join(chart_anchor(slug) for slug in missing)
        )


def _validate_citations(
    report: Report,
    df: pd.DataFrame,
    chart_refs: dict[str, dict[str, float | str | None]],
    legacy_chart_refs: dict[tuple[int, str, str | None], str],
) -> list[tuple[float, int, int]]:
    percentages = visible_percentages_with_spans(report.markdown)
    markers = list_citation_markers(report.markdown)
    duplicates = {marker for marker in markers if markers.count(marker) > 1}
    if duplicates:
        raise ValueError(
            f"citation markers must be unique in markdown: {sorted(duplicates)!r}"
        )
    unknown_markers = sorted(
        set(markers) - {citation.citation_id for citation in report.citations}
    )
    if unknown_markers:
        raise ValueError(
            f"markdown includes unknown citation markers: {unknown_markers!r}"
        )
    cited_spans: list[tuple[float, int, int]] = []
    for citation in report.citations:
        marker_span = find_citation_marker_span_optional(
            report.markdown, citation.citation_id
        )
        if citation.target.kind == "response_set":
            if citation.target.evidence is None:
                raise ValueError("response_set citation requires evidence")
            validate_evidence(citation.target.evidence, df)
            citation.target.evidence_id = stable_evidence_id(citation.target.evidence)
            if marker_span is None and citation.anchor_text is None:
                raise ValueError(
                    f"response_set citation {citation.citation_id!r} requires either an inline marker or anchor_text"
                )
            if citation.anchor_text is not None:
                find_anchor_span(
                    report.markdown, citation.anchor_text, citation.occurrence
                )
            continue
        if citation.target.kind != "chart_datum":
            continue
        datum_id = _resolve_chart_datum_target(citation, chart_refs, legacy_chart_refs)
        ref = chart_refs[datum_id]
        raw_value = ref["value"]
        if raw_value is None:
            raise ValueError(
                f"chart datum {datum_id!r} is missing a numeric value for citation resolution"
            )
        value = float(raw_value)
        citation.target.datum_id = datum_id
        citation.target.evidence_id = str(ref["evidence_id"])
        citation.target.evidence = None
        matched = None
        if marker_span is not None:
            marker_start, _ = marker_span
            matched = _find_cited_percentage_before_marker(
                report.markdown,
                percentages,
                marker_start,
                value,
            )
        elif citation.anchor_text is not None:
            anchor_start, anchor_end = find_anchor_span(
                report.markdown, citation.anchor_text, citation.occurrence
            )
            matched = _find_cited_percentage_in_anchor(
                percentages,
                anchor_start,
                anchor_end,
                value,
            )
        if matched is None:
            guidance = _format_chart_datum_citation_error(
                markdown=report.markdown,
                citation_id=citation.citation_id,
                value=value,
                marker_span=marker_span,
            )
            raise ValueError(
                f"citation {citation.citation_id!r} does not disambiguate a visible {value}% chart datum. {guidance}"
            )
        cited_spans.append(matched)
    return cited_spans


def _format_chart_datum_citation_error(
    *,
    markdown: str,
    citation_id: str,
    value: float,
    marker_span: tuple[int, int] | None,
) -> str:
    canonical = f"**{value:.1f}%**[ct:{citation_id}]"
    accepted = ", ".join(
        [
            f"{value:.1f}%[ct:{citation_id}]",
            canonical,
            f"**{value:.1f}%[ct:{citation_id}]**",
        ]
    )
    if marker_span is None:
        return (
            "Attach the citation marker to the exact visible percentage for this datum. "
            f"Accepted forms: {accepted}. Canonical form: {canonical}."
        )

    line_start, line_end = line_bounds(markdown, marker_span[0])
    line = markdown[line_start:line_end].strip()
    return (
        "Attach the marker to the exact visible percentage on the same line. "
        f"Accepted forms: {accepted}. Canonical form: {canonical}. "
        f"Current line: {line!r}"
    )


def _resolve_chart_datum_target(
    citation,
    chart_refs: dict[str, dict[str, float | str | None]],
    legacy_chart_refs: dict[tuple[int, str, str | None], str],
) -> str:
    if citation.target.datum_id is not None:
        if citation.target.datum_id in chart_refs:
            return citation.target.datum_id
    if citation.target.chart_index is not None and citation.target.label is not None:
        ref_key = (
            citation.target.chart_index,
            citation.target.label,
            citation.target.series,
        )
        datum_id = legacy_chart_refs.get(ref_key)
        if datum_id is not None:
            return datum_id

    if citation.target.evidence_id is not None:
        matches = [
            datum_id
            for datum_id, payload in chart_refs.items()
            if payload.get("evidence_id") == citation.target.evidence_id
        ]
        if len(matches) == 1:
            return matches[0]

    if citation.target.label is not None:
        matches = [
            datum_id
            for datum_id, payload in chart_refs.items()
            if payload.get("label") == citation.target.label
            and payload.get("series") == citation.target.series
        ]
        if len(matches) == 1:
            return matches[0]

    raise ValueError(
        "chart_datum citation requires a resolvable datum target via datum_id, evidence_id, or unique label. `chart_index` refers to chart position, not datum position."
    )


def _evidence_scope_id(evidence: Evidence) -> str | None:
    if evidence.question_id is not None:
        return evidence.question_id
    if evidence.source_column is not None:
        return evidence.source_column
    return None


def _find_cited_percentage_before_marker(
    markdown: str,
    percentages: list[tuple[float, int, int]],
    marker_start: int,
    value: float,
) -> tuple[float, int, int] | None:
    line_start, _ = line_bounds(markdown, marker_start)
    candidates = [
        (pct, start, end)
        for pct, start, end in percentages
        if line_start <= start < marker_start and abs(pct - value) <= 0.1
    ]
    if not candidates:
        return None
    pct, start, end = max(candidates, key=lambda item: item[2])
    between = markdown[end:marker_start]
    if not re.fullmatch(r"(?:\s|\*|_)+", between):
        return None
    return pct, start, end


def _find_cited_percentage_in_anchor(
    percentages: list[tuple[float, int, int]],
    anchor_start: int,
    anchor_end: int,
    value: float,
) -> tuple[float, int, int] | None:
    matches = [
        (pct, start, end)
        for pct, start, end in percentages
        if anchor_start <= start and end <= anchor_end and abs(pct - value) <= 0.1
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _validate_evidence_scope(
    question_id: str | None,
    source_column: str | None,
    expected_question_id: str,
) -> None:
    effective = question_id or source_column
    if effective is not None and effective != expected_question_id:
        raise ValueError(f"evidence provenance must stay on {expected_question_id!r}")


def _validate_question_field(
    question_id: str | None, expected_question_id: str
) -> None:
    if question_id is not None and question_id != expected_question_id:
        raise ValueError(f"evidence provenance must stay on {expected_question_id!r}")


_QUESTION_CODE_RE = re.compile(r"^(q|p)\d+(?:_[a-z0-9]+)?$", re.IGNORECASE)


def _validate_human_label(label: str, *, field_name: str) -> None:
    value = label.strip()
    if not value:
        raise ValueError(f"{field_name} must be human and non-empty")
    if _QUESTION_CODE_RE.match(value):
        raise ValueError(
            f"{field_name} must be human and user-facing, not a raw question code"
        )
