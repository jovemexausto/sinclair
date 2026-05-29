from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from ._helpers import line_bounds, visible_percentages_with_spans
from .models import (
    Chart,
    ChartDatum,
    ChartSeries,
    Citation,
    Report,
    ResponseRef,
)
from .provenance import (
    find_anchor_span,
    find_citation_marker_span_optional,
    list_chart_anchors,
    normalize_chart_slug,
    strip_citation_markers,
)
from .store import SurveyArtifactStore


class FrontendReference(BaseModel):
    ref_id: str
    kind: str
    citation_id: str | None = None
    datum_id: str | None = None
    evidence_id: str | None = None
    response_set_ref_id: str | None = None
    refs: list[str] = Field(default_factory=list)
    preview: list[str] = Field(default_factory=list)
    rationale: str | None = None
    nr_questao: str | None = None


class FrontendChartBlock(BaseModel):
    slug: str
    payload: dict[str, Any]


class FrontendRenderBundle(BaseModel):
    markdown: str
    chart_blocks: list[FrontendChartBlock] = Field(default_factory=list)
    references: dict[str, FrontendReference] = Field(default_factory=dict)


class FrontendEvidencePage(BaseModel):
    class Item(BaseModel):
        position: int
        respondent_id: str | None = None
        preview: str | None = None
        response_ref: ResponseRef | None = None

    ref_id: str
    evidence_id: str
    nr_questao: str | None = None
    rationale: str | None = None
    formula: str | None = None
    match_count: int | None = None
    base_count: int | None = None
    value_pct: float | None = None
    total_refs: int
    offset: int
    limit: int
    refs: list[str] = Field(default_factory=list)
    response_refs: list[ResponseRef] = Field(default_factory=list)
    preview: list[str] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)


class FrontendProvenance(BaseModel):
    ref_id: str
    reference: FrontendReference
    chart_slug: str | None = None
    chart_title: str | None = None
    datum: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None


class SurveyFrontendController:
    def __init__(
        self,
        *,
        store: SurveyArtifactStore,
        reports: dict[str, Report] | None = None,
    ) -> None:
        self.store = store
        self._reports = reports or {}
        self._bundles: dict[str, FrontendRenderBundle] = {}

    def register_report(self, report_key: str, report: Report) -> None:
        self._reports[report_key] = report
        self._bundles.pop(report_key, None)

    def render_report(self, report_key: str) -> FrontendRenderBundle:
        if report_key in self._bundles:
            return self._bundles[report_key]
        report = self._reports.get(report_key)
        if report is None:
            raise KeyError(f"unknown report key: {report_key}")
        bundle = hydrate_report_for_frontend(report, self.store)
        self._bundles[report_key] = bundle
        return bundle

    def get_reference(self, report_key: str, ref_id: str) -> FrontendReference:
        bundle = self.render_report(report_key)
        reference = bundle.references.get(ref_id)
        if reference is None:
            raise KeyError(f"unknown ref id for {report_key}: {ref_id}")
        return reference

    def get_provenance(self, report_key: str, ref_id: str) -> FrontendProvenance:
        bundle = self.render_report(report_key)
        reference = self.get_reference(report_key, ref_id)
        evidence = None
        if reference.evidence_id is not None:
            record = _require_evidence_record(reference.evidence_id, self.store)
            evidence = {
                "evidence_id": record.evidence_id,
                "nr_questao": record.evidence.question_id,
                "formula": f"{record.match_count}/{record.base_count}",
                "base_rule": record.evidence.base_rule,
                "rule": record.evidence.rule,
                "rationale": record.evidence.reason,
                "base_count": record.base_count,
                "match_count": record.match_count,
                "value_pct": record.value_pct,
                "source_column": record.evidence.source_column,
                "match_label": record.evidence.match_label,
            }
        chart_slug, chart_title, datum = _resolve_datum_payload(
            bundle, reference.datum_id
        )
        return FrontendProvenance(
            ref_id=ref_id,
            reference=reference,
            chart_slug=chart_slug,
            chart_title=chart_title,
            datum=datum,
            evidence=evidence,
        )

    def get_evidence_page(
        self,
        report_key: str,
        ref_id: str,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> FrontendEvidencePage:
        reference = self.get_reference(report_key, ref_id)
        evidence_id = _resolve_evidence_id_from_reference(reference)
        record = _require_evidence_record(evidence_id, self.store)
        visible_items = [
            (ref, response_ref, preview)
            for ref, response_ref, preview in zip(
                record.refs,
                record.response_refs,
                record.preview,
                strict=False,
            )
            if str(preview or "").strip()
        ]
        page_slice = visible_items[offset : offset + limit]
        page_refs = [ref for ref, _, _ in page_slice]
        page_response_refs = [response_ref for _, response_ref, _ in page_slice]
        page_preview = [preview for _, _, preview in page_slice]
        return FrontendEvidencePage(
            ref_id=ref_id,
            evidence_id=evidence_id,
            nr_questao=record.evidence.question_id,
            rationale=record.evidence.reason,
            formula=f"{record.match_count}/{record.base_count}",
            match_count=record.match_count,
            base_count=record.base_count,
            value_pct=record.value_pct,
            total_refs=len(visible_items),
            offset=offset,
            limit=limit,
            refs=page_refs,
            response_refs=page_response_refs,
            preview=page_preview,
            items=[
                FrontendEvidencePage.Item(
                    position=offset + index + 1,
                    respondent_id=response_ref.respondent_id if response_ref else None,
                    preview=page_preview[index] if index < len(page_preview) else None,
                    response_ref=response_ref,
                )
                for index, response_ref in enumerate(page_response_refs)
            ],
        )


def hydrate_report_for_frontend(
    report: Report,
    store: SurveyArtifactStore,
) -> FrontendRenderBundle:
    references: dict[str, FrontendReference] = {}
    chart_blocks = [
        FrontendChartBlock(
            slug=_require_chart_slug(chart),
            payload=_chart_payload(chart, store, references),
        )
        for chart in report.charts
    ]
    markdown = _hydrate_markdown_links(report, store, references)
    chart_registry = {block.slug: block for block in chart_blocks}
    for slug in list_chart_anchors(markdown):
        block = chart_registry.get(slug)
        if block is None:
            continue
        markdown = markdown.replace(
            f"[[chart:{slug}]]",
            "```chart\n" + json.dumps(block.payload, ensure_ascii=False) + "\n```",
        )
    return FrontendRenderBundle(
        markdown=markdown,
        chart_blocks=chart_blocks,
        references=references,
    )


def _hydrate_markdown_links(
    report: Report,
    store: SurveyArtifactStore,
    references: dict[str, FrontendReference],
) -> str:
    markdown = report.markdown
    percentages = visible_percentages_with_spans(markdown)
    replacements: list[tuple[int, int, str]] = []
    for citation in report.citations:
        ref_id = _line_binding_ref(citation, store, references)
        marker_span = find_citation_marker_span_optional(markdown, citation.citation_id)
        if marker_span is not None:
            citation_replacements = _replace_marker_with_link(
                markdown, citation, ref_id, marker_span, percentages
            )
            if citation_replacements:
                replacements.extend(citation_replacements)
                continue
        if citation.anchor_text is not None:
            anchor_start, anchor_end = find_anchor_span(
                markdown, citation.anchor_text, citation.occurrence
            )
            link_text = markdown[anchor_start:anchor_end]
            replacements.append(
                (anchor_start, anchor_end, _markdown_link(link_text, ref_id))
            )
    for start, end, replacement in sorted(replacements, reverse=True):
        markdown = markdown[:start] + replacement + markdown[end:]
    return strip_citation_markers(markdown)


def _replace_marker_with_link(
    markdown: str,
    citation: Citation,
    ref_id: str,
    marker_span: tuple[int, int],
    percentages: list[tuple[float, int, int]],
) -> list[tuple[int, int, str]]:
    marker_start, marker_end = marker_span
    if citation.target.kind == "chart_datum":
        pct_start, pct_end = _percentage_span_before_marker(
            markdown, marker_start, percentages
        )
        if pct_start is not None and pct_end is not None:
            link_start, link_end = _percentage_link_span(
                markdown,
                pct_start,
                pct_end,
                marker_start,
            )
            percentage = markdown[link_start:link_end]
            return [
                (marker_start, marker_end, ""),
                (link_start, link_end, _markdown_link(percentage, ref_id)),
            ]
    if citation.anchor_text is not None:
        anchor_start, anchor_end = find_anchor_span(
            markdown, citation.anchor_text, citation.occurrence
        )
        link_text = markdown[anchor_start:anchor_end]
        return [
            (marker_start, marker_end, ""),
            (anchor_start, anchor_end, _markdown_link(link_text, ref_id)),
        ]
    return [(marker_start, marker_end, "")]


def _chart_payload(
    chart: Chart,
    store: SurveyArtifactStore,
    references: dict[str, FrontendReference],
) -> dict[str, Any]:
    slug = _require_chart_slug(chart)
    payload = {
        "type": chart.type,
        "chart_id": slug,
        "title": chart.title,
        "nr_questao": chart.nr_questao,
        "unit": chart.unit,
        "data": [
            _chart_datum_payload(chart, datum, store, references)
            for datum in chart.data
        ],
    }
    if chart.type == "bar":
        payload["orientation"] = "horizontal"
        payload["sort"] = "desc"
        payload["base"] = _chart_base(chart, store)
    return payload


def _chart_datum_payload(
    chart: Chart,
    datum: ChartDatum,
    store: SurveyArtifactStore,
    references: dict[str, FrontendReference],
) -> dict[str, Any]:
    if datum.series:
        return {
            "datum_id": datum.datum_id,
            "label_key": normalize_chart_slug(datum.label),
            "label": datum.label,
            "nr_questao": chart.nr_questao,
            "value": datum.value_pct,
            "series": [
                _chart_series_payload(chart, datum, series, store, references)
                for series in datum.series
            ],
        }
    record = _require_evidence_record(datum.evidence_id, store)
    datum_ref_id = _datum_binding_ref(
        datum_id=datum.datum_id,
        evidence_id=record.evidence_id,
        nr_questao=record.evidence.question_id,
        rationale=record.evidence.reason,
        references=references,
    )
    response_ref_id = _response_set_ref(record.evidence_id)
    references.setdefault(
        response_ref_id,
        FrontendReference(
            ref_id=response_ref_id,
            kind="response_set",
            evidence_id=record.evidence_id,
            refs=record.refs,
            preview=record.preview[:8],
            rationale=record.evidence.reason,
            nr_questao=record.evidence.question_id,
        ),
    )
    return {
        "datum_id": datum.datum_id,
        "label_key": normalize_chart_slug(datum.label),
        "label": datum.label,
        "nr_questao": record.evidence.question_id or chart.nr_questao,
        "value": datum.value_pct,
        "base": record.base_count,
        "rationale": record.evidence.reason,
        "refs": record.refs,
        "refId": datum_ref_id,
        "responseSetRefId": response_ref_id,
    }


def _chart_series_payload(
    chart: Chart,
    datum: ChartDatum,
    series: ChartSeries,
    store: SurveyArtifactStore,
    references: dict[str, FrontendReference],
) -> dict[str, Any]:
    record = _require_evidence_record(series.evidence_id, store)
    datum_ref_id = _datum_binding_ref(
        datum_id=series.datum_id,
        evidence_id=record.evidence_id,
        nr_questao=record.evidence.question_id,
        rationale=record.evidence.reason,
        references=references,
    )
    response_ref_id = _response_set_ref(record.evidence_id)
    references.setdefault(
        response_ref_id,
        FrontendReference(
            ref_id=response_ref_id,
            kind="response_set",
            evidence_id=record.evidence_id,
            refs=record.refs,
            preview=record.preview[:8],
            rationale=record.evidence.reason,
            nr_questao=record.evidence.question_id,
        ),
    )
    return {
        "datum_id": series.datum_id,
        "label_key": normalize_chart_slug(datum.label),
        "label": datum.label,
        "series_key": normalize_chart_slug(series.name),
        "series": series.name,
        "nr_questao": record.evidence.question_id or chart.nr_questao,
        "value": series.value_pct,
        "base": record.base_count,
        "rationale": record.evidence.reason,
        "refs": record.refs,
        "refId": datum_ref_id,
        "responseSetRefId": response_ref_id,
    }


def _line_binding_ref(
    citation: Citation,
    store: SurveyArtifactStore,
    references: dict[str, FrontendReference],
) -> str:
    if citation.target.kind == "chart_datum":
        evidence_id = citation.target.evidence_id
        datum_id = citation.target.datum_id
        response_set_ref_id = (
            _response_set_ref(evidence_id or "") if evidence_id else None
        )
        if evidence_id is not None:
            record = _require_evidence_record(evidence_id, store)
            if response_set_ref_id is not None:
                references.setdefault(
                    response_set_ref_id,
                    FrontendReference(
                        ref_id=response_set_ref_id,
                        kind="response_set",
                        evidence_id=record.evidence_id,
                        refs=record.refs,
                        preview=record.preview[:8],
                        rationale=record.evidence.reason,
                        nr_questao=record.evidence.question_id,
                    ),
                )
            rationale = record.evidence.reason
            nr_questao = record.evidence.question_id
        else:
            rationale = None
            nr_questao = citation.target.nr_questao
        ref_id = _stable_ref_id(
            "rf_lb", citation.citation_id, datum_id or evidence_id or ""
        )
        references.setdefault(
            ref_id,
            FrontendReference(
                ref_id=ref_id,
                kind="line_binding",
                citation_id=citation.citation_id,
                datum_id=datum_id,
                evidence_id=evidence_id,
                response_set_ref_id=response_set_ref_id,
                rationale=rationale,
                nr_questao=nr_questao,
            ),
        )
        return ref_id

    evidence_id = citation.target.evidence_id
    if evidence_id is None:
        raise ValueError(
            f"citation {citation.citation_id!r} is missing evidence provenance"
        )
    record = _require_evidence_record(evidence_id, store)
    response_set_ref_id = _response_set_ref(evidence_id)
    references.setdefault(
        response_set_ref_id,
        FrontendReference(
            ref_id=response_set_ref_id,
            kind="response_set",
            evidence_id=record.evidence_id,
            refs=record.refs,
            preview=record.preview,
            rationale=record.evidence.reason,
            nr_questao=record.evidence.question_id,
        ),
    )
    ref_id = _stable_ref_id("rf_lb", citation.citation_id, evidence_id)
    references.setdefault(
        ref_id,
        FrontendReference(
            ref_id=ref_id,
            kind="line_binding",
            citation_id=citation.citation_id,
            evidence_id=evidence_id,
            response_set_ref_id=response_set_ref_id,
            rationale=record.evidence.reason,
            nr_questao=record.evidence.question_id,
        ),
    )
    return ref_id


def _chart_base(chart: Chart, store: SurveyArtifactStore) -> int | None:
    for datum in chart.data:
        evidence_id = datum.evidence_id
        if evidence_id:
            return _require_evidence_record(evidence_id, store).base_count
        for series in datum.series:
            if series.evidence_id:
                return _require_evidence_record(series.evidence_id, store).base_count
    return None


def _require_evidence_record(evidence_id: str | None, store: SurveyArtifactStore):
    if evidence_id is None:
        raise ValueError("missing evidence_id for frontend hydration")
    record = store.get_evidence(evidence_id)
    if record is None:
        raise ValueError(f"missing evidence record in store: {evidence_id}")
    return record


def _require_chart_slug(chart: Chart) -> str:
    if not chart.slug:
        raise ValueError("chart slug is required for frontend hydration")
    return chart.slug


def _response_set_ref(evidence_id: str) -> str:
    return _stable_ref_id("rf_rs", evidence_id)


def _datum_binding_ref(
    *,
    datum_id: str | None,
    evidence_id: str,
    nr_questao: str | None,
    rationale: str | None,
    references: dict[str, FrontendReference],
) -> str:
    ref_id = _stable_ref_id("rf_dt", datum_id or evidence_id)
    references.setdefault(
        ref_id,
        FrontendReference(
            ref_id=ref_id,
            kind="datum_binding",
            datum_id=datum_id,
            evidence_id=evidence_id,
            response_set_ref_id=_response_set_ref(evidence_id),
            rationale=rationale,
            nr_questao=nr_questao,
        ),
    )
    return ref_id


def _resolve_evidence_id_from_reference(reference: FrontendReference) -> str:
    if reference.kind == "response_set":
        if reference.evidence_id is None:
            raise ValueError("response_set reference is missing evidence_id")
        return reference.evidence_id
    if reference.response_set_ref_id is not None:
        if reference.evidence_id is None:
            raise ValueError("line binding reference is missing evidence_id")
        return reference.evidence_id
    if reference.evidence_id is None:
        raise ValueError("reference is missing evidence provenance")
    return reference.evidence_id


def _resolve_datum_payload(
    bundle: FrontendRenderBundle, datum_id: str | None
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    if datum_id is None:
        return None, None, None
    for chart_block in bundle.chart_blocks:
        datum = _find_datum_payload(chart_block.payload.get("data", []), datum_id)
        if datum is not None:
            return chart_block.slug, chart_block.payload.get("title"), datum
    return None, None, None


def _find_datum_payload(
    data: list[dict[str, Any]], datum_id: str
) -> dict[str, Any] | None:
    for datum in data:
        if datum.get("datum_id") == datum_id:
            return datum
        for series in datum.get("series", []):
            if series.get("datum_id") == datum_id:
                return series
    return None


def _stable_ref_id(prefix: str, *parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:24]}"


def _percentage_span_before_marker(
    markdown: str,
    marker_start: int,
    percentages: list[tuple[float, int, int]],
) -> tuple[int | None, int | None]:
    line_start, line_end = line_bounds(markdown, marker_start)
    matches = [
        (start, end)
        for _, start, end in percentages
        if line_start <= start < marker_start <= line_end and end <= marker_start
    ]
    if not matches:
        return None, None
    return max(matches, key=lambda item: item[1])


def _percentage_link_span(
    markdown: str,
    pct_start: int,
    pct_end: int,
    marker_start: int,
) -> tuple[int, int]:
    for marker in ("**", "__", "*", "_"):
        start = pct_start - len(marker)
        end = pct_end + len(marker)
        if start < 0 or end > marker_start:
            continue
        if markdown[start:pct_start] == marker and markdown[pct_end:end] == marker:
            return start, end
    return pct_start, pct_end


def _markdown_link(text: str, ref_id: str) -> str:
    return f"[{text}](ref:{ref_id})"
