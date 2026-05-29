from __future__ import annotations

import hashlib
import json
import re

from .models import Evidence

_CITATION_MARKER_RE = re.compile(r"\[ct:([A-Za-z0-9_-]+)\]")
_CHART_ANCHOR_RE = re.compile(r"\[\[chart:([A-Za-z0-9_-]+)\]\]")


def citation_marker(citation_id: str) -> str:
    return f"[ct:{citation_id}]"


def find_citation_marker_span(markdown: str, citation_id: str) -> tuple[int, int]:
    marker = citation_marker(citation_id)
    matches = list(re.finditer(re.escape(marker), markdown))
    if not matches:
        raise ValueError(f"citation marker not found in markdown: {marker}")
    if len(matches) > 1:
        raise ValueError(f"citation marker must appear exactly once: {marker}")
    match = matches[0]
    return match.start(), match.end()


def find_citation_marker_span_optional(
    markdown: str, citation_id: str
) -> tuple[int, int] | None:
    marker = citation_marker(citation_id)
    matches = list(re.finditer(re.escape(marker), markdown))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"citation marker must appear exactly once: {marker}")
    match = matches[0]
    return match.start(), match.end()


def find_anchor_span(
    markdown: str, anchor_text: str, occurrence: int
) -> tuple[int, int]:
    matches = list(re.finditer(re.escape(anchor_text), markdown))
    if len(matches) < occurrence:
        raise ValueError(f"anchor_text not found in markdown: {anchor_text!r}")
    match = matches[occurrence - 1]
    return match.start(), match.end()


def list_citation_markers(markdown: str) -> list[str]:
    return [match.group(1) for match in _CITATION_MARKER_RE.finditer(markdown)]


def strip_citation_markers(markdown: str) -> str:
    stripped = _CITATION_MARKER_RE.sub("", markdown)
    stripped = re.sub(r"\s+([),.;:])", r"\1", stripped)
    return re.sub(r" {2,}", " ", stripped)


def chart_anchor(chart_slug: str) -> str:
    return f"[[chart:{chart_slug}]]"


def list_chart_anchors(markdown: str) -> list[str]:
    return [match.group(1) for match in _CHART_ANCHOR_RE.finditer(markdown)]


def stable_chart_slug(*, chart_index: int, title: str, nr_questao: str | None) -> str:
    base = normalize_chart_slug(nr_questao or title) or f"chart-{chart_index + 1}"
    return f"{base}-{chart_index + 1}"


def normalize_chart_anchors(markdown: str, chart_slugs: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.isdigit():
            index = int(token) - 1
            if 0 <= index < len(chart_slugs):
                return chart_anchor(chart_slugs[index])
        if token in chart_slugs:
            return chart_anchor(token)
        return match.group(0)

    return _CHART_ANCHOR_RE.sub(replace, markdown)


def stable_evidence_id(evidence: Evidence) -> str:
    payload = json.dumps(
        evidence.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
    )
    return "ev_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def stable_datum_id(
    *, chart_index: int, label: str, series: str | None, evidence_id: str
) -> str:
    payload = json.dumps(
        {
            "chart_index" : chart_index,
            "label"       : label,
            "series"      : series,
            "evidence_id" : evidence_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "dt_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def normalize_chart_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.casefold()).strip("-")
    return re.sub(r"-{2,}", "-", slug)
