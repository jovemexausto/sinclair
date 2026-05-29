from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class Evidence(BaseModel):
    base_rule: str = Field(
        description="Boolean pandas expression using df/pd/np for the denominator."
    )
    rule: str = Field(
        description="Boolean pandas expression using df/pd/np for the observed group."
    )
    reason: str = Field(
        description="Concrete human-readable explanation of what this evidence captures."
    )
    source_column: str | None = Field(
        default=None,
        description="Optional source column used to render matching responses in the UI.",
    )
    match_label: str | None = Field(
        default=None,
        description="Optional exact matched value to render for closed or multi-answer fields.",
    )
    question_id: str | None = Field(
        default=None,
        description="Source question identifier for this evidence.",
    )


class ResponseRef(BaseModel):
    respondent_id: str
    question_id: str | None = None


class Finding(BaseModel):
    claim: str = Field(
        description="Analytical claim about behavior, tension, or decision impact."
    )
    implication: str = Field(
        description="What this claim changes in interpretation or action."
    )
    evidences: list[Evidence] = Field(
        description="Evidence objects supporting the claim."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this finding.")


class ChartSeries(BaseModel):
    name: str = Field(description="Series label for grouped charts.")
    value_pct: float | None = Field(
        default=None, description="Published percentage, if shown."
    )
    evidence: Evidence = Field(description="Evidence backing this series datum.")
    datum_id: str | None = Field(
        default=None, description="Canonical datum id for interactive hydration."
    )
    evidence_id: str | None = Field(
        default=None, description="Canonical evidence id for interactive hydration."
    )


class ChartDatum(BaseModel):
    label: str = Field(description="Human chart label.")
    value_pct: float | None = Field(
        default=None, description="Published percentage, if shown."
    )
    evidence: Evidence | None = Field(
        default=None, description="Evidence backing this datum."
    )
    datum_id: str | None = Field(
        default=None, description="Canonical datum id for interactive hydration."
    )
    evidence_id: str | None = Field(
        default=None, description="Canonical evidence id for interactive hydration."
    )
    series: list[ChartSeries] = Field(
        default_factory=list, description="Grouped-bar series."
    )


class Chart(BaseModel):
    type: Literal["bar", "grouped_bar"] = Field(description="Allowed chart type.")
    title: str = Field(description="Human-readable chart title.")
    slug: str | None = Field(
        default=None,
        description="Canonical chart slug for markdown anchors like [[chart:slug]].",
    )
    nr_questao: str | None = Field(
        default=None, description="Question identifier, if known."
    )
    unit: str = Field(default="%", description="Short unit token, e.g. %.")
    data: list[ChartDatum] = Field(
        description="Chart data with evidence-backed datums."
    )


class CitationTarget(BaseModel):
    kind: Literal["chart_datum", "response_set", "response"] = Field(
        description="Citation target type."
    )
    nr_questao: str | None = Field(
        default=None, description="Question id for response targets."
    )
    evidence: Evidence | None = Field(
        default=None, description="Evidence for response_set targets."
    )
    evidence_id: str | None = Field(
        default=None, description="Canonical evidence id for interactive hydration."
    )
    datum_id: str | None = Field(
        default=None, description="Canonical chart datum id for interactive hydration."
    )
    chart_index: int | None = Field(
        default=None, description="Chart index for chart_datum targets."
    )
    label: str | None = Field(
        default=None, description="Chart datum label for chart_datum targets."
    )
    series: str | None = Field(
        default=None, description="Grouped chart series name, if relevant."
    )


class Citation(BaseModel):
    citation_id: str = Field(description="Stable citation id.")
    anchor_text: str | None = Field(
        default=None,
        description="Optional literal text span to bind in markdown for user-facing links.",
    )
    occurrence: int = Field(
        default=1,
        ge=1,
        description="Which occurrence of anchor_text to bind when anchor_text is used.",
    )
    target: CitationTarget = Field(description="Where this citation points.")


class Report(BaseModel):
    markdown: str = Field(description="Long-form analytical narrative in markdown.")
    findings: list[Finding] = Field(
        description="Machine-readable claims used for synthesis and chat."
    )
    citations: list[Citation] = Field(
        description="Anchors for claims, quotes, or response sets."
    )
    charts: list[Chart] = Field(description="Evidence-backed chart payloads.")


class EvidenceRecord(BaseModel):
    evidence_id: str
    scope: str | None = None
    evidence: Evidence
    base_count: int
    match_count: int
    value_pct: float
    refs: list[str] = Field(default_factory=list)
    response_refs: list[ResponseRef] = Field(default_factory=list)
    preview: list[str] = Field(default_factory=list)


class FindingRecord(BaseModel):
    finding_id: str
    scope: str | None = None
    claim: str
    implication: str
    confidence: float
    evidence_ids: list[str] = Field(default_factory=list)
