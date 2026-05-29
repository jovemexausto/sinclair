from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from sinclair import Agent, AgentConfig
from sinclair.survey import (
    BaselineRun,
    BenchmarkResult,
    Chart,
    ChartDatum,
    Citation,
    CitationTarget,
    EmbeddingBackend,
    Evidence,
    Finding,
    FrontendEvidencePage,
    FrontendProvenance,
    FrontendRenderBundle,
    Report,
    SurveyApp,
    SurveyArtifactStore,
    SurveyDefaults,
    SurveyFrontendController,
    SurveyIdentityPolicy,
    SurveyStudy,
    SurveyToolKit,
    SurveyValidationPolicy,
    UseCaseArtifacts,
    baseline_question_report,
    benchmark_question_report,
    bundle_artifacts,
    hydrate_report_for_frontend,
    load_dataframe,
    load_artifacts,
    restore_store,
    save_artifacts,
    strip_citation_markers,
    validate_report,
)
from sinclair.survey.context import select_columns
import sinclair.survey.runtime as survey_runtime
from sinclair.survey.validators import validate_evidence


REAL_A = Path("examples/data/20557/20557.csv")
REAL_B = Path("examples/data/21081/21081.csv")


@dataclass
class ReportFixture:
    df: pd.DataFrame
    identity: SurveyIdentityPolicy
    respondent_id_col: str
    platform_col: str
    recommendation_col: str
    open_text_col: str
    profile_col: str


@dataclass
class MultiselectFixture:
    df: pd.DataFrame
    identity: SurveyIdentityPolicy
    respondent_id_col: str
    multi_col: str


class FakeEmbeddingBackend(EmbeddingBackend):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0
                    if any(
                        token in lowered for token in ["youtube", "platform", "channel"]
                    )
                    else 0.0,
                    1.0
                    if any(
                        token in lowered
                        for token in [
                            "analysis",
                            "technical",
                            "commentary",
                            "commentators",
                        ]
                    )
                    else 0.0,
                    1.0
                    if any(token in lowered for token in ["natura", "brand", "stocked"])
                    else 0.0,
                ]
            )
        return vectors


@pytest.fixture
def report_fixture() -> ReportFixture:
    df = load_dataframe(REAL_A)
    respondent_id_col = df.columns[0]
    platform_col = next(
        col
        for col in df.columns
        if df[col].astype(str).str.contains("YouTube", case=False, na=False).any()
    )
    recommendation_col = next(
        col
        for col in df.columns
        if df[col]
        .astype(str)
        .str.contains("Sim, com certeza", case=False, regex=False, na=False)
        .any()
    )
    profile_col = next(
        col
        for col in df.columns
        if df[col]
        .astype(str)
        .str.contains("Masculino|Feminino", case=False, regex=True, na=False)
        .any()
    )
    open_text_col = max(
        [col for col in df.columns if df[col].astype(str).str.len().mean() > 10],
        key=lambda col: df[col].astype(str).str.len().mean(),
    )
    identity = SurveyIdentityPolicy(respondent_id_column=respondent_id_col)
    return ReportFixture(
        df=df,
        identity=identity,
        respondent_id_col=respondent_id_col,
        platform_col=platform_col,
        recommendation_col=recommendation_col,
        open_text_col=open_text_col,
        profile_col=profile_col,
    )


@pytest.fixture
def multiselect_fixture() -> MultiselectFixture:
    df = load_dataframe(REAL_B)
    respondent_id_col = df.columns[0]
    multi_col = next(
        col
        for col in df.columns
        if df[col]
        .astype(str)
        .str.contains("Natura", case=False, regex=False, na=False)
        .any()
        and df[col]
        .astype(str)
        .str.contains("Avon", case=False, regex=False, na=False)
        .any()
    )
    identity = SurveyIdentityPolicy(respondent_id_column=respondent_id_col)
    return MultiselectFixture(
        df=df,
        identity=identity,
        respondent_id_col=respondent_id_col,
        multi_col=multi_col,
    )


def _defaults(identity: SurveyIdentityPolicy) -> SurveyDefaults:
    return SurveyDefaults(identity=identity)


def _valid_report(fx: ReportFixture) -> Report:
    platform_evidence = Evidence(
        base_rule=f"df[{fx.platform_col!r}].notna()",
        rule=f"df[{fx.platform_col!r}] == 'YouTube'",
        reason=f"Ao falar do canal preferido em {fx.platform_col}, menciona YouTube; exclui respostas vazias e inválidas.",
        source_column=fx.platform_col,
        match_label="YouTube",
        question_id=fx.platform_col,
    )
    intent_evidence = Evidence(
        base_rule=f"df[{fx.recommendation_col!r}].notna()",
        rule=f"df[{fx.recommendation_col!r}] == 'Sim, com certeza'",
        reason=f"Ao falar da intenção máxima de recomendação em {fx.recommendation_col}, menciona 'Sim, com certeza'; exclui respostas vazias e inválidas.",
        source_column=fx.recommendation_col,
        match_label="Sim, com certeza",
        question_id=fx.recommendation_col,
    )
    return Report(
        markdown=(
            "# Report\n\n"
            "The audience is concentrated around one primary channel, and recommendation intent is positive but not universal. "
            "The mix is concentrated enough to guide prioritization, while the recommendation signal still leaves room to deepen loyalty before expansion.\n\n"
            "## Platform gravity\n\n"
            "YouTube alone holds 44.4%[ct:ct_platform_share] of the audience base, which makes it the primary distribution anchor for reach and consistency.\n\n"
            "[[chart:1]]\n\n"
            "## Recommendation ceiling\n\n"
            "Only 38.9%[ct:ct_recommendation_certainty] say they would recommend with maximum certainty, which means enthusiasm exists but still needs reinforcement through programming depth and repeat value.\n\n"
            "[[chart:2]]"
        ),
        findings=[
            Finding(
                claim="The audience is concentrated around YouTube rather than fragmented across channels.",
                implication="Programming and promotion should optimize for YouTube first, then treat other channels as supporting distribution.",
                evidences=[platform_evidence],
                confidence=0.92,
            ),
            Finding(
                claim="Recommendation intent is solid but not yet overwhelming.",
                implication="The next gains likely come from strengthening habitual value, not only awareness.",
                evidences=[intent_evidence],
                confidence=0.86,
            ),
        ],
        citations=[
            Citation(
                citation_id="ct_platform_share",
                target=CitationTarget(
                    kind="chart_datum", chart_index=0, label="YouTube"
                ),
            ),
            Citation(
                citation_id="ct_recommendation_certainty",
                target=CitationTarget(
                    kind="chart_datum", chart_index=1, label="Sim, com certeza"
                ),
            ),
        ],
        charts=[
            Chart(
                type="bar",
                title="Primary platform preference",
                nr_questao=fx.platform_col,
                unit="%",
                data=[
                    ChartDatum(
                        label="YouTube",
                        value_pct=44.4,
                        evidence=platform_evidence,
                    )
                ],
            ),
            Chart(
                type="bar",
                title="Recommendation certainty",
                nr_questao=fx.recommendation_col,
                unit="%",
                data=[
                    ChartDatum(
                        label="Sim, com certeza",
                        value_pct=38.9,
                        evidence=intent_evidence,
                    )
                ],
            ),
        ],
    )


def test_load_dataframe_reads_real_dataset(report_fixture: ReportFixture):
    assert len(report_fixture.df) == 36
    assert report_fixture.platform_col in report_fixture.df.columns


def test_select_columns_star_keeps_all_columns(report_fixture: ReportFixture):
    selected = select_columns(
        report_fixture.df,
        report_fixture.open_text_col,
        "*",
        report_fixture.identity,
    )
    assert selected.columns.tolist() == report_fixture.df.columns.tolist()
    assert selected is not report_fixture.df


def test_select_columns_none_keeps_question_plus_identity(
    report_fixture: ReportFixture,
):
    selected = select_columns(
        report_fixture.df,
        report_fixture.open_text_col,
        None,
        report_fixture.identity,
    )
    assert selected.columns.tolist() == [
        report_fixture.respondent_id_col,
        report_fixture.open_text_col,
    ]


def test_validate_report_accepts_evidence_backed_report(
    report_fixture: ReportFixture,
):
    validate_report(_valid_report(report_fixture), report_fixture.df)


def test_report_study_uses_minimum_five_chart_policy(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame({"id": [1], "q1": ["x"]})
    captured: dict[str, int] = {}

    def fake_run_report(*args, **kwargs):
        defaults = kwargs.get("defaults")
        captured["min_charts"] = defaults.validation_policy.min_charts
        return Report(
            markdown="## Study\n\nTexto suficiente para passar.",
            findings=[
                Finding(
                    claim="A",
                    implication="B",
                    evidences=[
                        Evidence(
                            base_rule="df['q1'].notna()",
                            rule="df['q1'].notna()",
                            reason="Teste.",
                            source_column="q1",
                            question_id="q1",
                        )
                    ],
                    confidence=0.9,
                )
            ],
            citations=[],
            charts=[
                Chart(
                    type="bar",
                    title=f"Chart {i}",
                    nr_questao="q1",
                    data=[
                        ChartDatum(
                            label="A",
                            value_pct=100.0,
                            evidence=Evidence(
                                base_rule="df['q1'].notna()",
                                rule="df['q1'].notna()",
                                reason="Teste.",
                                source_column="q1",
                                question_id="q1",
                            ),
                        )
                    ],
                )
                for i in range(5)
            ],
        )

    monkeypatch.setattr(survey_runtime, "run_report", fake_run_report)
    monkeypatch.setattr(
        survey_runtime.SurveyArtifactStore,
        "ingest_report",
        lambda *args, **kwargs: None,
    )

    result = survey_runtime.report_study(df, defaults=SurveyDefaults())

    assert captured["min_charts"] == 5
    assert len(result.charts) == 5


def test_validate_report_accepts_other_columns_in_base_rule_for_question_scope(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts = [report.charts[0]]
    report.citations = [report.citations[0]]
    report.markdown = (
        "# Report\n\n"
        "This report is long enough to validate question-scoped base rules while keeping provenance on one question, even when the denominator uses a different column for segmentation before the active question is evaluated in the observed rule.\n\n"
        "## Platform gravity\n\n"
        "The dominant platform remains concentrated after segmentation, and the chart anchor below is enough for schema validation without publishing a visible percentage in markdown.\n\n"
        "[[chart:1]]"
    )
    report.findings = [report.findings[0]]
    report.findings[0].evidences[0] = Evidence(
        base_rule=(
            f"(df[{report_fixture.profile_col!r}].astype(str).str.len() >= 0)"
            f" & df[{report_fixture.platform_col!r}].notna()"
        ),
        rule=f"df[{report_fixture.platform_col!r}] == 'YouTube'",
        reason=f"Ao falar do canal preferido em {report_fixture.platform_col}, menciona YouTube; base filtrada.",
        source_column=report_fixture.platform_col,
        question_id=report_fixture.platform_col,
    )
    report.citations = []
    report.charts[0].data[0].value_pct = None
    report.charts[0].data[0].evidence = report.findings[0].evidences[0]

    validate_report(
        report,
        report_fixture.df,
        analysis_question_id=report_fixture.platform_col,
    )


def test_validate_report_uses_policy_minima(report_fixture: ReportFixture):
    report = _valid_report(report_fixture)
    report.charts = []
    with pytest.raises(ValueError, match="at least 1 chart"):
        validate_report(
            report,
            report_fixture.df,
            policy=SurveyValidationPolicy(min_charts=1),
        )


def test_validate_report_rejects_rule_outside_base(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts[0].data[0].evidence = Evidence(
        base_rule=f"df[{report_fixture.profile_col!r}] == 'Masculino'",
        rule=f"df[{report_fixture.platform_col!r}] == 'YouTube'",
        reason=f"Ao falar do canal preferido em {report_fixture.platform_col}, menciona YouTube; base filtrada.",
        question_id=report_fixture.platform_col,
    )
    with pytest.raises(ValueError, match="subset"):
        validate_report(report, report_fixture.df)


def test_validate_report_rejects_cross_question_rule_in_question_scope(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts = [report.charts[0]]
    report.citations = [report.citations[0]]
    report.markdown = (
        "# Report\n\n"
        "This report is long enough to validate scoped rule provenance and avoid failing earlier markdown-length checks before the scope validator inspects which question actually drives the observed rule in the evidence payload.\n\n"
        "## Platform gravity\n\n"
        "The narrative stays long enough for validation while avoiding a visible percentage that would need to stay numerically aligned with an intentionally invalid scoped rule.\n\n"
        "[[chart:1]]"
    )
    report.findings = [report.findings[0]]
    report.findings[0].evidences[0] = Evidence(
        base_rule=f"df[{report_fixture.platform_col!r}].notna()",
        rule=f"df[{report_fixture.recommendation_col!r}] == 'Sim, com certeza'",
        reason=f"Ao falar do canal preferido em {report_fixture.platform_col}, menciona respostas marcadas como 'Sim, com certeza'; exclui respostas vazias e inválidas.",
        source_column=report_fixture.platform_col,
        question_id=report_fixture.platform_col,
    )
    report.citations = []
    report.charts[0].data[0].value_pct = None
    report.charts[0].data[0].evidence = report.findings[0].evidences[0]

    with pytest.raises(ValueError, match="rule provenance"):
        validate_report(
            report,
            report_fixture.df,
            analysis_question_id=report_fixture.platform_col,
        )


def test_validate_report_rejects_cross_question_source_column_in_question_scope(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts = [report.charts[0]]
    report.citations = [report.citations[0]]
    report.markdown = (
        "# Report\n\n"
        "This report is long enough to validate scoped source provenance and avoid failing earlier markdown-length checks before the scope validator inspects which source column is being used as provenance for the evidence payload.\n\n"
        "## Platform gravity\n\n"
        "YouTube alone holds 44.4%[ct:ct_platform_share] of the audience base.\n\n"
        "[[chart:1]]"
    )
    report.findings = [report.findings[0]]
    report.findings[0].evidences[0].source_column = report_fixture.recommendation_col
    report.charts[0].data[0].evidence = report.findings[0].evidences[0]

    with pytest.raises(ValueError, match="evidence provenance"):
        validate_report(
            report,
            report_fixture.df,
            analysis_question_id=report_fixture.platform_col,
        )


def test_validate_report_rejects_cross_question_chart_in_question_scope(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts = [report.charts[0]]
    report.citations = [report.citations[0]]
    report.findings = [report.findings[0]]
    report.markdown = (
        "# Report\n\n"
        "This report is long enough to validate scoped chart provenance and avoid unrelated earlier failures before the validator checks whether the chart itself claims provenance from a question outside the active analytical scope.\n\n"
        "## Platform gravity\n\n"
        "YouTube alone holds 44.4%[ct:ct_platform_share] of the audience base.\n\n"
        "[[chart:1]]"
    )
    report.charts[0].nr_questao = report_fixture.recommendation_col

    with pytest.raises(ValueError, match="chart provenance"):
        validate_report(
            report,
            report_fixture.df,
            analysis_question_id=report_fixture.platform_col,
        )


def test_validate_report_rejects_unbacked_visible_percentage(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.markdown += "\n\n## Extra\n\nThis adds an unsupported 99.0% claim."
    with pytest.raises(ValueError, match="99.0%"):
        validate_report(report, report_fixture.df)


def test_validate_report_requires_each_duplicate_percentage_occurrence_to_be_cited(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts = [report.charts[0]]
    report.markdown = (
        "# Report\n\n"
        "This report is long enough to validate duplicate percentage disambiguation on real data. "
        "First signal is 44.4%[ct:ct_first_dup] and second signal is 44.4%[ct:ct_second_dup] on the same line, and each one must be cited explicitly.\n\n"
        "[[chart:1]]\n\n"
        "## What changes\n\n"
        "The same real percentage can appear twice and still require occurrence-aware disambiguation."
    )
    report.citations = [
        Citation(
            citation_id="ct_first_dup",
            target=CitationTarget(kind="chart_datum", chart_index=0, label="YouTube"),
        ),
        Citation(
            citation_id="ct_second_dup",
            target=CitationTarget(kind="chart_datum", chart_index=0, label="YouTube"),
        ),
    ]
    validate_report(report, report_fixture.df)
    report.citations = report.citations[:1]
    with pytest.raises(ValueError, match="44.4%|unknown citation markers"):
        validate_report(report, report_fixture.df)


def test_validate_report_rejects_missing_chart_datum_target(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.citations[0].target.label = "Missing label"
    with pytest.raises(ValueError, match="resolvable datum target"):
        validate_report(report, report_fixture.df)


def test_validate_report_rejects_missing_citation_marker(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.markdown = report.markdown.replace("[ct:ct_platform_share]", "")

    with pytest.raises(ValueError, match="does not disambiguate a visible 44.4%"):
        validate_report(report, report_fixture.df)


def test_validate_report_normalizes_cited_percentage_markdown_variants(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.markdown = report.markdown.replace(
        "44.4%[ct:ct_platform_share]",
        "**44.4%**[ct:ct_platform_share]",
    ).replace(
        "38.9%[ct:ct_recommendation_certainty]",
        "**38.9%[ct:ct_recommendation_certainty]**",
    )

    validate_report(report, report_fixture.df)

    assert "**44.4%**[ct:ct_platform_share]" in report.markdown
    assert "**38.9%**[ct:ct_recommendation_certainty]" in report.markdown


def test_validate_report_normalizes_plain_percentage_to_canonical_bold_citation(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)

    validate_report(report, report_fixture.df)

    assert "**44.4%**[ct:ct_platform_share]" in report.markdown
    assert "**38.9%**[ct:ct_recommendation_certainty]" in report.markdown


def test_validate_report_hydrates_canonical_datum_and_evidence_ids(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)

    validate_report(report, report_fixture.df)

    first_datum = report.charts[0].data[0]
    first_citation = report.citations[0]
    assert first_datum.datum_id is not None
    assert first_datum.evidence_id is not None
    assert first_citation.target.datum_id == first_datum.datum_id
    assert first_citation.target.evidence_id == first_datum.evidence_id


def test_validate_evidence_accepts_product_facing_reason(
    report_fixture: ReportFixture,
):
    evidence = Evidence(
        base_rule=f"df[{report_fixture.platform_col!r}] == 'YouTube'",
        rule=f"df[{report_fixture.platform_col!r}] == 'YouTube'",
        reason="identifica preferência principal",
        match_label="YouTube",
        source_column=report_fixture.platform_col,
        question_id=report_fixture.platform_col,
    )

    with pytest.raises(
        ValueError,
        match="evidence.reason must mention the source question or column",
    ):
        validate_evidence(evidence, report_fixture.df)


def test_validate_evidence_accepts_user_facing_reason(
    report_fixture: ReportFixture,
):
    evidence = Evidence(
        base_rule=f"df[{report_fixture.platform_col!r}].notna() & df[{report_fixture.platform_col!r}].astype(str).str.strip().ne('')",
        rule=f"df[{report_fixture.platform_col!r}] == 'YouTube'",
        reason=f"Ao falar do canal preferido em {report_fixture.platform_col}, menciona YouTube.",
        source_column=report_fixture.platform_col,
        question_id=report_fixture.platform_col,
    )

    validate_evidence(evidence, report_fixture.df)

    assert evidence.reason.startswith("Ao falar do canal preferido")


def test_validate_report_replaces_model_supplied_datum_ids_with_canonical_ones(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts[0].data[0].datum_id = "d1"
    report.citations[0].target.datum_id = "d1"

    validate_report(report, report_fixture.df)

    assert report.charts[0].data[0].datum_id.startswith("dt_")
    assert report.citations[0].target.datum_id == report.charts[0].data[0].datum_id


def test_validate_report_resolves_chart_datum_by_unique_label_when_index_is_wrong(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.citations[0].target.chart_index = 99
    report.citations[0].target.evidence_id = None
    report.citations[0].target.datum_id = None

    validate_report(report, report_fixture.df)

    assert report.citations[0].target.datum_id == report.charts[0].data[0].datum_id


def test_validate_report_discards_raw_chart_citation_evidence():
    df = pd.DataFrame({"q1": ["a", "a", "b"]})
    datum_evidence = Evidence(
        base_rule="df['q1'].notna()",
        rule="df['q1'] == 'a'",
        reason="Ao responder q1, menciona a.",
        match_label="a",
        source_column="q1",
        question_id="q1",
    )
    report = Report(
        markdown="# Report\n\n## Section\n\nA lidera com 66.7%[ct:ct_a].\n\n[[chart:chart-a]]",
        findings=[
            Finding(
                claim="A lidera.",
                implication="Priorizar A.",
                evidences=[datum_evidence],
                confidence=0.9,
            )
        ],
        citations=[
            Citation(
                citation_id="ct_a",
                target=CitationTarget(
                    kind="chart_datum",
                    chart_index=0,
                    label="A",
                    evidence=Evidence(
                        base_rule="df['q1'] == 'b'",
                        rule="df['q1'] == 'a'",
                        reason="Ao responder q1, menciona a.",
                        match_label="a",
                        source_column="q1",
                        question_id="q1",
                    ),
                ),
            )
        ],
        charts=[
            Chart(
                type="bar",
                title="Chart A",
                slug="chart-a",
                nr_questao="q1",
                unit="%",
                data=[
                    ChartDatum(
                        label="A",
                        value_pct=66.7,
                        evidence=datum_evidence,
                    )
                ],
            )
        ],
    )

    validate_report(
        report,
        df,
        policy=SurveyValidationPolicy(
            min_markdown_chars=1,
            min_findings=1,
            min_charts=1,
            min_citations=1,
        ),
    )

    assert report.citations[0].target.evidence is None
    assert (
        report.citations[0].target.evidence_id == report.charts[0].data[0].evidence_id
    )


def test_validate_report_keeps_same_percentage_distinct_across_two_charts(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.markdown = (
        "# Report\n\n"
        "The first anchor is 44.4%[ct:ct_platform_share] and the second anchor is 44.4%[ct:ct_platform_share_clone].\n\n"
        "[[chart:1]]\n\n"
        "[[chart:2]]\n\n"
        "[[chart:3]]\n\n"
        "## What changes\n\n"
        "Two different charts can expose the same percentage without collapsing provenance."
    )
    report.citations = [report.citations[0]]
    report.citations.append(
        Citation(
            citation_id="ct_platform_share_clone",
            target=CitationTarget(kind="chart_datum", chart_index=2, label="YouTube"),
        )
    )
    report.charts.append(
        Chart(
            type="bar",
            title="Primary platform preference clone",
            nr_questao=report_fixture.platform_col,
            unit="%",
            data=[
                ChartDatum(
                    label="YouTube",
                    value_pct=44.4,
                    evidence=report.charts[0].data[0].evidence,
                )
            ],
        )
    )

    validate_report(report, report_fixture.df)

    assert report.citations[0].target.datum_id != report.citations[1].target.datum_id


def test_validate_report_allows_multi_scope_chart_when_chart_scope_is_not_declared(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)
    report.charts = [
        Chart(
            type="bar",
            title="Mixed read",
            slug="mixed-read",
            nr_questao=None,
            unit="%",
            data=[
                report.charts[0].data[0],
                ChartDatum(
                    label="Sim, com certeza",
                    value_pct=38.9,
                    evidence=report.citations[1].target.evidence
                    or report.charts[1].data[0].evidence,
                ),
            ],
        )
    ]
    report.markdown = (
        "# Report\n\n"
        "This report is long enough to validate a mixed-scope chart in study mode. "
        "YouTube appears in 44.4%[ct:ct_platform_share] while maximum recommendation appears in 38.9%[ct:ct_recommendation_certainty].\n\n"
        "[[chart:mixed-read]]\n\n"
        "## What changes\n\n"
        "A single chart can combine different question scopes when each datum keeps its own evidence provenance."
    )

    validate_report(report, report_fixture.df)

    assert report.charts[0].nr_questao is None


def test_strip_citation_markers_removes_inline_tokens(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)

    stripped = strip_citation_markers(report.markdown)

    assert "[ct:" not in stripped
    assert "% )" not in stripped
    assert "44.4% of the audience base" in stripped


def test_hydrate_report_for_frontend_renders_chart_block_and_refs(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)

    validate_report(report, report_fixture.df)
    store.ingest_report(
        report,
        report_fixture.df,
        scope=f"question:{report_fixture.platform_col}",
    )
    bundle = hydrate_report_for_frontend(report, store)

    assert "```chart" in bundle.markdown
    assert "](ref:rf_lb_" in bundle.markdown
    assert any(ref.kind == "line_binding" for ref in bundle.references.values())
    assert any(ref.kind == "response_set" for ref in bundle.references.values())
    assert any(ref.kind == "datum_binding" for ref in bundle.references.values())
    chart_block = bundle.chart_blocks[0]
    assert chart_block.payload["chart_id"] == report.charts[0].slug
    assert chart_block.payload["data"][0]["refId"].startswith("rf_dt_")
    assert chart_block.payload["data"][0]["responseSetRefId"].startswith("rf_rs_")
    assert (
        chart_block.payload["data"][0]["rationale"]
        == report.charts[0].data[0].evidence.reason
    )


def test_hydrate_report_for_frontend_is_deterministic(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)

    validate_report(report, report_fixture.df)
    store.ingest_report(
        report,
        report_fixture.df,
        scope=f"question:{report_fixture.platform_col}",
    )
    first = hydrate_report_for_frontend(report, store)
    second = hydrate_report_for_frontend(report, store)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_hydrate_report_for_frontend_keeps_nr_questao_per_datum(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)
    report.charts = [
        Chart(
            type="bar",
            title="Mixed read",
            slug="mixed-read",
            nr_questao=None,
            unit="%",
            data=[report.charts[0].data[0], report.charts[1].data[0]],
        )
    ]
    report.markdown = (
        "# Report\n\n"
        "This report is long enough to validate frontend datum scopes. "
        "YouTube appears in 44.4%[ct:ct_platform_share] while recommendation certainty appears in 38.9%[ct:ct_recommendation_certainty].\n\n"
        "[[chart:mixed-read]]\n\n"
        "## What changes\n\n"
        "The frontend must see the source question for each datum in a mixed chart."
    )

    validate_report(report, report_fixture.df)
    store.ingest_report(report, report_fixture.df, scope="study:executive")
    bundle = hydrate_report_for_frontend(report, store)

    data = bundle.chart_blocks[0].payload["data"]
    assert data[0]["nr_questao"] == report_fixture.platform_col
    assert data[1]["nr_questao"] == report_fixture.recommendation_col


def test_frontend_controller_resolves_provenance_and_lazy_evidence_page(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)

    validate_report(report, report_fixture.df)
    store.ingest_report(
        report,
        report_fixture.df,
        scope=f"question:{report_fixture.platform_col}",
    )
    controller = SurveyFrontendController(
        store=store,
        reports={f"question:{report_fixture.platform_col}": report},
    )

    bundle = controller.render_report(f"question:{report_fixture.platform_col}")
    line_ref_id = next(
        ref_id
        for ref_id, ref in bundle.references.items()
        if ref.kind == "line_binding"
    )

    provenance = controller.get_provenance(
        f"question:{report_fixture.platform_col}", line_ref_id
    )
    page = controller.get_evidence_page(
        f"question:{report_fixture.platform_col}",
        line_ref_id,
        offset=0,
        limit=5,
    )
    datum_ref_id = next(
        ref_id
        for ref_id, ref in bundle.references.items()
        if ref.kind == "datum_binding"
    )
    datum_provenance = controller.get_provenance(
        f"question:{report_fixture.platform_col}", datum_ref_id
    )
    datum_page = controller.get_evidence_page(
        f"question:{report_fixture.platform_col}",
        datum_ref_id,
        offset=0,
        limit=5,
    )

    assert isinstance(provenance, FrontendProvenance)
    assert provenance.reference.ref_id == line_ref_id
    assert provenance.evidence is not None
    assert provenance.evidence["nr_questao"] == report_fixture.platform_col
    assert "/" in provenance.evidence["formula"]
    assert isinstance(page, FrontendEvidencePage)
    assert page.ref_id == line_ref_id
    assert page.total_refs >= len(page.refs)
    assert len(page.refs) <= 5
    assert page.formula == f"{page.match_count}/{page.base_count}"
    assert all(item.position >= 1 for item in page.items)
    assert isinstance(datum_provenance, FrontendProvenance)
    assert datum_provenance.reference.kind == "datum_binding"
    assert isinstance(datum_page, FrontendEvidencePage)
    assert datum_page.total_refs == page.total_refs


def test_survey_study_exposes_frontend_controller(
    report_fixture: ReportFixture,
):
    defaults = _defaults(report_fixture.identity)
    study = SurveyStudy(
        report_fixture.df,
        study_id="study-20557",
        defaults=defaults,
    )
    report = _valid_report(report_fixture)
    validate_report(report, report_fixture.df)
    study._question_reports[report_fixture.platform_col] = report
    study.store.ingest_report(
        report,
        report_fixture.df,
        scope=f"question:{report_fixture.platform_col}",
    )

    controller = study.frontend_controller()
    bundle = controller.render_report(f"question:{report_fixture.platform_col}")

    assert isinstance(controller, SurveyFrontendController)
    assert isinstance(bundle, FrontendRenderBundle)


def test_hydrate_report_for_frontend_wraps_anchor_text_links(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)
    report.charts = [report.charts[0]]
    report.markdown = (
        "# Report\n\n"
        "Quando o assunto é plataforma, o canal dominante é o YouTube e isso organiza a leitura do comportamento observado na base. "
        "Esse trecho precisa ser longo o bastante para continuar parecendo um report real e para passar pela validação mínima de markdown.\n\n"
        "[[chart:1]]\n\n"
        "## Resumo\n\n"
        "O YouTube segue como referência principal, com uma narrativa simples e longa o bastante para ser validada como saída editorial."
    )
    report.citations = [
        Citation(
            citation_id="ct_platform_anchor",
            anchor_text="o canal dominante é o YouTube",
            target=CitationTarget(
                kind="response_set",
                nr_questao=report_fixture.platform_col,
                evidence=report.charts[0].data[0].evidence,
            ),
        )
    ]

    validate_report(report, report_fixture.df)
    store.ingest_report(
        report,
        report_fixture.df,
        scope=f"question:{report_fixture.platform_col}",
    )
    bundle = hydrate_report_for_frontend(report, store)

    assert "[o canal dominante é o YouTube](ref:rf_lb_" in bundle.markdown


def test_validate_report_normalizes_numeric_chart_anchors_to_slugs(
    report_fixture: ReportFixture,
):
    report = _valid_report(report_fixture)

    validate_report(report, report_fixture.df)

    assert "[[chart:1]]" not in report.markdown
    assert report.charts[0].slug is not None
    assert f"[[chart:{report.charts[0].slug}]]" in report.markdown


def test_store_renders_only_matching_value_for_multiselect_dataset(
    multiselect_fixture: MultiselectFixture,
):
    store = SurveyArtifactStore(identity=multiselect_fixture.identity)
    evidence = Evidence(
        base_rule=f"df[{multiselect_fixture.multi_col!r}].notna()",
        rule=f"df[{multiselect_fixture.multi_col!r}].astype(str).str.contains('Natura', case=False, regex=False, na=False)",
        reason=f"Ao falar das marcas em estoque em {multiselect_fixture.multi_col}, menciona Natura; exclui respostas vazias e inválidas.",
        source_column=multiselect_fixture.multi_col,
        match_label="Natura",
        question_id=multiselect_fixture.multi_col,
    )
    evidence_id = store.save_evidence(
        evidence, multiselect_fixture.df, scope=multiselect_fixture.multi_col
    )
    record = store.get_evidence(evidence_id)
    assert record is not None
    assert record.preview
    assert all(item == "Natura" for item in record.preview)
    assert "Avon" not in record.preview
    assert record.response_refs
    assert all(
        ref.question_id == multiselect_fixture.multi_col for ref in record.response_refs
    )


def test_store_search_findings_returns_ingested_report_data(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)
    results = store.search_findings("YouTube")
    assert results
    assert results[0].scope == report_fixture.open_text_col


def test_store_lexical_search_is_fuzzy(report_fixture: ReportFixture):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)

    results = store.search_findings("youtub prioritise channel")

    assert results
    assert "YouTube" in results[0].claim


def test_store_semantic_search_finds_relevant_finding(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(
        identity=report_fixture.identity,
        embedding_backend=FakeEmbeddingBackend(),
    )
    report = _valid_report(report_fixture)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)

    results = store.semantic_search_findings(
        "Which channel should the team prioritize first?"
    )

    assert results
    assert "YouTube" in results[0].claim


def test_store_semantic_search_finds_relevant_evidence(
    multiselect_fixture: MultiselectFixture,
):
    store = SurveyArtifactStore(
        identity=multiselect_fixture.identity,
        embedding_backend=FakeEmbeddingBackend(),
    )
    evidence = Evidence(
        base_rule=f"df[{multiselect_fixture.multi_col!r}].notna()",
        rule=f"df[{multiselect_fixture.multi_col!r}].astype(str).str.contains('Natura', case=False, regex=False, na=False)",
        reason=f"Ao falar das marcas em estoque em {multiselect_fixture.multi_col}, menciona Natura; exclui respostas vazias e inválidas.",
        source_column=multiselect_fixture.multi_col,
        match_label="Natura",
        question_id=multiselect_fixture.multi_col,
    )
    store.save_evidence(
        evidence, multiselect_fixture.df, scope=multiselect_fixture.multi_col
    )

    results = store.semantic_search_evidence("Which stocked brand is Natura?")

    assert results
    assert results[0].evidence.match_label == "Natura"


def test_store_semantic_search_falls_back_to_lexical(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)

    results = store.semantic_search_findings("YouTube")

    assert results
    assert results[0].scope == report_fixture.open_text_col


def test_store_hybrid_search_prefers_semantic_plus_lexical(
    report_fixture: ReportFixture,
):
    store = SurveyArtifactStore(
        identity=report_fixture.identity,
        embedding_backend=FakeEmbeddingBackend(),
    )
    report = _valid_report(report_fixture)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)

    results = store.hybrid_search_findings(
        "Which platform should distribution concentrate on?"
    )

    assert results
    assert "YouTube" in results[0].claim


def test_store_tool_output_is_compact_json(report_fixture: ReportFixture):
    store = SurveyArtifactStore(identity=report_fixture.identity)
    report = _valid_report(report_fixture)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)
    tools = {tool.name: tool for tool in store.as_tools()}
    output = tools["list_findings"].invoke({})
    assert "\n" not in output
    assert '"finding_id"' in output


def test_store_search_tool_uses_hybrid_output(report_fixture: ReportFixture):
    store = SurveyArtifactStore(
        identity=report_fixture.identity,
        embedding_backend=FakeEmbeddingBackend(),
    )
    report = _valid_report(report_fixture)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)
    tools = {tool.name: tool for tool in store.as_tools()}

    output = tools["search_findings"].invoke(
        {"query": "Which platform should we prioritize?"}
    )

    payload = json.loads(output)
    assert payload
    assert "YouTube" in payload[0]["claim"]
    assert '"preview"' not in output


def test_survey_app_exposes_context_store_and_identity_defaults(
    report_fixture: ReportFixture,
):
    defaults = _defaults(report_fixture.identity)
    app = SurveyApp(
        report_fixture.df,
        study_context="Understand audience behavior and channel preference.",
        question_map={report_fixture.platform_col: "Which platform is used most?"},
        column_metadata=f"{report_fixture.platform_col}=channel; {report_fixture.recommendation_col}=advocacy; {report_fixture.open_text_col}=qualitative feedback",
        defaults=defaults,
    )
    assert app.study_context is not None
    assert app.store is not None
    assert (
        app.defaults.identity.respondent_id_column == report_fixture.respondent_id_col
    )


def test_survey_study_exports_shared_store_artifacts(
    report_fixture: ReportFixture,
):
    defaults = _defaults(report_fixture.identity)
    study = SurveyStudy(
        report_fixture.df,
        study_id="study-20557",
        defaults=defaults,
    )
    report = _valid_report(report_fixture)
    study._question_reports[report_fixture.open_text_col] = report
    study._study_report = report
    study._chat_reports["default"] = [report]
    study.store.ingest_report(
        report,
        report_fixture.df,
        scope=f"question:{report_fixture.open_text_col}",
    )

    bundle = study.export_artifacts()

    assert bundle.metadata["study_id"] == "study-20557"
    assert report_fixture.open_text_col in bundle.question_reports
    assert bundle.study_report is not None
    assert bundle.chat_reports
    assert bundle.store_snapshot


def test_toolkit_get_final_chart_numbers_freezes_exact_percentages(
    report_fixture: ReportFixture,
):
    toolkit = SurveyToolKit(df=report_fixture.df, identity=report_fixture.identity)
    tools = {tool.name: tool for tool in toolkit.as_tools()}

    output = tools["get_final_chart_numbers"].invoke(
        {
            "question_id": report_fixture.recommendation_col,
            "intent": "Estou congelando os percentuais do gráfico de recomendação.",
            "title": "Recommendation certainty",
            "chart_slug": "recommendation-certainty",
            "items": [
                {
                    "label": "Sim, com certeza",
                    "rule": f"df[{report_fixture.recommendation_col!r}] == 'Sim, com certeza'",
                    "reason": "Maximum recommendation certainty.",
                }
            ],
        }
    )

    payload = json.loads(output)
    assert payload["chart"]["slug"] == "recommendation-certainty"
    assert (
        payload["publishable_datums"][0]["canonical_citation_markdown"]
        == "**38.9%**[ct:ct_1]"
    )
    assert payload["publishable_datums"][0]["target"]["evidence_id"].startswith("ev_")
    assert payload["chart"]["data"][0]["evidence"]["match_label"] == "Sim, com certeza"
    assert (
        payload["chart"]["data"][0]["evidence"]["reason"]
        == f"Ao responder {report_fixture.recommendation_col}, menciona Sim, com certeza."
    )


def test_toolkit_get_final_chart_numbers_accepts_mentions_shorthand(
    report_fixture: ReportFixture,
):
    toolkit = SurveyToolKit(df=report_fixture.df, identity=report_fixture.identity)
    tools = {tool.name: tool for tool in toolkit.as_tools()}

    output = tools["get_final_chart_numbers"].invoke(
        {
            "question_id": report_fixture.platform_col,
            "intent": "Estou congelando um bucket de YouTube.",
            "title": "Platform preference",
            "chart_slug": "platform-preference",
            "items": [
                {
                    "label": "YouTube",
                    "rule": "mentions(YouTube)",
                    "reason": "Quando fala do canal preferido, cita YouTube.",
                }
            ],
        }
    )

    payload = json.loads(output)
    assert payload["publishable_datums"][0]["value_pct"] > 0
    assert payload["chart"]["data"][0]["evidence"]["match_label"] == "YouTube"
    assert (
        payload["chart"]["data"][0]["evidence"]["reason"]
        == f"Ao responder {report_fixture.platform_col}, menciona YouTube."
    )


def test_toolkit_get_final_chart_numbers_rejects_zero_match_datums(
    report_fixture: ReportFixture,
):
    toolkit = SurveyToolKit(df=report_fixture.df, identity=report_fixture.identity)
    tools = {tool.name: tool for tool in toolkit.as_tools()}

    with pytest.raises(ValueError, match="zero matches"):
        tools["get_final_chart_numbers"].invoke(
            {
                "question_id": report_fixture.platform_col,
                "intent": "Estou congelando um bucket vazio.",
                "title": "Platform preference",
                "chart_slug": "platform-preference-empty",
                "items": [
                    {
                        "label": "Nada",
                        "rule": f"df[{report_fixture.platform_col!r}] == 'inexistente'",
                        "reason": "Bucket propositalmente vazio.",
                    }
                ],
            }
        )


def test_benchmark_question_report_returns_cold_and_warm(
    report_fixture: ReportFixture,
):
    app = SurveyApp(report_fixture.df, defaults=_defaults(report_fixture.identity))

    calls: list[str] = []

    def fake_report_question(
        question_column: str,
        *,
        prompt: str | None = None,
        verbose: bool | None = None,
    ):
        calls.append(question_column)
        return _valid_report(report_fixture)

    app.report_question = fake_report_question  # type: ignore[method-assign]

    result = benchmark_question_report(app, report_fixture.open_text_col, prompt="test")

    assert isinstance(result, BenchmarkResult)
    assert result.cold.mode == "cold"
    assert result.warm.mode == "warm"
    assert calls == [
        report_fixture.open_text_col,
        report_fixture.open_text_col,
    ]


def test_baseline_question_report_runs_once(report_fixture: ReportFixture):
    app = SurveyApp(report_fixture.df, defaults=_defaults(report_fixture.identity))

    calls: list[str] = []

    def fake_report_question(
        question_column: str,
        *,
        prompt: str | None = None,
        verbose: bool | None = None,
    ):
        calls.append(question_column)
        return _valid_report(report_fixture)

    app.report_question = fake_report_question  # type: ignore[method-assign]

    result = baseline_question_report(app, report_fixture.open_text_col, prompt="test")

    assert isinstance(result, BaselineRun)
    assert result.question_column == report_fixture.open_text_col
    assert calls == [report_fixture.open_text_col]


def test_can_save_and_load_use_case_artifacts(
    tmp_path: Path, report_fixture: ReportFixture
):
    report = _valid_report(report_fixture)
    bundle = bundle_artifacts(
        question_reports={report_fixture.open_text_col: report},
        metadata={"dataset": "real-a"},
    )

    path = save_artifacts(bundle, tmp_path / "artifacts.json")
    loaded = load_artifacts(path)

    assert isinstance(loaded, UseCaseArtifacts)
    assert report_fixture.open_text_col in loaded.question_reports
    assert loaded.metadata["dataset"] == "real-a"


def test_can_restore_store_from_artifact_bundle(report_fixture: ReportFixture):
    report = _valid_report(report_fixture)
    store = SurveyArtifactStore(identity=report_fixture.identity)
    store.ingest_report(report, report_fixture.df, scope=report_fixture.open_text_col)
    bundle = bundle_artifacts(store=store)

    restored = SurveyArtifactStore(identity=report_fixture.identity)
    restore_store(bundle, restored)

    assert restored.has_artifacts()
    assert restored.list_findings()
