from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SurveyValidationPolicy:
    min_markdown_chars: int = 180
    require_sections: bool = True
    min_findings: int = 1
    min_charts: int = 0
    min_citations: int = 0
    require_percent_citations: bool = True
    allowed_chart_types: tuple[str, ...] = ("bar", "grouped_bar")
    chart_unit: str = "%"


@dataclass(slots=True)
class SurveyIdentityPolicy:
    respondent_id_column: str = "respondent_id"
    fallback_respondent_id_columns: tuple[str, ...] = ("id",)


@dataclass(slots=True)
class SurveyDefaults:
    model: str = "gpt-5.4"
    temperature: float = 1.0
    reasoning_level: str = "high"
    max_iterations: int = 128
    finalization_window: int = 16
    tool_retries: int = 1
    verbose: bool = False
    identity: SurveyIdentityPolicy = field(default_factory=SurveyIdentityPolicy)
    validation_policy: SurveyValidationPolicy = field(
        default_factory=SurveyValidationPolicy
    )
