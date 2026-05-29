from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from sinclair._intent import validate_intent_text

from ._helpers import eval_mask
from .config import SurveyIdentityPolicy
from .models import Evidence
from .provenance import normalize_chart_slug, stable_evidence_id
from .validators import validate_evidence


class IntentInput(BaseModel):
    intent: str = Field(
        description="One short first-person product-facing status in progress."
    )

    @field_validator("intent")
    @classmethod
    def _validate_intent(cls, value: str) -> str:
        return validate_intent_text(value)


class PublishableDatumInput(BaseModel):
    label: str
    rule: str
    base_rule: str = ""
    reason: str


class PublishableChartArgs(IntentInput):
    question_id: str
    title: str
    items: list[PublishableDatumInput] = Field(min_length=1)
    chart_slug: str = ""
    chart_type: Literal["bar"] = "bar"


@dataclass(slots=True)
class SurveyToolKit:
    df: pd.DataFrame
    identity: SurveyIdentityPolicy

    def as_tools(self) -> list[StructuredTool]:
        def _json(payload) -> str:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        def _normalize_rule(expr: str, question_id: str) -> str:
            stripped = expr.strip()
            match = re.fullmatch(r"mentions\((.+)\)", stripped)
            if match:
                pattern = match.group(1).strip()
                return (
                    f"df[{question_id!r}].astype(str).str.contains("
                    f"{pattern!r}, case=False, regex=True, na=False)"
                )
            return stripped

        def _reject_bare_negation_rule(rule: str) -> None:
            compact = re.sub(r"\s+", "", rule.casefold())
            forbidden = {
                r"\b(?:n[aã]o)\b",
                r"\b(n[aã]o)\b",
                r"\bn[aã]o\b",
                r"\b(?:não)\b",
                r"\bnão\b",
                r"\bnao\b",
            }
            if any(token in compact for token in forbidden):
                raise ValueError(
                    "rules cannot match bare 'não' by itself; use an explicit phrase and exclusions when needed"
                )

        def get_final_chart_numbers(
            question_id: str,
            intent: str,
            title: str,
            items: list[PublishableDatumInput],
            chart_slug: str = "",
            chart_type: Literal["bar"] = "bar",
        ) -> str:
            _ = intent
            if question_id not in self.df.columns:
                return _json(
                    {
                        "error": "question_id not found",
                        "question_id": question_id,
                    }
                )
            chart_slug = normalize_chart_slug(chart_slug) or normalize_chart_slug(title)
            chart_data = []
            datums = []
            for index, item in enumerate(items, start=1):
                base_rule = _normalize_rule(item.base_rule, question_id)
                rule = _normalize_rule(item.rule, question_id)
                _reject_bare_negation_rule(rule)
                evidence = Evidence(
                    base_rule=base_rule
                    or f"df[{question_id!r}].notna() & df[{question_id!r}].astype(str).str.strip().ne('')",
                    rule=rule,
                    reason=item.reason,
                    source_column=question_id,
                    question_id=question_id,
                )
                validate_evidence(evidence, self.df)
                base_mask = eval_mask(evidence.base_rule, self.df)
                rule_mask = eval_mask(evidence.rule, self.df)
                match_mask = base_mask & rule_mask
                base_count = int(base_mask.sum())
                match_count = int(match_mask.sum())
                if base_count <= 0:
                    raise ValueError(
                        f"datum {item.label!r} has an empty base set; choose a broader denominator"
                    )
                if match_count <= 0:
                    raise ValueError(
                        f"datum {item.label!r} has zero matches; choose a broader rule"
                    )
                value_pct = (
                    round(match_count / base_count * 100, 1) if base_count else 0.0
                )
                evidence_id = stable_evidence_id(evidence)
                chart_data.append(
                    {
                        "label": item.label,
                        "value_pct": value_pct,
                        "evidence": evidence.model_dump(mode="json"),
                        "datum_id": None,
                        "evidence_id": evidence_id,
                        "series": [],
                    }
                )
                datums.append(
                    {
                        "citation_id": f"ct_{index}",
                        "label": item.label,
                        "value_pct": value_pct,
                        "canonical_citation_markdown": f"**{value_pct:.1f}%**[ct:ct_{index}]",
                        "target": {
                            "kind": "chart_datum",
                            "nr_questao": question_id,
                            "evidence_id": evidence_id,
                            "label": item.label,
                            "series": None,
                        },
                    }
                )
            return _json(
                {
                    "question_id": question_id,
                    "chart": {
                        "type": chart_type,
                        "title": title,
                        "slug": chart_slug,
                        "nr_questao": question_id,
                        "unit": "%",
                        "data": chart_data,
                    },
                    "publishable_datums": datums,
                    "instruction": "Use only these exact value_pct numbers if you publish percentages in markdown. Prefer the provided canonical_citation_markdown snippet for each datum.",
                }
            )

        return [
            StructuredTool.from_function(
                get_final_chart_numbers,
                name="get_final_chart_numbers",
                args_schema=PublishableChartArgs,
                description="Freeze one publishable bar chart for a single question scope before markdown publication. `mentions(pattern)` is accepted as shorthand for text matching.",
            ),
        ]
