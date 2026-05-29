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
    label: str = Field(description="Human bucket label to publish in the chart.")
    rule: str = Field(
        description="Boolean pandas rule for the observed group. `mentions(...)` is accepted as shorthand."
    )
    base_rule: str = Field(
        default="",
        description="Optional denominator rule. Leave empty to use non-empty answers from the same question.",
    )
    reason: str = Field(
        description="Short human-readable criterion shown in the evidence modal. Describe what the respondent actually says, not the business takeaway, not the bucket name, and not the question code. Good: 'Quando fala do banco principal, cita Nubank.' or 'Quando descreve a compra por impulso, menciona promoção e desconto.' Bad: 'Marca mais recorrente na carteira atual.' or 'Entra no recorte observado.' or 'Ao responder Q37_SNTS, menciona promo.'"
    )


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
            if not stripped:
                return stripped

            def _replace_mentions(match: re.Match[str]) -> str:
                raw_token = match.group(1)
                token = _strip_literal(raw_token)
                return (
                    f"df[{question_id!r}].astype(str).str.contains("
                    f"{token!r}, case=False, regex=False, na=False)"
                )

            return re.sub(r"mentions\(([^()]+)\)", _replace_mentions, stripped)

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

        def _strip_literal(token: str) -> str:
            stripped = token.strip()
            if (
                len(stripped) >= 2
                and stripped[0] == stripped[-1]
                and stripped[0] in {'"', "'"}
            ):
                return stripped[1:-1].strip()
            return stripped

        def _infer_match_label(rule: str) -> str | None:
            stripped = rule.strip()
            mention_match = re.search(r"mentions\(([^()]+)\)", stripped)
            if mention_match:
                label = _strip_literal(mention_match.group(1))
                return label or None
            equality_match = re.search(
                r"(?:==|\.eq\()\s*(['\"])(?P<value>.+?)\1\)?",
                stripped,
            )
            if equality_match:
                label = equality_match.group("value").strip()
                return label or None
            return None

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
                match_label = _infer_match_label(item.rule)
                base_rule = _normalize_rule(item.base_rule, question_id)
                rule = _normalize_rule(item.rule, question_id)
                _reject_bare_negation_rule(rule)
                evidence = Evidence(
                    base_rule=base_rule
                    or f"df[{question_id!r}].notna() & df[{question_id!r}].astype(str).str.strip().ne('')",
                    rule=rule,
                    reason=item.reason,
                    source_column=question_id,
                    match_label=match_label,
                    question_id=question_id,
                )
                try:
                    validate_evidence(evidence, self.df)
                except ValueError as exc:
                    if str(exc) == "rule must be an explicit subset of base_rule":
                        raise ValueError(
                            "rule must be an explicit subset of base_rule; leave base_rule empty or make it broader than rule"
                        ) from exc
                    raise
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
                description="Freeze one publishable bar chart for a single question scope before markdown publication. `mentions(pattern)` is accepted as shorthand for text matching. The `reason` for each datum must be one short human sentence describing the observable criterion shown in the evidence modal.",
            ),
        ]
