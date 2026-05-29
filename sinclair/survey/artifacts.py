from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import Report
from .store import SurveyArtifactStore


class UseCaseArtifacts(BaseModel):
    question_reports: dict[str, Report] = Field(default_factory=dict)
    study_report: Report | None = None
    chat_reports: list[Report] = Field(default_factory=list)
    store_snapshot: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def save_artifacts(bundle: UseCaseArtifacts, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def load_artifacts(path: str | Path) -> UseCaseArtifacts:
    payload = Path(path).read_text(encoding="utf-8")
    return UseCaseArtifacts.model_validate_json(payload)


def bundle_artifacts(
    *,
    question_reports: dict[str, Report] | None = None,
    study_report: Report | None = None,
    chat_reports: list[Report] | None = None,
    store: SurveyArtifactStore | None = None,
    metadata: dict[str, Any] | None = None,
) -> UseCaseArtifacts:
    return UseCaseArtifacts(
        question_reports=question_reports or {},
        study_report=study_report,
        chat_reports=chat_reports or [],
        store_snapshot=store.export_snapshot() if store is not None else {},
        metadata=metadata or {},
    )


def restore_store(bundle: UseCaseArtifacts, store: SurveyArtifactStore) -> None:
    if bundle.store_snapshot:
        store.import_snapshot(bundle.store_snapshot)


def save_store_snapshot(store: SurveyArtifactStore, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(store.export_snapshot(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_store_snapshot(path: str | Path, store: SurveyArtifactStore) -> None:
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    store.import_snapshot(snapshot)
