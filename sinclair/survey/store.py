from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from threading import RLock
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import StructuredTool

from ._helpers import (
    extract_response_previews,
    extract_refs,
    extract_response_refs,
    iter_report_evidences,
)
from .config import SurveyIdentityPolicy
from .embeddings import EmbeddingBackend
from .models import Evidence, EvidenceRecord, FindingRecord, Report
from .provenance import stable_evidence_id
from .validators import canonicalize_evidence_reason, validate_evidence


class CacheBackend(ABC):
    @abstractmethod
    def get(self, namespace: str, key: str) -> dict[str, Any] | None: ...
    @abstractmethod
    def set(self, namespace: str, key: str, value: dict[str, Any]) -> None: ...
    @abstractmethod
    def items(self, namespace: str) -> list[tuple[str, dict[str, Any]]]: ...
    @abstractmethod
    def delete(self, namespace: str, key: str) -> None: ...
    @abstractmethod
    def clear(self, namespace: str | None = None) -> None: ...


class MemoryCacheBackend(CacheBackend):
    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = RLock()

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._data.get(namespace, {}).get(key)

    def set(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault(namespace, {})[key] = value

    def items(self, namespace: str) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            return list(self._data.get(namespace, {}).items())

    def delete(self, namespace: str, key: str) -> None:
        with self._lock:
            self._data.get(namespace, {}).pop(key, None)

    def clear(self, namespace: str | None = None) -> None:
        with self._lock:
            if namespace is None:
                self._data.clear()
            else:
                self._data.pop(namespace, None)


class SurveyArtifactStore:
    def __init__(
        self,
        cache: CacheBackend | None = None,
        *,
        identity: SurveyIdentityPolicy | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self._cache = cache or MemoryCacheBackend()
        self.identity = identity or SurveyIdentityPolicy()
        self.embedding_backend = embedding_backend

    def ingest_report(
        self, report: Report, df: pd.DataFrame, *, scope: str | None = None
    ) -> None:
        for evidence in iter_report_evidences(report):
            self.save_evidence(evidence, df, scope=scope)
        for idx, finding in enumerate(report.findings):
            finding_id = self._finding_id(scope, idx, finding.claim)
            linked_ids = [self._evidence_id(evidence) for evidence in finding.evidences]
            record = FindingRecord(
                finding_id=finding_id,
                scope=scope,
                claim=finding.claim,
                implication=finding.implication,
                confidence=finding.confidence,
                evidence_ids=linked_ids,
            )
            self._cache.set("findings", finding_id, record.model_dump(mode="json"))
            self._index_finding(record)

    def save_evidence(
        self, evidence: Evidence, df: pd.DataFrame, *, scope: str | None = None
    ) -> str:
        evidence = canonicalize_evidence_reason(_with_question_id(evidence, scope))
        evidence_id = self._evidence_id(evidence)
        record = self._resolve_evidence_record(evidence_id, evidence, df, scope=scope)
        self._cache.set("evidence", evidence_id, record.model_dump(mode="json"))
        self._index_evidence(record)
        return evidence_id

    def get_finding(self, finding_id: str) -> FindingRecord | None:
        payload = self._cache.get("findings", finding_id)
        return FindingRecord.model_validate(payload) if payload else None

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        payload = self._cache.get("evidence", evidence_id)
        return EvidenceRecord.model_validate(payload) if payload else None

    def list_evidences(self, *, scope: str | None = None) -> list[EvidenceRecord]:
        records = [
            EvidenceRecord.model_validate(v) for _, v in self._cache.items("evidence")
        ]
        return records if scope is None else [r for r in records if r.scope == scope]

    def artifact_summary(self, *, scope: str | None = None) -> dict[str, Any]:
        scopes = {
            str(value.get("scope"))
            for _, value in self._cache.items("findings")
            if value.get("scope")
        } | {
            str(value.get("scope"))
            for _, value in self._cache.items("evidence")
            if value.get("scope")
        }
        return {
            "scope": scope,
            "finding_count": len(self.list_findings(scope=scope)),
            "evidence_count": len(self.list_evidences(scope=scope)),
            "scopes": sorted(scopes),
        }

    def search_evidence(
        self, query: str, *, scope: str | None = None
    ) -> list[EvidenceRecord]:
        return self._lexical_rank(
            query,
            self.list_evidences(scope=scope),
            text_getter=_evidence_search_text,
            scope=scope,
        )

    def list_findings(self, *, scope: str | None = None) -> list[FindingRecord]:
        records = [
            FindingRecord.model_validate(v) for _, v in self._cache.items("findings")
        ]
        return records if scope is None else [r for r in records if r.scope == scope]

    def search_findings(
        self, query: str, *, scope: str | None = None
    ) -> list[FindingRecord]:
        return self._lexical_rank(
            query,
            self.list_findings(scope=scope),
            text_getter=_finding_search_text,
            scope=scope,
        )

    def hybrid_search_findings(
        self,
        query: str,
        *,
        top_k: int = 5,
        scope: str | None = None,
    ) -> list[FindingRecord]:
        return self._hybrid_search(
            query,
            namespace="finding_vectors",
            getter=self.get_finding,
            lexical=lambda q: self.search_findings(q, scope=scope),
            text_getter=_finding_search_text,
            top_k=top_k,
            scope=scope,
        )

    def hybrid_search_evidence(
        self,
        query: str,
        *,
        top_k: int = 8,
        scope: str | None = None,
    ) -> list[EvidenceRecord]:
        return self._hybrid_search(
            query,
            namespace="evidence_vectors",
            getter=self.get_evidence,
            lexical=lambda q: self.search_evidence(q, scope=scope),
            text_getter=_evidence_search_text,
            top_k=top_k,
            scope=scope,
        )

    def semantic_search_findings(
        self,
        query: str,
        *,
        top_k: int = 5,
        scope: str | None = None,
    ) -> list[FindingRecord]:
        return self._semantic_search(
            query,
            namespace="finding_vectors",
            getter=self.get_finding,
            lexical=lambda q: self.search_findings(q, scope=scope),
            top_k=top_k,
            scope=scope,
        )

    def semantic_search_evidence(
        self,
        query: str,
        *,
        top_k: int = 8,
        scope: str | None = None,
    ) -> list[EvidenceRecord]:
        return self._semantic_search(
            query,
            namespace="evidence_vectors",
            getter=self.get_evidence,
            lexical=lambda q: self.search_evidence(q, scope=scope),
            top_k=top_k,
            scope=scope,
        )

    def discard(
        self,
        *,
        kind: Literal["all", "evidence", "findings"] = "all",
        scope: str | None = None,
    ) -> None:
        namespace_map = {
            "evidence": ["evidence", "evidence_vectors"],
            "findings": ["findings", "finding_vectors"],
            "all": [
                "evidence",
                "evidence_vectors",
                "findings",
                "finding_vectors",
            ],
        }
        for namespace in namespace_map[kind]:
            if scope is None:
                self._cache.clear(namespace)
            else:
                for key, value in self._cache.items(namespace):
                    if value.get("scope") == scope:
                        self._cache.delete(namespace, key)

    def rebuild(self, df: pd.DataFrame) -> None:
        evidence_items = [
            EvidenceRecord.model_validate(v) for _, v in self._cache.items("evidence")
        ]
        findings_items = self._cache.items("findings")
        self._cache.clear("evidence")
        self._cache.clear("findings")
        self._cache.clear("evidence_vectors")
        self._cache.clear("finding_vectors")
        for record in evidence_items:
            self.save_evidence(record.evidence, df, scope=record.scope)
        for key, value in findings_items:
            self._cache.set("findings", key, value)
        for _, value in self._cache.items("findings"):
            self._index_finding(FindingRecord.model_validate(value))

    def has_artifacts(self, *, scope: str | None = None) -> bool:
        if scope is None:
            return bool(self._cache.items("findings") or self._cache.items("evidence"))
        return bool(self.list_findings(scope=scope) or self.list_evidences(scope=scope))

    def export_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        namespaces = [
            "evidence",
            "evidence_vectors",
            "findings",
            "finding_vectors",
        ]
        return {
            namespace: [
                {"key": key, "value": value}
                for key, value in self._cache.items(namespace)
            ]
            for namespace in namespaces
        }

    def import_snapshot(self, snapshot: dict[str, list[dict[str, Any]]]) -> None:
        self.discard(kind="all")
        for namespace, items in snapshot.items():
            for item in items:
                self._cache.set(namespace, item["key"], item["value"])

    def as_tools(self) -> list[StructuredTool]:
        def _tool_json(payload: Any) -> str:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        def _finding_summary(record: FindingRecord) -> dict[str, Any]:
            return {
                "finding_id": record.finding_id,
                "scope": record.scope,
                "claim": record.claim,
                "implication": record.implication,
                "confidence": record.confidence,
                "evidence_ids": record.evidence_ids,
            }

        def _evidence_summary(record: EvidenceRecord) -> dict[str, Any]:
            evidence = record.evidence
            return {
                "evidence_id": record.evidence_id,
                "scope": record.scope,
                "base_count": record.base_count,
                "match_count": record.match_count,
                "value_pct": record.value_pct,
                "evidence": {
                    "base_rule": evidence.base_rule,
                    "rule": evidence.rule,
                    "reason": evidence.reason,
                    "source_column": evidence.source_column,
                    "match_label": evidence.match_label,
                    "question_id": evidence.question_id,
                },
            }

        def _normalize_scope(scope: str = "") -> str | None:
            normalized = scope.strip()
            return normalized or None

        def _search_findings(query: str, scope: str = "") -> str:
            return _tool_json(
                [
                    _finding_summary(r)
                    for r in self.hybrid_search_findings(
                        query, scope=_normalize_scope(scope)
                    )
                ]
            )

        def _search_evidence(query: str, scope: str = "") -> str:
            return _tool_json(
                [
                    _evidence_summary(r)
                    for r in self.hybrid_search_evidence(
                        query, scope=_normalize_scope(scope)
                    )
                ]
            )

        def _list_findings(scope: str = "") -> str:
            return _tool_json(
                [
                    _finding_summary(r)
                    for r in self.list_findings(scope=_normalize_scope(scope))
                ]
            )

        return [
            StructuredTool.from_function(
                _search_findings,
                name="search_findings",
                description="Search previously saved survey findings with hybrid retrieval. Pass `scope` to limit reuse to the current question or report scope before reopening df.",
            ),
            StructuredTool.from_function(
                _search_evidence,
                name="search_evidence",
                description="Search saved survey evidences with hybrid retrieval. Pass `scope` to limit reuse to the current question or report scope before reopening df.",
            ),
            StructuredTool.from_function(
                _list_findings,
                name="list_findings",
                description="List saved survey findings. Pass `scope` to inspect the current question or report scope first.",
            ),
        ]

    def _resolve_evidence_record(
        self,
        evidence_id: str,
        evidence: Evidence,
        df: pd.DataFrame,
        *,
        scope: str | None,
    ) -> EvidenceRecord:
        evidence = canonicalize_evidence_reason(_with_question_id(evidence, scope))
        validate_evidence(evidence, df)
        from ._helpers import eval_mask

        base_mask = eval_mask(evidence.base_rule, df)
        rule_mask = eval_mask(evidence.rule, df)
        match_mask = base_mask & rule_mask
        base_count = int(base_mask.sum())
        match_count = int(match_mask.sum())
        return EvidenceRecord(
            evidence_id=evidence_id,
            scope=scope,
            evidence=evidence,
            base_count=base_count,
            match_count=match_count,
            value_pct=round(match_count / base_count * 100, 1),
            refs=extract_refs(df, match_mask, self.identity),
            response_refs=extract_response_refs(
                df, match_mask, self.identity, evidence.question_id
            ),
            preview=extract_response_previews(df, match_mask, evidence),
        )

    def _evidence_id(self, evidence: Evidence) -> str:
        return stable_evidence_id(canonicalize_evidence_reason(evidence))

    def _finding_id(self, scope: str | None, idx: int, claim: str) -> str:
        payload = f"{scope or ''}|{idx}|{claim}"
        return "fd_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _index_finding(self, record: FindingRecord) -> None:
        if self.embedding_backend is None:
            return
        vector = self.embedding_backend.embed_texts([_finding_search_text(record)])[0]
        self._cache.set(
            "finding_vectors",
            record.finding_id,
            {"scope": record.scope, "embedding": vector},
        )

    def _index_evidence(self, record: EvidenceRecord) -> None:
        if self.embedding_backend is None:
            return
        vector = self.embedding_backend.embed_texts([_evidence_search_text(record)])[0]
        self._cache.set(
            "evidence_vectors",
            record.evidence_id,
            {"scope": record.scope, "embedding": vector},
        )

    def _semantic_search(
        self,
        query: str,
        *,
        namespace: str,
        getter,
        lexical,
        top_k: int,
        scope: str | None,
    ):
        if self.embedding_backend is None:
            return lexical(query)[:top_k]
        items = self._cache.items(namespace)
        if not items:
            return lexical(query)[:top_k]
        query_vector = self.embedding_backend.embed_texts([query])[0]
        scored: list[tuple[float, Any]] = []
        for artifact_id, payload in items:
            if scope is not None and payload.get("scope") != scope:
                continue
            score = _cosine_similarity(query_vector, payload["embedding"])
            record = getter(artifact_id)
            if record is not None:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def _hybrid_search(
        self,
        query: str,
        *,
        namespace: str,
        getter,
        lexical,
        text_getter,
        top_k: int,
        scope: str | None,
    ):
        lexical_results = lexical(query)
        lexical_scores = {
            _record_id(record): _lexical_score(query, text_getter(record))
            for record in lexical_results
        }

        if self.embedding_backend is None:
            return lexical_results[:top_k]

        items = self._cache.items(namespace)
        if not items:
            return lexical_results[:top_k]

        query_vector = self.embedding_backend.embed_texts([query])[0]
        semantic_scores: dict[str, float] = {}
        records: dict[str, Any] = {}

        for artifact_id, payload in items:
            if scope is not None and payload.get("scope") != scope:
                continue
            record = getter(artifact_id)
            if record is None:
                continue
            records[artifact_id] = record
            semantic_scores[artifact_id] = _cosine_similarity(
                query_vector, payload["embedding"]
            )
            lexical_scores.setdefault(
                artifact_id, _lexical_score(query, text_getter(record))
            )

        ranked = []
        for artifact_id, record in records.items():
            semantic = semantic_scores.get(artifact_id, 0.0)
            lexical_score = lexical_scores.get(artifact_id, 0.0)
            score = (0.65 * semantic) + (0.35 * lexical_score)
            ranked.append((score, record))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in ranked[:top_k]]

    def _lexical_rank(self, query: str, records, *, text_getter, scope: str | None):
        scored = []
        for record in records:
            if scope is not None and getattr(record, "scope", None) != scope:
                continue
            score = _lexical_score(query, text_getter(record))
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored]


def _finding_search_text(record: FindingRecord) -> str:
    return "\n".join(
        filter(None, [record.claim, record.implication, record.scope or ""])
    )


def _evidence_search_text(record: EvidenceRecord) -> str:
    evidence = record.evidence
    return "\n".join(
        filter(
            None,
            [
                evidence.question_id or "",
                evidence.source_column or "",
                evidence.reason,
                evidence.match_label or "",
            ],
        )
    )


def _record_id(record: Any) -> str:
    if isinstance(record, FindingRecord):
        return record.finding_id
    if isinstance(record, EvidenceRecord):
        return record.evidence_id
    raise TypeError(f"unsupported record type: {type(record).__name__}")


def _lexical_score(query: str, text: str) -> float:
    q = _normalize_search_text(query)
    t = _normalize_search_text(text)
    if not q or not t:
        return 0.0

    if q in t:
        return 1.0

    q_tokens = set(_tokenize(q))
    t_tokens = set(_tokenize(t))
    token_overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
    ratio = SequenceMatcher(None, q, t).ratio()
    partial = max(
        (
            SequenceMatcher(None, q, chunk).ratio()
            for chunk in _window_chunks(t, len(q))
        ),
        default=0.0,
    )
    return max(token_overlap, 0.55 * ratio + 0.45 * partial)


def _normalize_search_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ÿ]+", text.casefold())


def _window_chunks(text: str, query_len: int) -> list[str]:
    if query_len <= 0 or len(text) <= query_len:
        return [text]
    step = max(query_len // 2, 1)
    return [text[i : i + query_len] for i in range(0, len(text) - query_len + 1, step)]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _with_question_id(evidence: Evidence, scope: str | None) -> Evidence:
    if evidence.question_id is not None:
        return evidence
    derived_question_id = evidence.source_column
    if (
        derived_question_id is None
        and isinstance(scope, str)
        and scope.startswith("question:")
    ):
        derived_question_id = scope.split(":", 1)[1]
    if derived_question_id is None:
        return evidence
    return evidence.model_copy(update={"question_id": derived_question_id})
