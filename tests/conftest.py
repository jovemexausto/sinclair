"""Shared test fixtures."""
from __future__ import annotations

import pytest
from pydantic import BaseModel


class SimpleOutput(BaseModel):
    answer: str
    confidence: float


class StrictOutput(BaseModel):
    trend: str
    peak_month: str
    avg_revenue: float


@pytest.fixture
def simple_schema():
    return SimpleOutput


@pytest.fixture
def strict_schema():
    return StrictOutput