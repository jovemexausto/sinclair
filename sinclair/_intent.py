from __future__ import annotations

import re


_MAX_INTENT_LENGTH = 120
_FIRST_PERSON_PREFIXES = (
    "estou ",
    "estou a ",
    "i am ",
    "i'm ",
    "estoy ",
)


def validate_intent_text(value: str) -> str:
    raw = str(value or "")
    if "\n" in raw:
        raise ValueError("intent must be a single sentence")
    intent = " ".join(raw.strip().split())
    if not intent:
        raise ValueError("intent is required")
    if len(intent) > _MAX_INTENT_LENGTH:
        raise ValueError(
            f"intent must be at most {_MAX_INTENT_LENGTH} characters"
        )
    if not intent.casefold().startswith(_FIRST_PERSON_PREFIXES):
        raise ValueError(
            "intent must be a short first-person status in progress, high-level and product-facing only; it should read like a user-facing progress update, not a debug log. Do not mention internal errors, tool names, ids, regex, or columns. Example: 'Estou ajustando o recorte principal.'"
        )
    if len(re.findall(r"[.!?]+", intent)) > 1:
        raise ValueError("intent must be a single sentence")
    return intent
