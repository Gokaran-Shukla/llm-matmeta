"""Shared prompt construction and the validate-and-retry loop."""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient
from .schema import validate_against


def user_prompt(schema: dict[str, Any], text: str, errors: list[str] | None = None, extra: str = "") -> str:
    msg = f"JSON schema:\n{json.dumps(schema, indent=1)}\n\n{extra}Source text:\n{text}\n"
    if errors:
        msg += "\nYour previous answer failed validation with these errors; return a corrected object:\n"
        msg += "\n".join(f"- {e}" for e in errors)
    return msg


def ask_llm(
    client: LLMClient,
    system: str,
    schema: dict[str, Any],
    text: str,
    retries: int = 1,
    extra: str = "",
) -> tuple[Any, list[str]]:
    """Ask for JSON matching ``schema``; on validation failure feed the errors back ``retries`` times."""
    reply = client.complete_json(system, user_prompt(schema, text, extra=extra))
    errors = validate_against(reply, schema)
    for _ in range(retries):
        if not errors:
            break
        reply = client.complete_json(system, user_prompt(schema, text, errors, extra=extra))
        errors = validate_against(reply, schema)
    return reply, errors
