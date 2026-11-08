"""Hybrid extraction pipeline: rule-based parsing + LLM + schema validation + cross-checking.

Design
------
1. A deterministic parser reads the raw files and produces (a) a *trusted* partial record and
   (b) a short evidence text.
2. The LLM sees **only the evidence text and the JSON schema**, not the rule-based draft, so its
   answer is an independent second reading of the same files.
3. The LLM reply is validated against the schema; on failure the errors are fed back once.
4. Both records are merged field by field. Deterministic values win for fields the parser
   understands; disagreements are reported as conflicts (useful to catch LLM hallucinations
   *and* parser blind spots). Every field records where its value came from.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .llm import LLMClient
from .parsers import ParsedCalculation, detect_and_parse
from .schema import SCHEMAS, empty_calculation, validate

SYSTEM_PROMPT = (
    "You are a careful research-data curator for computational materials science. "
    "Extract metadata from the provided files into JSON that strictly follows the given JSON schema. "
    "Use only information present in the text; if a value is not stated or cannot be inferred with "
    "certainty, use null. Convert energies to eV. Reply with a single JSON object and nothing else."
)

ABSTRACT_SYSTEM_PROMPT = (
    "You are a research-data curator for materials science. From a paper abstract, extract the materials "
    "studied, the methods used (computational, experimental or theoretical) and the physical properties "
    "reported, following the given JSON schema. Do not invent values. Reply with a single JSON object."
)


def _user_prompt(schema: dict[str, Any], text: str, errors: list[str] | None = None) -> str:
    msg = f"JSON schema:\n{json.dumps(schema, indent=1)}\n\nSource text:\n{text}\n"
    if errors:
        msg += "\nYour previous answer failed validation with these errors; return a corrected object:\n"
        msg += "\n".join(f"- {e}" for e in errors)
    return msg


def _ask_llm(client: LLMClient, system: str, schema: dict[str, Any], kind: str, text: str, retries: int = 1):
    reply = client.complete_json(system, _user_prompt(schema, text))
    errors = validate(reply, kind)
    for _ in range(retries):
        if not errors:
            break
        reply = client.complete_json(system, _user_prompt(schema, text, errors))
        errors = validate(reply, kind)
    return reply, errors


def _equal(a: Any, b: Any, rtol: float = 1e-3) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-6)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def merge(rule: dict[str, Any], llm: dict[str, Any], prefix: str = ""):
    """Merge two records. Returns (merged, provenance, conflicts)."""
    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for key in rule.keys() | llm.keys():
        path = f"{prefix}{key}"
        r, l = rule.get(key), llm.get(key)
        if isinstance(r, dict) or isinstance(l, dict):
            m, p, c = merge(r or {}, l or {}, prefix=f"{path}.")
            merged[key], provenance, conflicts = m, provenance | p, conflicts + c
        elif r is not None:
            merged[key] = r
            provenance[path] = "parser" if l is None or _equal(r, l) else "parser (llm disagreed)"
            if l is not None and not _equal(r, l):
                conflicts.append({"field": path, "parser": r, "llm": l})
            elif l is not None:
                provenance[path] = "parser+llm"
        else:
            merged[key] = l
            provenance[path] = "llm" if l is not None else "missing"
    return merged, provenance, conflicts


def extract_calculation(path: str | Path, client: LLMClient | None = None) -> dict[str, Any]:
    """Extract metadata for a VASP directory or a QE input file.

    With ``client=None`` only the deterministic parser is used (no LLM call).
    """
    parsed: ParsedCalculation = detect_and_parse(path)
    llm_record, llm_errors = empty_calculation(), []
    if client is not None:
        llm_record, llm_errors = _ask_llm(
            client, SYSTEM_PROMPT, SCHEMAS["calculation"], "calculation", parsed.evidence
        )
        if llm_errors:  # never let an invalid LLM answer contaminate the record
            llm_record = empty_calculation()

    record, provenance, conflicts = merge(parsed.record, llm_record)
    record = {k: record[k] for k in empty_calculation()}  # stable key order
    return {
        "metadata": record,
        "provenance": dict(sorted(provenance.items())),
        "conflicts": conflicts,
        "validation_errors": validate(record, "calculation"),
        "llm_validation_errors": llm_errors,
        "source": {
            "path": str(path),
            "code": parsed.code,
            "files": parsed.files,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extractor": f"llm-matmeta {__version__}",
            "llm_used": client is not None,
        },
    }


def extract_abstract(text: str, client: LLMClient) -> dict[str, Any]:
    """Extract materials / methods / properties from a paper abstract."""
    record, errors = _ask_llm(client, ABSTRACT_SYSTEM_PROMPT, SCHEMAS["abstract"], "abstract", text)
    return {
        "metadata": record,
        "validation_errors": errors,
        "source": {
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extractor": f"llm-matmeta {__version__}",
        },
    }
