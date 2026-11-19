"""Hybrid extraction pipeline: rule-based parsing + LLM + schema validation + cross-checking.

Design
------
1. A deterministic parser reads the raw files and produces (a) a *trusted* partial record and
   (b) a short evidence text (input files verbatim, selected output lines).
2. The LLM sees **only the evidence text and the JSON schema**, not the rule-based draft, so its
   answer is an independent second reading of the same evidence.
3. The LLM reply is validated against the schema; on failure the errors are fed back once.
4. Both records are merged field by field. Deterministic values win for fields the parser
   understands; disagreements are reported as conflicts (useful to catch LLM hallucinations
   *and* parser blind spots). Every field records where its value came from.
5. If the LLM cannot be reached or answers nonsense, the parser record is still returned.

Files without a parser (Wannier90 ``.wout``, job scripts) are handled in :mod:`matmeta.files`
and can be attached to a calculation record with ``extras=True``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .files import extract_file, find_extras
from .llm import LLMClient
from .parsers import ParsedCalculation, detect_and_parse
from .prompting import ask_llm
from .schema import SCHEMAS, empty_calculation, validate

SYSTEM_PROMPT = (
    "You are a careful research-data curator for computational materials science. "
    "Extract metadata from the provided files into JSON that strictly follows the given JSON schema, "
    "including the conventions stated in the schema descriptions. "
    "Use only information present in the text; if a value is not stated or cannot be inferred with "
    "certainty, use null. Convert energies to eV. Reply with a single JSON object and nothing else."
)

ABSTRACT_SYSTEM_PROMPT = (
    "You are a research-data curator for materials science. From a paper abstract, extract the materials "
    "studied, the methods used (computational, experimental or theoretical) and the physical properties "
    "reported, following the given JSON schema. Do not invent values. Reply with a single JSON object."
)


def _equal(a: Any, b: Any, rtol: float = 1e-3) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-6)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y, rtol) for x, y in zip(a, b))
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
            merged[key], provenance, conflicts = m, {**provenance, **p}, conflicts + c
        elif r is not None:
            merged[key] = r
            if l is None:
                provenance[path] = "parser"
            elif _equal(r, l):
                provenance[path] = "parser+llm"
            else:
                provenance[path] = "parser (llm disagreed)"
                conflicts.append({"field": path, "parser": r, "llm": l})
        else:
            merged[key] = l
            provenance[path] = "llm" if l is not None else "missing"
    return merged, provenance, sorted(conflicts, key=lambda c: c["field"])


def extract_calculation(
    path: str | Path,
    client: LLMClient | None = None,
    extras: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Extract metadata for a VASP directory or a QE input file.

    With ``client=None`` only the deterministic parser is used (no LLM call). With ``extras=True``
    (and a client) Wannier90 ``.wout`` files and job scripts in the run directory are read too and
    attached under ``attachments``.
    """
    parsed: ParsedCalculation = detect_and_parse(path)
    llm_record, llm_errors, llm_error = empty_calculation(), [], None
    if client is not None:
        try:
            llm_record, llm_errors = ask_llm(client, SYSTEM_PROMPT, SCHEMAS["calculation"], parsed.evidence)
        except Exception as exc:  # unreachable server, bad key, timeout, unparsable reply ...
            llm_record, llm_error = empty_calculation(), f"{type(exc).__name__}: {exc}"
        if llm_errors or not isinstance(llm_record, dict):  # never let an invalid answer contaminate the record
            llm_record = empty_calculation()

    record, provenance, conflicts = merge(parsed.record, llm_record)
    record = {k: record[k] for k in empty_calculation()}  # stable key order
    result: dict[str, Any] = {
        "metadata": record,
        "provenance": dict(sorted(provenance.items())),
        "conflicts": conflicts,
        "validation_errors": validate(record, "calculation"),
        "llm_validation_errors": llm_errors,
        "llm_error": llm_error,
        "source": {
            "path": str(path),
            "code": parsed.code,
            "files": parsed.files,
            "structure_from": parsed.structure_file,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extractor": f"llm-matmeta {__version__}",
            "llm_used": client is not None and llm_error is None,
        },
    }
    if extras:
        result["attachments"] = []
        run_dir = Path(path) if Path(path).is_dir() else Path(path).parent
        if client is not None:
            for f, kind in find_extras(run_dir):
                att = extract_file(f, client, kind, strict=strict)
                att["file"] = f.name
                result["attachments"].append(att)
    return result


def extract_abstract(text: str, client: LLMClient) -> dict[str, Any]:
    """Extract materials / methods / properties from a paper abstract."""
    llm_error = None
    try:
        record, errors = ask_llm(client, ABSTRACT_SYSTEM_PROMPT, SCHEMAS["abstract"], text)
    except Exception as exc:
        record, errors, llm_error = {"materials": [], "methods": [], "properties": [], "keywords": []}, [], str(exc)
    return {
        "metadata": record,
        "validation_errors": errors,
        "llm_error": llm_error,
        "source": {
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extractor": f"llm-matmeta {__version__}",
        },
    }
