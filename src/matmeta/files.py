"""LLM-only extraction for files that have no rule-based parser.

Wannier90 ``.wout`` files, batch job scripts, output of codes without a parser (SIESTA, ...),
logs of in-house codes: the model is the only reader, so there is nothing to cross-check against.
Instead every value must come with the line it was read from, and the line is checked:

1. the quoted line must occur in the file (whitespace-insensitive), and
2. the value must occur in the quoted line (numbers to 1e-4 relative; Ry, Ha and Bohr unit
   conversions are recognised).

A value that fails either check is marked ``unverified`` (or removed with ``strict=True``). This
catches most invented values cheaply and works for any text format.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .llm import LLMClient
from .parsers import BOHR_TO_ANG, HA_TO_EV, RY_TO_EV, _FLOAT, _read
from .prompting import ask_llm
from .schema import SCHEMAS, empty_record

FILE_SYSTEM_PROMPT = (
    "You are a careful research-data curator for computational materials science. Read the source "
    "text and reply with ONE JSON object with exactly two keys: 'record', which must follow the given "
    "JSON schema, and 'quotes', which maps the dotted path of every non-null field in 'record' "
    "(for example 'wannierisation.omega_total_A2') to the line of the source text the value was read "
    "from, copied character for character. Use only information written in the text; use null when a "
    "value is not stated. Do not compute values except the unit conversions the schema asks for. "
    "Reply with the JSON object and nothing else."
)

KEYWORDS = {
    "wannier": ["release", "version", "number of wannier", "number of input bloch", "number of bands",
                "grid size", "outer:", "inner:", "omega", "final state", "convergence criteria",
                "total number of iterations", "<-- dis", "spinor", "num_wann", "dis_", "final spread"],
    "jobscript": [],  # job scripts are short: sent whole
    "calculation": ["version", "release", "energy", "fermi", "converge", "cutoff", "k-point", "kpoint", "kgrid",
                    "meshcutoff", "functional", "xc", "spin", "magnet", "smearing", "temperature", "atoms",
                    "lattice", "cell"],
}

# values that are named rather than written: accept them if a tell-tale token is in the quote
ALIASES = {"slurm": ("#sbatch", "srun", "slurm"), "pbs": ("#pbs", "qsub"), "lsf": ("#bsub",), "sge": ("#$",),
           "wannier90": ("wannier90", "wannier 90")}

UNIT_FACTORS = (1.0, RY_TO_EV, HA_TO_EV, BOHR_TO_ANG, BOHR_TO_ANG ** 2, 1 / RY_TO_EV)


# --------------------------------------------------------------------------------------
# detection and condensation
# --------------------------------------------------------------------------------------
def looks_like_jobscript(path: Path, max_bytes: int = 200_000) -> bool:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return False
        head = _read(path, 20_000)
    except (OSError, UnicodeDecodeError):
        return False
    if "\x00" in head:
        return False
    return bool(re.search(r"^#(SBATCH|PBS|BSUB|\$)\s", head, re.M)) or (
        head.startswith("#!") and bool(re.search(r"\b(srun|mpirun|mpiexec)\b", head))
    )


def detect_kind(path: Path) -> str:
    if path.suffix == ".wout":
        return "wannier"
    if looks_like_jobscript(path):
        return "jobscript"
    return "calculation"


def condense(text: str, keywords: list[str], max_chars: int = 24_000, head: int = 80, tail: int = 150,
             context: int = 1, max_hits: int = 300) -> tuple[str, bool]:
    """Return (excerpt, condensed?). Short texts are sent whole; long ones as head + tail + keyword hits."""
    if len(text) <= max_chars:
        return text, False
    lines = text.splitlines()
    keep: set[int] = set(range(min(head, len(lines)))) | set(range(max(0, len(lines) - tail), len(lines)))
    hits = 0
    for i, ln in enumerate(lines):
        low = ln.lower()
        if hits < max_hits and any(k in low for k in keywords):
            keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
            hits += 1
    out: list[str] = []
    prev = -1
    for i in sorted(keep):
        if i > prev + 1:
            out.append(f"[... {i - prev - 1} lines skipped ...]")
        out.append(lines[i])
        prev = i
    if prev < len(lines) - 1:
        out.append(f"[... {len(lines) - 1 - prev} lines skipped ...]")
    return "\n".join(out), True


# --------------------------------------------------------------------------------------
# quote verification
# --------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join(s.split())


def _number_in(value: float, quote: str) -> bool:
    nums = [float(x) for x in re.findall(_FLOAT, quote.replace("D", "E").replace("d", "e"))]
    for n in nums:
        for f in UNIT_FACTORS:
            if math.isclose(n * f, float(value), rel_tol=1e-4, abs_tol=1e-8):
                return True
    return False


def check_quote(value: Any, quote: Any, source_norm: str) -> str:
    """Provenance label for one LLM value."""
    if value is None:
        return "missing"
    if not isinstance(quote, str) or not quote.strip():
        return "llm (unverified: no quote)"
    parts = [_norm(q) for q in quote.splitlines() if q.strip()]
    if not all(p in source_norm for p in parts):
        return "llm (unverified: quote not in file)"
    q = " ".join(parts)
    if isinstance(value, bool):
        return "llm (quote found)"
    if isinstance(value, (int, float)):
        return "llm (verified)" if _number_in(value, q) else "llm (unverified: value not in quote)"
    if isinstance(value, str):
        v, ql = _norm(value).lower(), q.lower()
        if v in ql or any(tok in ql for tok in ALIASES.get(v, ())):
            return "llm (verified)"
        return "llm (unverified: value not in quote)"
    if isinstance(value, list):
        ok = all(
            _number_in(v, q) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v).lower() in q.lower()
            for v in value
        )
        return "llm (verified)" if ok else "llm (unverified: value not in quote)"
    return "llm (quote found)"


def _flatten(rec: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _set(rec: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for p in parents:
        rec = rec[p]
    rec[leaf] = value


def _reply_schema(kind: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["record", "quotes"],
        "properties": {
            "record": SCHEMAS[kind],
            "quotes": {"type": "object", "additionalProperties": {"type": ["string", "null"]}},
        },
    }


# --------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------
def extract_text(text: str, client: LLMClient, kind: str, strict: bool = False, max_chars: int = 24_000) -> dict:
    """LLM-only extraction of ``text`` into a record of type ``kind`` with quote verification."""
    if kind not in KEYWORDS:
        raise ValueError(f"unknown schema {kind!r}; choose from {sorted(KEYWORDS)}")
    excerpt, condensed = condense(text, KEYWORDS[kind], max_chars=max_chars)
    record, quotes, errors, llm_error = empty_record(SCHEMAS[kind]), {}, [], None
    try:
        reply, errors = ask_llm(client, FILE_SYSTEM_PROMPT, _reply_schema(kind), excerpt)
        if not errors:
            record, quotes = reply["record"], reply["quotes"] or {}
    except Exception as exc:  # network, server or JSON errors: return an empty record, keep going
        llm_error = f"{type(exc).__name__}: {exc}"

    # fill keys the model left out so every record has the full shape
    full = empty_record(SCHEMAS[kind])
    for path, val in _flatten(record).items():
        try:
            _set(full, path, val)
        except (KeyError, TypeError):
            pass
    record = full

    src = _norm(text)
    provenance: dict[str, str] = {}
    for path, val in _flatten(record).items():
        label = check_quote(val, quotes.get(path), src)
        if strict and label.startswith("llm (unverified"):
            _set(record, path, None)
            label = "dropped " + label[len("llm "):]
        provenance[path] = label
    unverified = sorted(p for p, lab in provenance.items() if "unverified" in lab)
    return {
        "metadata": record,
        "provenance": dict(sorted(provenance.items())),
        "quotes": {p: quotes.get(p) for p in sorted(provenance) if quotes.get(p)},
        "unverified": unverified,
        "llm_validation_errors": errors,
        "llm_error": llm_error,
        "source": {
            "schema": kind,
            "chars_total": len(text),
            "chars_sent": len(excerpt),
            "condensed": condensed,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extractor": f"llm-matmeta {__version__}",
            "llm_used": True,
            "strict": strict,
        },
    }


def extract_file(path: str | Path, client: LLMClient, kind: str = "auto", strict: bool = False) -> dict:
    p = Path(path)
    kind = detect_kind(p) if kind == "auto" else kind
    res = extract_text(_read(p), client, kind, strict=strict)
    res["source"] = {"path": str(p), **res["source"]}
    return res


def find_extras(directory: str | Path) -> list[tuple[Path, str]]:
    """Files in a run directory that have no parser but are worth reading: .wout and job scripts."""
    d = Path(directory)
    found = [(p, "wannier") for p in sorted(d.glob("*.wout"))]
    skip = {"INCAR", "POSCAR", "CONTCAR", "KPOINTS", "OUTCAR", "POTCAR", "OSZICAR", "std.out", "std.err"}
    for p in sorted(d.iterdir()):
        if p.name not in skip and p.suffix not in {".py", ".wout"} and looks_like_jobscript(p):
            found.append((p, "jobscript"))
    return found
