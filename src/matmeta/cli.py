"""Command-line interface.

Examples
--------
    matmeta calc examples/vasp_Fe_bcc                   # LLM + parser (needs an LLM endpoint)
    matmeta calc examples/qe_Si/si.scf.in --no-llm      # parser only, fully offline
    matmeta calc runs/*/ --jsonld -o metadata/          # batch, JSON-LD output
    matmeta calc examples/vasp_Fe_soc --extras          # also read .wout files and job scripts
    matmeta file examples/vasp_Fe_soc/wannier90.wout    # LLM only, every value checked against its quote
    matmeta abstract examples/abstract.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import extract_abstract, extract_calculation
from .files import extract_file
from .llm import OpenAICompatibleClient
from .mapping import load_context, to_jsonld


def _write(obj, out_dir: Path | None, stem: str) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    if out_dir is None:
        print(text)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.json").write_text(text + "\n")
        print(f"wrote {out_dir / (stem + '.json')}", file=sys.stderr)


def _stem(p: str) -> str:
    path = Path(p).resolve()
    return path.name or "run"


def _report(label: str, res: dict) -> None:
    if res.get("llm_error"):
        what = "record is parser-only" if "conflicts" in res else "record is empty"
        print(f"[warning] {label}: LLM call failed, {what}: {res['llm_error']}", file=sys.stderr)
    if res.get("llm_validation_errors"):
        print(f"[warning] {label}: LLM reply failed schema validation and was discarded", file=sys.stderr)
    for conf in res.get("conflicts", []):
        print(f"[conflict] {label}: {conf['field']} parser={conf['parser']!r} llm={conf['llm']!r}", file=sys.stderr)
    for field in res.get("unverified", []):
        print(f"[unverified] {label}: {field} = {_get(res['metadata'], field)!r} ({res['provenance'][field]})",
              file=sys.stderr)
    for att in res.get("attachments", []):
        _report(f"{label}/{att['file']}", att)


def _get(rec: dict, path: str):
    for part in path.split("."):
        rec = rec.get(part) if isinstance(rec, dict) else None
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="matmeta", description="Extract metadata from materials-science data with LLMs.")
    ap.add_argument("--model", help="model name (default: $MATMETA_MODEL or gpt-4o-mini)")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $MATMETA_BASE_URL)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calc", help="VASP run directories or QE input files")
    c.add_argument("paths", nargs="+")
    c.add_argument("--no-llm", action="store_true", help="rule-based parsing only")
    c.add_argument("--extras", action="store_true",
                   help="also read Wannier90 .wout files and job scripts in each run directory (needs an LLM)")
    c.add_argument("--strict", action="store_true", help="drop LLM-only values whose quote check fails")
    c.add_argument("--jsonld", action="store_true", help="emit JSON-LD instead of the full report")
    c.add_argument("--context", help="JSON file with a custom JSON-LD context (ontology mapping)")
    c.add_argument("-o", "--out-dir", type=Path)

    f = sub.add_parser("file", help="any text file without a parser (LLM only, values checked against quotes)")
    f.add_argument("files", nargs="+", type=Path)
    f.add_argument("--schema", default="auto", choices=["auto", "wannier", "jobscript", "calculation"],
                   help="record type (auto: .wout -> wannier, batch scripts -> jobscript, else calculation)")
    f.add_argument("--strict", action="store_true", help="drop values whose quote check fails")
    f.add_argument("-o", "--out-dir", type=Path)

    a = sub.add_parser("abstract", help="text file(s) containing paper abstracts")
    a.add_argument("files", nargs="+", type=Path)
    a.add_argument("-o", "--out-dir", type=Path)

    args = ap.parse_args(argv)
    client = OpenAICompatibleClient(base_url=args.base_url, model=args.model)
    failures = 0

    if args.cmd == "calc":
        if args.extras and args.no_llm:
            print("[warning] --extras needs an LLM; ignored with --no-llm", file=sys.stderr)
        ctx = load_context(args.context) if args.context else None
        for p in args.paths:
            try:
                res = extract_calculation(p, None if args.no_llm else client, extras=args.extras, strict=args.strict)
            except Exception as exc:  # keep going in batch mode
                print(f"[error] {p}: {exc}", file=sys.stderr)
                failures += 1
                continue
            _report(p, res)
            _write(to_jsonld(res, ctx, record_id=str(p)) if args.jsonld else res, args.out_dir, _stem(p))
        return 1 if failures else 0

    if args.cmd == "file":
        for fp in args.files:
            try:
                res = extract_file(fp, client, args.schema, strict=args.strict)
            except Exception as exc:
                print(f"[error] {fp}: {exc}", file=sys.stderr)
                failures += 1
                continue
            _report(str(fp), res)
            failures += bool(res["llm_error"])
            _write(res, args.out_dir, fp.name)
        return 1 if failures else 0

    for fp in args.files:
        res = extract_abstract(fp.read_text(), client)
        _report(str(fp), res)
        _write(res, args.out_dir, fp.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
