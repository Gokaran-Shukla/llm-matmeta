"""Command-line interface.

Examples
--------
    matmeta calc examples/vasp_Fe_bcc                 # LLM + parser (needs an LLM endpoint)
    matmeta calc examples/qe_Si/si.scf.in --no-llm    # parser only, fully offline
    matmeta calc runs/*/ --jsonld -o metadata/        # batch, JSON-LD output
    matmeta abstract examples/abstract.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import extract_abstract, extract_calculation
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="matmeta", description="Extract metadata from materials-science data with LLMs.")
    ap.add_argument("--model", help="model name (default: $MATMETA_MODEL or gpt-4o-mini)")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $MATMETA_BASE_URL)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calc", help="VASP run directories or QE input files")
    c.add_argument("paths", nargs="+")
    c.add_argument("--no-llm", action="store_true", help="rule-based parsing only")
    c.add_argument("--jsonld", action="store_true", help="emit JSON-LD instead of the full report")
    c.add_argument("--context", help="JSON file with a custom JSON-LD context (ontology mapping)")
    c.add_argument("-o", "--out-dir", type=Path)

    a = sub.add_parser("abstract", help="text file(s) containing paper abstracts")
    a.add_argument("files", nargs="+", type=Path)
    a.add_argument("-o", "--out-dir", type=Path)

    args = ap.parse_args(argv)
    client = OpenAICompatibleClient(base_url=args.base_url, model=args.model)

    if args.cmd == "calc":
        ctx = load_context(args.context) if args.context else None
        failures = 0
        for p in args.paths:
            try:
                res = extract_calculation(p, None if args.no_llm else client)
            except Exception as exc:  # keep going in batch mode
                print(f"[error] {p}: {exc}", file=sys.stderr)
                failures += 1
                continue
            for conf in res["conflicts"]:
                print(f"[conflict] {p}: {conf['field']} parser={conf['parser']!r} llm={conf['llm']!r}", file=sys.stderr)
            _write(to_jsonld(res, ctx, record_id=str(p)) if args.jsonld else res, args.out_dir, Path(p).name or "run")
        return 1 if failures else 0

    for f in args.files:
        _write(extract_abstract(f.read_text(), client), args.out_dir, f.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
