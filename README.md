# llm-matmeta

📘 **Complete manual (PDF):** [docs/llm-matmeta_manual.pdf](docs/llm-matmeta_manual.pdf) — documentation, tutorials and quizzes in one file.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24334736.svg)](https://doi.org/10.5281/zenodo.24334736)
[![tests](https://github.com/Gokaran-Shukla/llm-matmeta/actions/workflows/tests.yml/badge.svg)](https://github.com/Gokaran-Shukla/llm-matmeta/actions/workflows/tests.yml)

**Hybrid rule-based + LLM metadata extraction for computational materials-science data.**

📄 **Full formalism:** [`docs/llm-matmeta-formalism.pdf`](docs/llm-matmeta-formalism.pdf), with definitions of every
operator (evidence, merge, provenance, quote verification), the VASP default rules, and what each check can
and cannot guarantee.
🎞️ **Slides:** [`docs/llm-matmeta-slides.pdf`](docs/llm-matmeta-slides.pdf) (Beamer, 16:9; source `docs/slides.tex`).

`llm-matmeta` turns the semi-structured output of DFT codes (VASP, Quantum ESPRESSO) and the
unstructured text of paper abstracts into validated, machine-readable metadata (JSON and JSON-LD).
It is a small prototype for making simulation data **FAIR** (findable, accessible, interoperable,
reusable) without asking researchers to fill in metadata forms by hand.

## Why hybrid?

LLMs are good at reading heterogeneous text, but they can misread numbers or pick the wrong line in
a long log. Rule-based parsers are exact, but brittle and blind to anything they were not written
for. This tool uses both and checks one against the other:

```
 raw files ──► rule-based parser ──► trusted draft ─────────────┐
     │                │                                         ├─► merge ─► validated record
     │                └──► condensed evidence ─► LLM ─► JSON ───┘     │       + provenance per field
     │                                          (schema-validated,    │       + list of conflicts
     │                                           retried on error)    └─► JSON-LD (ontology mapping)
```

* **Context-efficient:** only the relevant lines of an OUTCAR / pw.x output (often hundreds of MB)
  are sent to the model.
* **Independent second reading:** the LLM never sees the parser's draft, so agreement is meaningful.
* **Schema-constrained:** replies are validated with JSON Schema; invalid replies are retried once
  with the validation errors, then discarded rather than allowed to corrupt the record.
* **Provenance and conflicts:** every field records whether it came from the `parser`, the `llm`,
  or both (`parser+llm`). Disagreements are listed, which catches LLM hallucinations *and* parser
  blind spots.
* **Provider-agnostic:** works with any OpenAI-compatible endpoint, including local open-weight
  models through Ollama or vLLM, so research data never has to leave the institution.
* **LLM-only files with quote checking (v0.2):** for files no parser understands (Wannier90 `.wout`,
  batch job scripts, other codes' output), the model must return the exact source line for every value.
  matmeta checks that the line is in the file and contains the value; failures are flagged, or removed with `--strict`.
* **Ontology mapping:** records can be exported as JSON-LD with a user-supplied context that maps
  fields to the terms of a target ontology.

## Installation

```bash
git clone https://github.com/Gokaran-Shukla/llm-matmeta.git
cd llm-matmeta
python -m pip install --upgrade pip   # editable installs need pip >= 21.3
pip install -e ".[dev]"
pytest -q          # 31 tests, run fully offline
```

## Configure an LLM

```bash
# OpenAI (or any hosted OpenAI-compatible API)
export MATMETA_API_KEY=sk-...
export MATMETA_MODEL=gpt-4o-mini

# or a local open-weight model with Ollama
ollama pull llama3.1
export MATMETA_BASE_URL=http://localhost:11434/v1
export MATMETA_MODEL=llama3.1
```

## Usage

```bash
# a VASP run directory (INCAR, POSCAR, KPOINTS, OUTCAR)
matmeta calc examples/vasp_Fe_bcc

# a Quantum ESPRESSO input (the matching .out file is picked up automatically)
matmeta calc examples/qe_Si/si.scf.in

# parser only, no LLM call (offline)
matmeta calc examples/vasp_Fe_bcc --no-llm

# batch processing, JSON-LD output, custom ontology mapping
matmeta calc runs/*/ --jsonld --context my_ontology_context.json -o metadata/

# also read Wannier90 .wout files and job scripts found in the run directory
matmeta calc examples/vasp_Fe_soc --extras

# any file without a parser: LLM only, every value checked against its quoted line
matmeta file examples/vasp_Fe_soc/wannier90.wout          # schema detected: wannier
matmeta file examples/vasp_Fe_soc/z_jobs_ibex             # schema detected: jobscript
matmeta file siesta.out --schema calculation --strict     # drop values that fail the check

# materials / methods / properties from paper abstracts
matmeta abstract examples/abstract.txt
```

Diagnostics go to stderr: `[conflict]` (parser and LLM disagree), `[unverified]` (quote check failed),
`[warning]` (LLM unreachable or invalid reply; the parser record is still written) and `[error]`.

From Python:

```python
from matmeta import extract_calculation, to_jsonld
from matmeta.llm import OpenAICompatibleClient

result = extract_calculation("examples/vasp_Fe_bcc", OpenAICompatibleClient())
print(result["metadata"]["results"]["total_energy_eV"])
print(result["conflicts"])           # fields where LLM and parser disagree
doc = to_jsonld(result)              # JSON-LD for a knowledge graph
```

Example output (parser-only mode) is in [`examples/output/`](examples/output/).

## What is extracted

| Group      | Fields |
|------------|--------|
| software   | name, version |
| method     | task (scf / relax / vc-relax / nscf / bands / md / phonon), XC functional, spin polarisation, spin–orbit coupling, noncollinear magnetism, DFT+U, van der Waals correction |
| structure  | formula, number of atoms, species, lattice vectors (Å) |
| numerics   | plane-wave cutoff, k-point mesh, smearing type and width, energy convergence threshold (all in eV) |
| results    | convergence, total (free) energy, energy(σ→0), Fermi energy, total magnetisation and, for noncollinear runs, the (mx, my, mz) vector |
| Wannier90  | release, num_wann, num_bands, MP grid, spinors, outer/frozen windows, disentanglement convergence and Ω_I, wannierisation convergence, Ω_I, Ω_D, Ω_OD, total spread (Å²) |
| job scripts| scheduler, job name, partition, nodes, tasks, CPUs, GPUs, memory, time limit, constraint, loaded modules, executable, launcher |
| abstracts  | materials (formula, dimensionality, role), methods (computational / experimental / theoretical, software), reported properties with values and units, keywords |

Units are normalised (Ry → eV for Quantum ESPRESSO). For VASP, tags missing from the INCAR are taken from
the parameter block VASP prints in the OUTCAR (with VASP's own defaults as a last resort), relaxations report
the CONTCAR structure, and OUTCARs are streamed (constant memory, `.gz` accepted). The full schemas are in
[`src/matmeta/schema.py`](src/matmeta/schema.py).

## Project layout

```
src/matmeta/
  parsers.py   streamed rule-based VASP and Quantum ESPRESSO parsers + evidence text
  llm.py       OpenAI-compatible client and an offline mock client
  prompting.py prompt construction and the validate-and-retry loop
  extract.py   hybrid pipeline: merge, provenance, conflicts, LLM fallback, --extras
  files.py     LLM-only files: condensation, quote verification, file-type detection
  schema.py    JSON schemas: calculation, abstract, wannier, jobscript
  mapping.py   JSON-LD export with configurable ontology context
  cli.py       `matmeta` command-line tool
tests/         offline unit tests (parsers, merge logic, retries, HTTP client, CLI)
examples/      small VASP and QE runs (incl. a SOC + Wannier90 + SLURM run) and an example abstract
docs/          formalism (LaTeX source and PDF)
```

## Roadmap

- Benchmark extraction accuracy of several open-weight and hosted LLMs on a curated set of real
  calculations (precision/recall per field).
- Consistency checks on extracted records (e.g. Ω = Ω_I + Ω_D + Ω_OD) to catch misattributed values.
- Rule-based parsers for more codes (SIESTA, Wannier90, phonopy, LAMMPS), moving them from the LLM-only to
  the hybrid path, and experimental file formats.
- Mapping to community ontologies for materials science and export to research-data platforms.
- Batch extraction from full-text papers, with links between publications and the underlying
  simulation data.

## Author

Gokaran Shukla, King Abdullah University of Science and Technology (KAUST)
[gokaran-shukla.github.io](https://gokaran-shukla.github.io)

## Citing llm-matmeta

If llm-matmeta helps your work, please cite the software:

> G. Shukla, *llm-matmeta: hybrid rule-based and LLM metadata extraction for computational materials science*, version 0.1.0 (2026). Zenodo. https://doi.org/10.5281/zenodo.24334736

```bibtex
@software{shukla_llmmatmeta_2026,
  author    = {Shukla, Gokaran},
  title     = {llm-matmeta: hybrid rule-based and LLM metadata extraction for computational materials science},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24334736},
  url       = {https://github.com/Gokaran-Shukla/llm-matmeta}
}
```

GitHub's **"Cite this repository"** button (from `CITATION.cff`) gives the same reference in APA and BibTeX.

## License

MIT
