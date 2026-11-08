# llm-matmeta

[![tests](https://github.com/Gokaran-Shukla/llm-matmeta/actions/workflows/tests.yml/badge.svg)](https://github.com/Gokaran-Shukla/llm-matmeta/actions/workflows/tests.yml)

**Hybrid rule-based + LLM metadata extraction for computational materials-science data.**

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
* **Ontology mapping:** records can be exported as JSON-LD with a user-supplied context that maps
  fields to the terms of a target ontology.

## Installation

```bash
git clone https://github.com/Gokaran-Shukla/llm-matmeta.git
cd llm-matmeta
pip install -e ".[dev]"
pytest -q          # 12 tests, run fully offline
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

# materials / methods / properties from paper abstracts
matmeta abstract examples/abstract.txt
```

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
| results    | convergence, total energy, Fermi energy, total magnetisation |
| abstracts  | materials (formula, dimensionality, role), methods (computational / experimental / theoretical, software), reported properties with values and units, keywords |

Units are normalised (Ry → eV for Quantum ESPRESSO). The full schemas are in
[`src/matmeta/schema.py`](src/matmeta/schema.py).

## Project layout

```
src/matmeta/
  parsers.py   rule-based VASP and Quantum ESPRESSO parsers + evidence condensation
  llm.py       OpenAI-compatible client and an offline mock client
  extract.py   pipeline: prompt, validation/retry, merge, provenance, conflicts
  schema.py    JSON schemas for calculations and abstracts
  mapping.py   JSON-LD export with configurable ontology context
  cli.py       `matmeta` command-line tool
tests/         offline unit tests (parsers, merge logic, retries, HTTP client, CLI)
examples/      small VASP and QE runs and an example abstract
```

## Roadmap

- Benchmark extraction accuracy of several open-weight and hosted LLMs on a curated set of real
  calculations (precision/recall per field).
- More codes (SIESTA, Wannier90, phonopy, LAMMPS) and experimental file formats.
- Mapping to community ontologies for materials science and export to research-data platforms.
- Batch extraction from full-text papers, with links between publications and the underlying
  simulation data.

## Author

Gokaran Shukla, King Abdullah University of Science and Technology (KAUST)
[gokaran-shukla.github.io](https://gokaran-shukla.github.io)

## License

MIT
