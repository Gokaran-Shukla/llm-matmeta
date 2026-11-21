"""JSON schemas for extracted metadata.

Record types
------------
* ``calculation`` - metadata of a single DFT calculation (VASP, Quantum ESPRESSO, or any other code
  read in LLM-only mode).
* ``abstract``    - materials, methods and properties mentioned in a paper abstract.
* ``wannier``     - a Wannier90 run, read from its ``.wout`` file.
* ``jobscript``   - an HPC batch script (SLURM, PBS, ...).

Every leaf value may be ``null`` so that the extractor can say "not found" instead of guessing.
``description`` strings are part of the prompt: they tell the model which convention to use, so that
disagreements with the rule-based parser are real disagreements and not differences of notation.
"""

from __future__ import annotations

import copy
from typing import Any

import jsonschema


def _t(kind: str, description: str | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": [kind, "null"], **extra}
    if description:
        out["description"] = description
    return out


def _num(d=None):
    return _t("number", d)


def _str(d=None, **kw):
    return _t("string", d, **kw)


def _bool(d=None):
    return _t("boolean", d)


def _int(d=None):
    return _t("integer", d)


def _obj(props: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    out = {"type": "object", "additionalProperties": False, "properties": props}
    if description:
        out["description"] = description
    return out


_VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}

CALCULATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DFTCalculationMetadata",
    "type": "object",
    "additionalProperties": False,
    "required": ["software", "method", "structure", "numerics", "results"],
    "properties": {
        "software": _obj({"name": _str("code name, e.g. VASP, Quantum ESPRESSO, SIESTA"), "version": _str()}),
        "method": _obj(
            {
                "task": _str(
                    "kind of run. VASP: IBRION=-1 (also the default when NSW is 0 or absent) -> scf; "
                    "ICHARG=11 -> nscf; IBRION=1,2,3 with NSW>0 -> relax (vc-relax if ISIF=3); "
                    "IBRION=0 -> md; IBRION=5-8 -> phonon",
                    enum=["scf", "relax", "vc-relax", "nscf", "bands", "md", "phonon", "other", None],
                ),
                "xc_functional": _str(
                    "standard name of the exchange-correlation functional (LDA, PBE, PBEsol, rPBE, PW91, "
                    "SCAN, r2SCAN, HSE06, PBE0 ...), never the raw input tag: VASP GGA=PE means PBE, "
                    "and a POTCAR line 'PAW_PBE' means PBE when no GGA tag is set"
                ),
                "spin_polarized": _bool(),
                "spin_orbit_coupling": _bool(),
                "noncollinear": _bool(),
                "hubbard_u": _bool("true if DFT+U is used (VASP LDAU, QE lda_plus_u / HUBBARD card)"),
                "van_der_waals_correction": _str("as written in the input, e.g. 'IVDW=12' or 'grimme-d3'"),
            }
        ),
        "structure": _obj(
            {
                "formula": _str(
                    "formula of the FULL simulation cell, not reduced, elements in input order, "
                    "e.g. Fe2 for a 2-atom Fe cell, Cu4In4P8S24"
                ),
                "n_atoms": _int(),
                "species": {"type": ["array", "null"], "items": {"type": "string"}},
                "lattice_vectors_angstrom": {"type": ["array", "null"], "items": _VEC3},
            }
        ),
        "numerics": _obj(
            {
                "plane_wave_cutoff_eV": _num(),
                "kpoint_mesh": {"type": ["array", "null"], "items": {"type": "integer"},
                                "description": "automatic (Gamma / Monkhorst-Pack) mesh only, else null"},
                "smearing_type": _str(
                    "one of Gaussian, Fermi-Dirac, Methfessel-Paxton, tetrahedron, tetrahedron (Bloechl), "
                    "cold, or the code's own name. VASP ISMEAR: 0 Gaussian, -1 Fermi-Dirac, "
                    ">=1 Methfessel-Paxton, -4 tetrahedron, -5 tetrahedron (Bloechl)"
                ),
                "smearing_width_eV": _num(),
                "energy_convergence_eV": _num(),
            }
        ),
        "results": _obj(
            {
                "converged": _bool(),
                "total_energy_eV": _num("final total (free) energy; VASP: last 'free energy TOTEN'"),
                "energy_sigma0_eV": _num("energy extrapolated to zero smearing; VASP: last 'energy(sigma->0)'"),
                "fermi_energy_eV": _num(),
                "total_magnetization_muB": _num(
                    "total magnetic moment of the cell; for noncollinear runs the length of the vector"
                ),
                "magnetization_vector_muB": {**_VEC3, "type": ["array", "null"],
                                             "description": "(mx, my, mz) for noncollinear runs, else null"},
            }
        ),
        "notes": _str(),
    },
}

ABSTRACT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AbstractMetadata",
    "type": "object",
    "additionalProperties": False,
    "required": ["materials", "methods", "properties", "keywords"],
    "properties": {
        "materials": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "formula": _str(),
                    "dimensionality": _str(enum=["0D", "1D", "2D", "3D", "interface", None]),
                    "role": _str(),
                },
            },
        },
        "methods": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "kind"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["computational", "experimental", "theoretical"]},
                    "software": _str(),
                },
            },
        },
        "properties": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "value": _str(), "unit": _str()},
            },
        },
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
}

WANNIER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Wannier90RunMetadata",
    "type": "object",
    "additionalProperties": False,
    "required": ["software", "system", "disentanglement", "wannierisation"],
    "properties": {
        "software": _obj({"name": _str("Wannier90"), "version": _str("release number, e.g. 3.1.0")}),
        "system": _obj(
            {
                "num_wann": _int("number of Wannier functions"),
                "num_bands": _int("number of input Bloch states / bands"),
                "kpoint_mesh": {"type": ["array", "null"], "items": {"type": "integer"},
                                "description": "MP grid size, e.g. [8, 8, 1]"},
                "spinors": _bool(),
            }
        ),
        "disentanglement": _obj(
            {
                "used": _bool("true if a disentanglement (subspace selection) step was run"),
                "outer_window_min_eV": _num("dis_win_min"),
                "outer_window_max_eV": _num("dis_win_max"),
                "frozen_window_min_eV": _num("dis_froz_min"),
                "frozen_window_max_eV": _num("dis_froz_max"),
                "converged": _bool(),
                "final_omega_i_A2": _num("Final Omega_I in Ang^2"),
            }
        ),
        "wannierisation": _obj(
            {
                "num_iter": _int("requested number of iterations"),
                "converged": _bool("true only if the output states that convergence criteria were satisfied"),
                "omega_i_A2": _num(),
                "omega_d_A2": _num(),
                "omega_od_A2": _num(),
                "omega_total_A2": _num("Final Spread, Omega Total, in Ang^2"),
            }
        ),
        "notes": _str(),
    },
}

JOBSCRIPT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BatchJobMetadata",
    "type": "object",
    "additionalProperties": False,
    "required": ["scheduler", "resources", "software"],
    "properties": {
        "scheduler": _str("slurm, pbs, lsf, sge or other"),
        "job_name": _str(),
        "resources": _obj(
            {
                "partition": _str("partition / queue"),
                "nodes": _int(),
                "ntasks": _int("total MPI tasks if written (--ntasks, or -n/-np of the launcher), else null"),
                "ntasks_per_node": _int(),
                "cpus_per_task": _int(),
                "gpus": _int("GPUs per node, from --gres=gpu:N or --gpus"),
                "memory": _str("as written, e.g. 360G"),
                "time_limit": _str("as written, e.g. 24:00:00"),
                "constraint": _str(),
            }
        ),
        "software": _obj(
            {
                "modules": {"type": ["array", "null"], "items": {"type": "string"},
                            "description": "every module loaded, exactly as written"},
                "executable": _str("main program run, e.g. vasp_std, vasp_ncl, pw.x, wannier90.x"),
                "launcher": _str("srun, mpirun, mpiexec, or null"),
            }
        ),
        "notes": _str(),
    },
}

SCHEMAS = {
    "calculation": CALCULATION_SCHEMA,
    "abstract": ABSTRACT_SCHEMA,
    "wannier": WANNIER_SCHEMA,
    "jobscript": JOBSCRIPT_SCHEMA,
}


def validate_against(obj: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty list = valid)."""
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(obj), key=lambda e: [str(p) for p in e.absolute_path])
    ]


def validate(record: dict[str, Any], kind: str) -> list[str]:
    return validate_against(record, SCHEMAS[kind])


def empty_record(schema: dict[str, Any]) -> dict[str, Any]:
    """A record with the shape of ``schema`` (nested objects) and every leaf set to null."""
    out: dict[str, Any] = {}
    for key, sub in schema.get("properties", {}).items():
        out[key] = empty_record(sub) if sub.get("type") == "object" else None
    return out


def empty_calculation() -> dict[str, Any]:
    """A schema-valid calculation record with every field set to null."""
    return copy.deepcopy(_EMPTY_CALC)


_EMPTY_CALC = empty_record(CALCULATION_SCHEMA)
