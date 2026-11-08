"""JSON schemas for extracted metadata.

Two record types are supported:

* ``CALCULATION_SCHEMA`` - metadata of a single DFT calculation (VASP or Quantum ESPRESSO).
* ``ABSTRACT_SCHEMA``    - materials, methods and properties mentioned in a paper abstract.

Every leaf value may be ``null`` so that the extractor can say "not found" instead of guessing.
"""

from __future__ import annotations

from typing import Any

import jsonschema

_num = {"type": ["number", "null"]}
_str = {"type": ["string", "null"]}
_bool = {"type": ["boolean", "null"]}
_int = {"type": ["integer", "null"]}

CALCULATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DFTCalculationMetadata",
    "type": "object",
    "additionalProperties": False,
    "required": ["software", "method", "structure", "numerics", "results"],
    "properties": {
        "software": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"name": _str, "version": _str},
        },
        "method": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task": {
                    "type": ["string", "null"],
                    "enum": ["scf", "relax", "vc-relax", "nscf", "bands", "md", "phonon", "other", None],
                },
                "xc_functional": _str,
                "spin_polarized": _bool,
                "spin_orbit_coupling": _bool,
                "noncollinear": _bool,
                "hubbard_u": _bool,
                "van_der_waals_correction": _str,
            },
        },
        "structure": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "formula": _str,
                "n_atoms": _int,
                "species": {"type": ["array", "null"], "items": {"type": "string"}},
                "lattice_vectors_angstrom": {
                    "type": ["array", "null"],
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                },
            },
        },
        "numerics": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "plane_wave_cutoff_eV": _num,
                "kpoint_mesh": {"type": ["array", "null"], "items": {"type": "integer"}},
                "smearing_type": _str,
                "smearing_width_eV": _num,
                "energy_convergence_eV": _num,
            },
        },
        "results": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "converged": _bool,
                "total_energy_eV": _num,
                "fermi_energy_eV": _num,
                "total_magnetization_muB": _num,
            },
        },
        "notes": _str,
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
                    "formula": _str,
                    "dimensionality": {"type": ["string", "null"], "enum": ["0D", "1D", "2D", "3D", "interface", None]},
                    "role": _str,
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
                    "software": _str,
                },
            },
        },
        "properties": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "value": _str, "unit": _str},
            },
        },
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
}

SCHEMAS = {"calculation": CALCULATION_SCHEMA, "abstract": ABSTRACT_SCHEMA}


def validate(record: dict[str, Any], kind: str) -> list[str]:
    """Return a list of human-readable validation errors (empty list = valid)."""
    validator = jsonschema.Draft202012Validator(SCHEMAS[kind])
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    ]


def empty_calculation() -> dict[str, Any]:
    """A schema-valid calculation record with every field set to null."""
    return {
        "software": {"name": None, "version": None},
        "method": {
            "task": None,
            "xc_functional": None,
            "spin_polarized": None,
            "spin_orbit_coupling": None,
            "noncollinear": None,
            "hubbard_u": None,
            "van_der_waals_correction": None,
        },
        "structure": {"formula": None, "n_atoms": None, "species": None, "lattice_vectors_angstrom": None},
        "numerics": {
            "plane_wave_cutoff_eV": None,
            "kpoint_mesh": None,
            "smearing_type": None,
            "smearing_width_eV": None,
            "energy_convergence_eV": None,
        },
        "results": {
            "converged": None,
            "total_energy_eV": None,
            "fermi_energy_eV": None,
            "total_magnetization_muB": None,
        },
        "notes": None,
    }
