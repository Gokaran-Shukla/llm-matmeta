"""Map extracted records onto a vocabulary and serialise them as JSON-LD.

The default context uses a placeholder namespace. To publish into a specific knowledge graph,
pass your own ``context`` that maps each field name to a term of the target ontology
(for example a materials-science application ontology).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_NAMESPACE = "https://example.org/matmeta#"

FIELD_TERMS = {
    "software": "Software", "name": "name", "version": "softwareVersion",
    "method": "SimulationMethod", "task": "calculationTask", "xc_functional": "exchangeCorrelationFunctional",
    "spin_polarized": "isSpinPolarized", "spin_orbit_coupling": "includesSpinOrbitCoupling",
    "noncollinear": "isNoncollinear", "hubbard_u": "usesHubbardU",
    "van_der_waals_correction": "vanDerWaalsCorrection",
    "structure": "AtomicStructure", "formula": "chemicalFormula", "n_atoms": "numberOfAtoms",
    "species": "chemicalSpecies", "lattice_vectors_angstrom": "latticeVectors",
    "numerics": "NumericalParameters", "plane_wave_cutoff_eV": "planeWaveCutoff",
    "kpoint_mesh": "kPointMesh", "smearing_type": "smearingMethod", "smearing_width_eV": "smearingWidth",
    "energy_convergence_eV": "energyConvergenceThreshold",
    "results": "SimulationResults", "converged": "isConverged", "total_energy_eV": "totalEnergy",
    "fermi_energy_eV": "fermiEnergy", "total_magnetization_muB": "totalMagnetization", "notes": "comment",
}

UNITS = {
    "plane_wave_cutoff_eV": "eV", "smearing_width_eV": "eV", "energy_convergence_eV": "eV",
    "total_energy_eV": "eV", "fermi_energy_eV": "eV", "total_magnetization_muB": "Bohr magneton",
    "lattice_vectors_angstrom": "angstrom",
}


def default_context(namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
    ctx: dict[str, Any] = {"mm": namespace, "unit": "mm:unit", "value": "mm:value"}
    ctx.update({k: f"mm:{v}" for k, v in FIELD_TERMS.items()})
    return ctx


def _annotate(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in node.items():
        if val is None:
            continue
        if isinstance(val, dict):
            out[key] = _annotate(val)
        elif key in UNITS:
            out[key] = {"value": val, "unit": UNITS[key]}
        else:
            out[key] = val
    return out


def to_jsonld(result: dict[str, Any], context: dict[str, Any] | None = None, record_id: str | None = None):
    """Turn the output of :func:`extract_calculation` into a JSON-LD document (nulls dropped)."""
    doc: dict[str, Any] = {"@context": context or default_context(), "@type": "mm:DFTCalculation"}
    if record_id:
        doc["@id"] = record_id
    doc.update(_annotate(result["metadata"]))
    return doc


def load_context(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
