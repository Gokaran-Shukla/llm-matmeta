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
    # calculation
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
    "energy_sigma0_eV": "totalEnergyZeroSmearing",
    "fermi_energy_eV": "fermiEnergy", "total_magnetization_muB": "totalMagnetization",
    "magnetization_vector_muB": "magnetizationVector", "notes": "comment",
    # wannier90
    "system": "WannierSystem", "num_wann": "numberOfWannierFunctions", "num_bands": "numberOfBands",
    "spinors": "usesSpinors", "disentanglement": "Disentanglement", "used": "isUsed",
    "outer_window_min_eV": "outerWindowMin", "outer_window_max_eV": "outerWindowMax",
    "frozen_window_min_eV": "frozenWindowMin", "frozen_window_max_eV": "frozenWindowMax",
    "final_omega_i_A2": "finalOmegaI", "wannierisation": "Wannierisation", "num_iter": "numberOfIterations",
    "omega_i_A2": "omegaI", "omega_d_A2": "omegaD", "omega_od_A2": "omegaOD", "omega_total_A2": "totalSpread",
    # job script
    "scheduler": "scheduler", "job_name": "jobName", "resources": "ComputeResources", "partition": "partition",
    "nodes": "numberOfNodes", "ntasks": "numberOfTasks", "ntasks_per_node": "tasksPerNode",
    "cpus_per_task": "cpusPerTask", "gpus": "gpusPerNode", "memory": "memory", "time_limit": "timeLimit",
    "constraint": "nodeConstraint", "modules": "loadedModules", "executable": "executable",
    "launcher": "launcher",
    # links
    "attachments": "hasPart", "file": "fileName",
}

UNITS = {
    "plane_wave_cutoff_eV": "eV", "smearing_width_eV": "eV", "energy_convergence_eV": "eV",
    "total_energy_eV": "eV", "energy_sigma0_eV": "eV", "fermi_energy_eV": "eV",
    "total_magnetization_muB": "Bohr magneton", "magnetization_vector_muB": "Bohr magneton",
    "lattice_vectors_angstrom": "angstrom",
    "outer_window_min_eV": "eV", "outer_window_max_eV": "eV", "frozen_window_min_eV": "eV",
    "frozen_window_max_eV": "eV", "final_omega_i_A2": "square angstrom", "omega_i_A2": "square angstrom",
    "omega_d_A2": "square angstrom", "omega_od_A2": "square angstrom", "omega_total_A2": "square angstrom",
}

TYPES = {"wannier": "mm:Wannier90Run", "jobscript": "mm:BatchJob", "calculation": "mm:Calculation"}


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
    parts = [
        {"@type": TYPES.get(a["source"]["schema"], "mm:File"), "file": a.get("file"), **_annotate(a["metadata"])}
        for a in result.get("attachments", [])
    ]
    if parts:
        doc["attachments"] = parts
    return doc


def load_context(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
