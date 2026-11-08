"""Deterministic (rule-based) parsers for VASP and Quantum ESPRESSO files.

These parsers serve two purposes:

1. They produce a *trusted* draft of the numeric metadata (energies, cutoffs, k-meshes ...),
   which is later used to cross-check the LLM output.
2. They condense large output files (an OUTCAR can be hundreds of MB) into a short
   "evidence" text that fits comfortably in an LLM context window.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import empty_calculation

RY_TO_EV = 13.605693122994  # CODATA 2018


@dataclass
class ParsedCalculation:
    """Result of rule-based parsing: a partial record plus condensed evidence for the LLM."""

    code: str
    record: dict[str, Any] = field(default_factory=empty_calculation)
    evidence: str = ""
    files: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _read(path: Path, max_bytes: int | None = None) -> str:
    with open(path, "r", errors="replace") as fh:
        return fh.read(max_bytes) if max_bytes else fh.read()


def _formula(species_counts: dict[str, int]) -> str:
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in species_counts.items())


def _grep(text: str, pattern: str, max_hits: int = 3, last: bool = False) -> list[str]:
    hits = [ln.strip() for ln in text.splitlines() if re.search(pattern, ln)]
    return hits[-max_hits:] if last else hits[:max_hits]


def _to_bool(value: str) -> bool | None:
    v = value.strip().strip(".").upper()
    if v in {"T", "TRUE"}:
        return True
    if v in {"F", "FALSE"}:
        return False
    return None


# --------------------------------------------------------------------------------------
# VASP
# --------------------------------------------------------------------------------------
def parse_incar(text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for raw in text.splitlines():
        line = re.split(r"[#!]", raw, maxsplit=1)[0]
        for stmt in line.split(";"):
            if "=" in stmt:
                key, val = stmt.split("=", 1)
                tags[key.strip().upper()] = val.strip()
    return tags


def parse_poscar(text: str) -> dict[str, Any]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    scale = float(lines[1].split()[0])
    lattice = [[round(float(x) * scale, 6) for x in lines[i].split()[:3]] for i in range(2, 5)]
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    return {
        "lattice": lattice,
        "species": species,
        "counts": dict(zip(species, counts)),
        "n_atoms": sum(counts),
    }


_VASP_GGA = {"PE": "PBE", "PS": "PBEsol", "91": "PW91", "RP": "rPBE", "AM": "AM05", "CA": "LDA"}
_VASP_IBRION_TASK = {"-1": "scf", "0": "md", "1": "relax", "2": "relax", "3": "relax",
                     "5": "phonon", "6": "phonon", "7": "phonon", "8": "phonon"}
_VASP_ISMEAR = {"-5": "tetrahedron (Bloechl)", "-4": "tetrahedron", "-1": "Fermi-Dirac",
                "0": "Gaussian", "1": "Methfessel-Paxton", "2": "Methfessel-Paxton"}


def parse_vasp(directory: str | Path) -> ParsedCalculation:
    d = Path(directory)
    pc = ParsedCalculation(code="vasp")
    rec = pc.record
    rec["software"]["name"] = "VASP"
    evidence: list[str] = []

    incar: dict[str, str] = {}
    if (d / "INCAR").exists():
        incar_text = _read(d / "INCAR")
        incar = parse_incar(incar_text)
        pc.files.append("INCAR")
        evidence.append("### INCAR\n" + incar_text.strip())

    if (d / "POSCAR").exists():
        pos = parse_poscar(_read(d / "POSCAR"))
        pc.files.append("POSCAR")
        rec["structure"].update(
            formula=_formula(pos["counts"]),
            n_atoms=pos["n_atoms"],
            species=pos["species"],
            lattice_vectors_angstrom=pos["lattice"],
        )
        evidence.append(f"### POSCAR (parsed)\nspecies={pos['counts']} lattice={pos['lattice']}")

    if (d / "KPOINTS").exists():
        kp = _read(d / "KPOINTS")
        pc.files.append("KPOINTS")
        lines = [ln.strip() for ln in kp.splitlines() if ln.strip()]
        if len(lines) >= 4 and lines[2][:1].upper() in {"G", "M"}:
            rec["numerics"]["kpoint_mesh"] = [int(x) for x in lines[3].split()[:3]]
        evidence.append("### KPOINTS\n" + kp.strip())

    # INCAR-derived method/numerics
    if "ENCUT" in incar:
        rec["numerics"]["plane_wave_cutoff_eV"] = float(incar["ENCUT"].split()[0])
    if "EDIFF" in incar:
        rec["numerics"]["energy_convergence_eV"] = float(incar["EDIFF"].split()[0])
    if "SIGMA" in incar:
        rec["numerics"]["smearing_width_eV"] = float(incar["SIGMA"].split()[0])
    if "ISMEAR" in incar:
        rec["numerics"]["smearing_type"] = _VASP_ISMEAR.get(incar["ISMEAR"].split()[0])
    rec["method"]["spin_polarized"] = incar.get("ISPIN", "1").split()[0] == "2" or None
    for tag, key in (("LSORBIT", "spin_orbit_coupling"), ("LNONCOLLINEAR", "noncollinear"), ("LDAU", "hubbard_u")):
        if tag in incar:
            rec["method"][key] = _to_bool(incar[tag])
    if rec["method"]["spin_orbit_coupling"]:
        rec["method"]["noncollinear"] = True
    if "GGA" in incar:
        rec["method"]["xc_functional"] = _VASP_GGA.get(incar["GGA"].split()[0].upper())
    if "IBRION" in incar:
        task = _VASP_IBRION_TASK.get(incar["IBRION"].split()[0], "other")
        if task == "relax" and incar.get("ISIF", "2").split()[0] in {"3", "6", "7"}:
            task = "vc-relax"
        if incar.get("NSW", "0").split()[0] == "0" and task in {"relax", "vc-relax", "md"}:
            task = "scf"
        rec["method"]["task"] = task
    if "ICHARG" in incar and incar["ICHARG"].split()[0] == "11":
        rec["method"]["task"] = "nscf"
    if "IVDW" in incar and incar["IVDW"].split()[0] != "0":
        rec["method"]["van_der_waals_correction"] = f"IVDW={incar['IVDW'].split()[0]}"

    # OUTCAR: only grep the lines we need (OUTCARs are huge)
    if (d / "OUTCAR").exists():
        out = _read(d / "OUTCAR")
        pc.files.append("OUTCAR")
        ver = re.search(r"vasp\.(\d+\.\d+\.\d+)", out)
        if ver:
            rec["software"]["version"] = ver.group(1)
        toten = re.findall(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.\d+)", out)
        if toten:
            rec["results"]["total_energy_eV"] = float(toten[-1])
        ef = re.findall(r"E-fermi\s*:\s*(-?\d+\.\d+)", out)
        if ef:
            rec["results"]["fermi_energy_eV"] = float(ef[-1])
        mag = re.findall(r"number of electron\s+\S+\s+magnetization\s+(-?\d+\.\d+)", out)
        if mag:
            rec["results"]["total_magnetization_muB"] = float(mag[-1])
        if "reached required accuracy" in out or "aborting loop because EDIFF is reached" in out:
            rec["results"]["converged"] = True
        snippets = (
            _grep(out, r"vasp\.\d", 1)
            + _grep(out, r"POTCAR:", 6)
            + _grep(out, r"GGA\s*=|LEXCH|METAGGA", 3)
            + _grep(out, r"free\s+energy\s+TOTEN", 2, last=True)
            + _grep(out, r"E-fermi", 1, last=True)
            + _grep(out, r"magnetization", 2, last=True)
            + _grep(out, r"reached required accuracy|EDIFF is reached", 1, last=True)
        )
        evidence.append("### OUTCAR (selected lines)\n" + "\n".join(snippets))

    pc.evidence = "\n\n".join(evidence)
    return pc


# --------------------------------------------------------------------------------------
# Quantum ESPRESSO (pw.x)
# --------------------------------------------------------------------------------------
def parse_qe_input(text: str) -> dict[str, Any]:
    params: dict[str, str] = {}
    for m in re.finditer(r"^\s*([A-Za-z_]+(?:\(\d+\))?)\s*=\s*([^,\n]+)", text, re.M):
        params[m.group(1).lower()] = m.group(2).strip().strip("'\"")
    cards: dict[str, Any] = {"params": params}
    km = re.search(r"K_POINTS\s*\{?\s*automatic\s*\}?\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", text, re.I)
    if km:
        cards["kmesh"] = [int(km.group(i)) for i in (1, 2, 3)]
    ap = re.search(r"ATOMIC_POSITIONS[^\n]*\n((?:\s*[A-Z][a-z]?\d*\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+.*\n?)+)", text)
    if ap:
        atoms = [ln.split()[0] for ln in ap.group(1).splitlines() if ln.strip()]
        cards["atoms"] = atoms
    return cards


def parse_qe(input_file: str | Path, output_file: str | Path | None = None) -> ParsedCalculation:
    pc = ParsedCalculation(code="quantum-espresso")
    rec = pc.record
    rec["software"]["name"] = "Quantum ESPRESSO"
    evidence: list[str] = []

    inp_text = _read(Path(input_file))
    pc.files.append(Path(input_file).name)
    inp = parse_qe_input(inp_text)
    p = inp["params"]
    evidence.append("### pw.x input\n" + inp_text.strip())

    calc = p.get("calculation", "scf").lower()
    rec["method"]["task"] = calc if calc in {"scf", "relax", "vc-relax", "nscf", "bands", "md"} else "other"
    if "ecutwfc" in p:
        rec["numerics"]["plane_wave_cutoff_eV"] = round(float(p["ecutwfc"].replace("d", "e")) * RY_TO_EV, 3)
    if "conv_thr" in p:
        rec["numerics"]["energy_convergence_eV"] = float(p["conv_thr"].lower().replace("d", "e")) * RY_TO_EV
    if "degauss" in p:
        rec["numerics"]["smearing_width_eV"] = round(float(p["degauss"].replace("d", "e")) * RY_TO_EV, 4)
    if "smearing" in p:
        rec["numerics"]["smearing_type"] = p["smearing"]
    if "kmesh" in inp:
        rec["numerics"]["kpoint_mesh"] = inp["kmesh"]
    if "nspin" in p:
        rec["method"]["spin_polarized"] = p["nspin"] in {"2", "4"}
    for key, field_ in (("lspinorb", "spin_orbit_coupling"), ("noncolin", "noncollinear")):
        if key in p:
            rec["method"][field_] = _to_bool(p[key])
    if "lda_plus_u" in p:
        rec["method"]["hubbard_u"] = _to_bool(p["lda_plus_u"])
    if "input_dft" in p:
        rec["method"]["xc_functional"] = p["input_dft"]
    if "vdw_corr" in p:
        rec["method"]["van_der_waals_correction"] = p["vdw_corr"]
    if "atoms" in inp:
        counts = Counter(re.sub(r"\d+$", "", a) for a in inp["atoms"])
        rec["structure"].update(formula=_formula(dict(counts)), n_atoms=sum(counts.values()), species=list(counts))

    if output_file and Path(output_file).exists():
        out = _read(Path(output_file))
        pc.files.append(Path(output_file).name)
        ver = re.search(r"Program PWSCF v\.?\s*([\w.]+)", out)
        if ver:
            rec["software"]["version"] = ver.group(1)
        etot = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", out, re.M)
        if etot:
            rec["results"]["total_energy_eV"] = round(float(etot[-1]) * RY_TO_EV, 6)
        ef = re.findall(r"the Fermi energy is\s+(-?\d+\.\d+)\s+ev", out, re.I)
        if ef:
            rec["results"]["fermi_energy_eV"] = float(ef[-1])
        mag = re.findall(r"total magnetization\s+=\s+(-?\d+\.\d+)", out)
        if mag:
            rec["results"]["total_magnetization_muB"] = float(mag[-1])
        if "convergence has been achieved" in out:
            rec["results"]["converged"] = True
        elif "convergence NOT achieved" in out:
            rec["results"]["converged"] = False
        snippets = (
            _grep(out, r"Program PWSCF", 1)
            + _grep(out, r"Exchange-correlation", 2)
            + _grep(out, r"^!\s+total energy", 2, last=True)
            + _grep(out, r"Fermi energy", 1, last=True)
            + _grep(out, r"total magnetization", 1, last=True)
            + _grep(out, r"convergence (has been|NOT) achieved", 1, last=True)
        )
        evidence.append("### pw.x output (selected lines)\n" + "\n".join(snippets))

    pc.evidence = "\n\n".join(evidence)
    return pc


def detect_and_parse(path: str | Path) -> ParsedCalculation:
    """Parse a VASP run directory, or a QE input file (``foo.in``, looks for ``foo.out``)."""
    p = Path(path)
    if p.is_dir():
        if (p / "INCAR").exists() or (p / "OUTCAR").exists():
            return parse_vasp(p)
        ins = sorted(p.glob("*.in"))
        if ins:
            out = ins[0].with_suffix(".out")
            return parse_qe(ins[0], out if out.exists() else None)
        raise ValueError(f"No VASP or Quantum ESPRESSO files found in {p}")
    out = p.with_suffix(".out")
    return parse_qe(p, out if out.exists() else None)
