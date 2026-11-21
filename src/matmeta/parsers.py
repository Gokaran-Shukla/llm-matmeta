"""Deterministic (rule-based) parsers for VASP and Quantum ESPRESSO files.

These parsers serve two purposes:

1. They produce a *trusted* draft of the numeric metadata (energies, cutoffs, k-meshes ...),
   which is later used to cross-check the LLM output.
2. They condense large output files (an OUTCAR can be hundreds of MB) into a short
   "evidence" text that fits comfortably in an LLM context window.

Input files (INCAR, KPOINTS, the POSCAR header, the pw.x input) are passed to the model verbatim,
so the model's reading of the *settings* is independent of the parser. Output files are streamed
line by line, never loaded whole, and may be gzip-compressed (``OUTCAR.gz``).
"""

from __future__ import annotations

import gzip
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .schema import empty_calculation

RY_TO_EV = 13.605693122994  # CODATA 2018
HA_TO_EV = 2 * RY_TO_EV
BOHR_TO_ANG = 0.529177210903

_FLOAT = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?"


@dataclass
class ParsedCalculation:
    """Result of rule-based parsing: a partial record plus condensed evidence for the LLM."""

    code: str
    record: dict[str, Any] = field(default_factory=empty_calculation)
    evidence: str = ""
    files: list[str] = field(default_factory=list)
    structure_file: str | None = None


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _find(directory: Path, name: str) -> Path | None:
    """``name`` or ``name.gz`` inside ``directory``."""
    for cand in (directory / name, directory / f"{name}.gz"):
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def _read(path: Path, max_bytes: int | None = None) -> str:
    with _open(path) as fh:
        return fh.read(max_bytes) if max_bytes else fh.read()


def _lines(path: Path) -> Iterator[str]:
    with _open(path) as fh:
        yield from fh


def _formula(species_counts: dict[str, int]) -> str:
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in species_counts.items())


def _grep(text: str, pattern: str, max_hits: int = 3, last: bool = False) -> list[str]:
    hits = [ln.strip() for ln in text.splitlines() if re.search(pattern, ln)]
    return hits[-max_hits:] if last else hits[:max_hits]


def _floats(text: str) -> list[float]:
    return [float(x) for x in re.findall(_FLOAT, text)]


def _to_bool(value: str) -> bool | None:
    v = value.strip().strip(".").upper()
    if v in {"T", "TRUE"}:
        return True
    if v in {"F", "FALSE"}:
        return False
    return None


def _moment(values: list[float]) -> tuple[float | None, list[float] | None]:
    """Total moment and, for three components, the vector (mx, my, mz)."""
    if len(values) >= 3:
        vec = values[:3]
        return round(math.sqrt(sum(v * v for v in vec)), 7), vec
    if values:
        return values[0], None
    return None, None


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
        "header": "\n".join(lines[:8]),
    }


_VASP_GGA = {"PE": "PBE", "PS": "PBEsol", "91": "PW91", "RP": "rPBE", "AM": "AM05", "CA": "LDA"}
_VASP_IBRION_TASK = {"-1": "scf", "0": "md", "1": "relax", "2": "relax", "3": "relax",
                     "5": "phonon", "6": "phonon", "7": "phonon", "8": "phonon"}
_VASP_ISMEAR = {"-5": "tetrahedron (Bloechl)", "-4": "tetrahedron", "-1": "Fermi-Dirac",
                "0": "Gaussian", "1": "Methfessel-Paxton", "2": "Methfessel-Paxton"}

# Parameters echoed in the OUTCAR header with the values VASP actually used (defaults included).
_OUTCAR_PARAMS = ("IBRION", "NSW", "ISIF", "ISPIN", "ICHARG", "ISMEAR", "SIGMA", "ENCUT", "EDIFF",
                  "LSORBIT", "LNONCOLLINEAR", "LDAU", "IVDW", "GGA", "LEXCH", "METAGGA",
                  "LHFCALC", "HFSCREEN", "LWANNIER90")
_OUTCAR_PARAM_RE = re.compile(r"^\s*(" + "|".join(_OUTCAR_PARAMS) + r")\s*=\s*([^\s;]+)")


def _first(value: str) -> str:
    return value.split()[0] if value.split() else value


def scan_outcar(path: Path) -> dict[str, Any]:
    """Stream an OUTCAR once and keep only what is needed (constant memory)."""
    info: dict[str, Any] = {"params": {}, "potcar": [], "param_lines": [], "version_line": None,
                            "version": None, "converged": False, "conv_line": None}
    toten: deque[str] = deque(maxlen=2)
    sigma0: deque[str] = deque(maxlen=1)
    efermi: deque[str] = deque(maxlen=1)
    mag: deque[str] = deque(maxlen=2)
    for raw in _lines(path):
        line = raw.rstrip("\n")
        if info["version"] is None:
            ver = re.search(r"vasp\.(\d+\.\d+\.\d+)", line)
            if ver:
                info["version"], info["version_line"] = ver.group(1), line.strip()
                continue
        if "POTCAR:" in line and len(info["potcar"]) < 6 and line.strip() not in info["potcar"]:
            info["potcar"].append(line.strip())
            continue
        for m in _OUTCAR_PARAM_RE.finditer(line):
            key = m.group(1)
            if key not in info["params"]:
                info["params"][key] = m.group(2)
                info["param_lines"].append(line.strip())
        # second assignment on the same line, e.g. "ISMEAR =  0;  SIGMA = 0.05"
        if ";" in line:
            for m in re.finditer(r";\s*(" + "|".join(_OUTCAR_PARAMS) + r")\s*=\s*([^\s;]+)", line):
                info["params"].setdefault(m.group(1), m.group(2))
        if re.search(r"free\s+energy\s+TOTEN", line):
            toten.append(line.strip())
        elif "energy(sigma->0)" in line:
            sigma0.append(line.strip())
        elif "E-fermi" in line:
            efermi.append(line.strip())
        elif re.search(r"number of electron\s+\S+\s+magnetization", line):
            mag.append(line.strip())
        elif "reached required accuracy" in line or "aborting loop because EDIFF is reached" in line:
            info["converged"], info["conv_line"] = True, line.strip()
    info.update(toten=list(toten), sigma0=list(sigma0), efermi=list(efermi), mag=list(mag))
    return info


def _vasp_task(tags: dict[str, str]) -> str:
    nsw = int(float(_first(tags.get("NSW", "0"))))
    ibrion = _first(tags.get("IBRION", "-1" if nsw in (0, -1) else "0"))  # VASP's own default
    task = _VASP_IBRION_TASK.get(ibrion, "other")
    if task == "relax" and _first(tags.get("ISIF", "2")) in {"3", "6", "7"}:
        task = "vc-relax"
    if nsw == 0 and task in {"relax", "vc-relax", "md"}:
        task = "scf"
    if _first(tags.get("ICHARG", "")) == "11":
        task = "nscf"
    return task


def _vasp_xc(tags: dict[str, str]) -> str | None:
    meta = _first(tags.get("METAGGA", "")).upper()
    if meta and meta not in {"--", "NONE", "F"}:
        return {"R2SCAN": "r2SCAN"}.get(meta, meta)
    if _to_bool(tags.get("LHFCALC", "F")):
        hfs = tags.get("HFSCREEN")
        if hfs and abs(float(_first(hfs)) - 0.2) < 1e-6:
            return "HSE06"
        return "hybrid (LHFCALC)"
    for key in ("GGA", "LEXCH"):
        val = _first(tags.get(key, "")).upper()
        if val and val != "--":
            return _VASP_GGA.get(val, val)
    return None


def parse_vasp(directory: str | Path) -> ParsedCalculation:
    d = Path(directory)
    pc = ParsedCalculation(code="vasp")
    rec = pc.record
    rec["software"]["name"] = "VASP"
    evidence: list[str] = []

    incar: dict[str, str] = {}
    if p := _find(d, "INCAR"):
        incar_text = _read(p)
        incar = parse_incar(incar_text)
        pc.files.append(p.name)
        evidence.append("### INCAR\n" + incar_text.strip())

    if p := _find(d, "KPOINTS"):
        kp = _read(p)
        pc.files.append(p.name)
        lines = [ln.strip() for ln in kp.splitlines() if ln.strip()]
        if len(lines) >= 4 and lines[2][:1].upper() in {"G", "M"}:
            rec["numerics"]["kpoint_mesh"] = [int(x) for x in lines[3].split()[:3]]
        evidence.append("### KPOINTS\n" + "\n".join(lines[:6]) + ("\n..." if len(lines) > 6 else ""))

    out: dict[str, Any] = {"params": {}}
    if p := _find(d, "OUTCAR"):
        out = scan_outcar(p)
        pc.files.append(p.name)

    # Effective tags: explicit INCAR values first, then the values VASP reports using.
    tags = {**out["params"], **incar}

    if "ENCUT" in tags:
        rec["numerics"]["plane_wave_cutoff_eV"] = float(_first(tags["ENCUT"]))
    if "EDIFF" in tags:
        rec["numerics"]["energy_convergence_eV"] = float(_first(tags["EDIFF"]))
    if "SIGMA" in tags:
        rec["numerics"]["smearing_width_eV"] = float(_first(tags["SIGMA"]))
    if "ISMEAR" in tags:
        ism = _first(tags["ISMEAR"])
        rec["numerics"]["smearing_type"] = _VASP_ISMEAR.get(ism, "Methfessel-Paxton" if ism.isdigit() else None)
    if "ISPIN" in tags:
        rec["method"]["spin_polarized"] = _first(tags["ISPIN"]) == "2"
    for tag, key in (("LSORBIT", "spin_orbit_coupling"), ("LNONCOLLINEAR", "noncollinear"), ("LDAU", "hubbard_u")):
        if tag in tags:
            rec["method"][key] = _to_bool(tags[tag])
    if rec["method"]["spin_orbit_coupling"]:
        rec["method"]["noncollinear"] = True
    rec["method"]["xc_functional"] = _vasp_xc(tags)
    if incar or out["params"]:
        rec["method"]["task"] = _vasp_task(tags)
    if "IVDW" in tags and _first(tags["IVDW"]) != "0":
        rec["method"]["van_der_waals_correction"] = f"IVDW={_first(tags['IVDW'])}"

    # Structure: the final geometry (CONTCAR) for runs that move atoms, else POSCAR.
    moves = rec["method"]["task"] in {"relax", "vc-relax", "md"}
    for name in (("CONTCAR", "POSCAR") if moves else ("POSCAR", "CONTCAR")):
        if p := _find(d, name):
            try:
                pos = parse_poscar(_read(p))
            except (ValueError, IndexError):
                continue
            pc.files.append(p.name)
            pc.structure_file = p.name
            rec["structure"].update(
                formula=_formula(pos["counts"]),
                n_atoms=pos["n_atoms"],
                species=pos["species"],
                lattice_vectors_angstrom=pos["lattice"],
            )
            evidence.append(f"### {name} (first lines)\n{pos['header']}")
            break

    if out.get("version") is not None or out.get("toten"):
        rec["software"]["version"] = out["version"]
        if out["toten"]:
            rec["results"]["total_energy_eV"] = _floats(out["toten"][-1].split("=")[-1])[0]
        if out["sigma0"]:
            rec["results"]["energy_sigma0_eV"] = _floats(out["sigma0"][-1].split("energy(sigma->0)")[-1])[0]
        if out["efermi"]:
            rec["results"]["fermi_energy_eV"] = _floats(out["efermi"][-1].split(":", 1)[1])[0]
        if out["mag"]:
            total, vec = _moment(_floats(out["mag"][-1].split("magnetization", 1)[1]))
            rec["results"]["total_magnetization_muB"] = total
            rec["results"]["magnetization_vector_muB"] = vec
        if out["converged"]:
            rec["results"]["converged"] = True
        snippets = ([out["version_line"]] if out["version_line"] else []) + out["potcar"] + out["param_lines"]
        snippets += out["toten"] + out["sigma0"] + out["efermi"] + out["mag"]
        snippets += [out["conv_line"]] if out["conv_line"] else []
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


def scan_qe_output(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"version": None, "xc": None, "converged": None, "lines": []}
    etot: deque[str] = deque(maxlen=2)
    ef: deque[str] = deque(maxlen=1)
    mag: deque[str] = deque(maxlen=1)
    head: list[str] = []
    conv: str | None = None
    for raw in _lines(path):
        line = raw.strip()
        if info["version"] is None and (m := re.search(r"Program PWSCF v\.?\s*([\w.]+)", line)):
            info["version"] = m.group(1)
            head.append(line)
        elif info["xc"] is None and (m := re.search(r"Exchange-correlation\s*=\s*(.+)", line)):
            info["xc"] = m.group(1).strip()
            head.append(line)
        elif re.match(r"!\s+total energy", line):
            etot.append(line)
        elif re.search(r"the Fermi energy is", line, re.I):
            ef.append(line)
        elif re.search(r"total magnetization\s+=", line):
            mag.append(line)
        elif "convergence has been achieved" in line:
            info["converged"], conv = True, line
        elif "convergence NOT achieved" in line:
            info["converged"], conv = False, line
    info.update(etot=list(etot), ef=list(ef), mag=list(mag))
    info["lines"] = head + info["etot"] + info["ef"] + info["mag"] + ([conv] if conv else [])
    return info


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
        rec["method"]["xc_functional"] = p["input_dft"].upper()
    if "vdw_corr" in p:
        rec["method"]["van_der_waals_correction"] = p["vdw_corr"]
    if "atoms" in inp:
        counts = Counter(re.sub(r"\d+$", "", a) for a in inp["atoms"])
        rec["structure"].update(formula=_formula(dict(counts)), n_atoms=sum(counts.values()), species=list(counts))

    if output_file and Path(output_file).exists():
        out = scan_qe_output(Path(output_file))
        pc.files.append(Path(output_file).name)
        rec["software"]["version"] = out["version"]
        if out["xc"] and rec["method"]["xc_functional"] is None:
            name = out["xc"].split("(")[0].strip()
            if re.fullmatch(r"[A-Za-z0-9+\-]+", name):  # a single named functional, e.g. PBE
                rec["method"]["xc_functional"] = name.upper()
        if out["etot"]:
            rec["results"]["total_energy_eV"] = round(_floats(out["etot"][-1].split("=")[1])[0] * RY_TO_EV, 6)
        if out["ef"]:
            rec["results"]["fermi_energy_eV"] = _floats(out["ef"][-1].split("is", 1)[1])[0]
        if out["mag"]:
            total, vec = _moment(_floats(out["mag"][-1].split("=", 1)[1].split("Bohr")[0]))
            rec["results"]["total_magnetization_muB"] = total
            rec["results"]["magnetization_vector_muB"] = vec
        rec["results"]["converged"] = out["converged"]
        evidence.append("### pw.x output (selected lines)\n" + "\n".join(out["lines"]))

    pc.evidence = "\n\n".join(evidence)
    return pc


def detect_and_parse(path: str | Path) -> ParsedCalculation:
    """Parse a VASP run directory, or a QE input file (``foo.in``, looks for ``foo.out``)."""
    p = Path(path)
    if p.is_dir():
        if _find(p, "INCAR") or _find(p, "OUTCAR"):
            return parse_vasp(p)
        ins = sorted(p.glob("*.in"))
        if len(ins) > 1:
            raise ValueError(f"{p} has several QE inputs ({', '.join(i.name for i in ins)}); pass one file")
        if ins:
            out = ins[0].with_suffix(".out")
            return parse_qe(ins[0], out if out.exists() else None)
        raise ValueError(f"No VASP or Quantum ESPRESSO files found in {p}")
    out = p.with_suffix(".out")
    return parse_qe(p, out if out.exists() else None)
