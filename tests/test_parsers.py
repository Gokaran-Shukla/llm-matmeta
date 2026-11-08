from pathlib import Path

import pytest

from matmeta.parsers import RY_TO_EV, parse_incar, parse_qe, parse_vasp
from matmeta.schema import validate

EX = Path(__file__).resolve().parents[1] / "examples"


def test_incar_handles_comments_and_semicolons():
    tags = parse_incar("ENCUT = 500 ! cutoff\nISMEAR = 1 ; SIGMA = 0.1\n# comment\n")
    assert tags == {"ENCUT": "500", "ISMEAR": "1", "SIGMA": "0.1"}


def test_vasp_fe():
    pc = parse_vasp(EX / "vasp_Fe_bcc")
    r = pc.record
    assert r["software"] == {"name": "VASP", "version": "6.4.2"}
    assert r["structure"]["formula"] == "Fe2"
    assert r["structure"]["lattice_vectors_angstrom"][0] == [2.83, 0.0, 0.0]
    assert r["numerics"]["plane_wave_cutoff_eV"] == 500.0
    assert r["numerics"]["kpoint_mesh"] == [16, 16, 16]
    assert r["numerics"]["smearing_type"] == "Methfessel-Paxton"
    assert r["method"]["xc_functional"] == "PBE"
    assert r["method"]["spin_polarized"] is True
    assert r["method"]["task"] == "scf"
    # the LAST ionic/electronic values must be used
    assert r["results"]["total_energy_eV"] == pytest.approx(-16.46318822)
    assert r["results"]["fermi_energy_eV"] == pytest.approx(9.4817)
    assert r["results"]["total_magnetization_muB"] == pytest.approx(4.4702851)
    assert r["results"]["converged"] is True
    assert validate(r, "calculation") == []
    assert "POTCAR" in pc.evidence


def test_qe_si():
    pc = parse_qe(EX / "qe_Si" / "si.scf.in", EX / "qe_Si" / "si.scf.out")
    r = pc.record
    assert r["software"]["version"] == "7.2"
    assert r["structure"]["formula"] == "Si2"
    assert r["numerics"]["plane_wave_cutoff_eV"] == pytest.approx(40 * RY_TO_EV, rel=1e-6)
    assert r["numerics"]["kpoint_mesh"] == [8, 8, 8]
    assert r["numerics"]["energy_convergence_eV"] == pytest.approx(1e-10 * RY_TO_EV)
    # only the line starting with "!" is the final total energy
    assert r["results"]["total_energy_eV"] == pytest.approx(-22.83942715 * RY_TO_EV, rel=1e-9)
    assert r["results"]["fermi_energy_eV"] == pytest.approx(6.4812)
    assert r["results"]["converged"] is True
    assert validate(r, "calculation") == []
