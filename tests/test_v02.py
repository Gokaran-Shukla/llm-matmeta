"""Tests for v0.2: parser fixes, LLM fallback, LLM-only files with quote verification, extras."""

import gzip
import json
import shutil
from pathlib import Path

import pytest

from matmeta import extract_calculation, extract_file, extract_text, to_jsonld
from matmeta.cli import main
from matmeta.extract import _equal
from matmeta.files import FILE_SYSTEM_PROMPT, check_quote, condense, detect_kind, find_extras
from matmeta.llm import MockClient
from matmeta.parsers import parse_qe, parse_vasp
from matmeta.schema import validate

EX = Path(__file__).resolve().parents[1] / "examples"
SOC = EX / "vasp_Fe_soc"


# ---------------------------------------------------------------- parser fixes
def test_noncollinear_moment_sigma0_and_defaults_from_outcar():
    r = parse_vasp(SOC).record
    assert r["method"]["task"] == "scf"  # no IBRION/NSW in INCAR -> OUTCAR says IBRION=-1
    assert r["method"]["xc_functional"] == "PBE"  # no GGA tag -> LEXCH = PE
    assert r["method"]["spin_orbit_coupling"] is True and r["method"]["noncollinear"] is True
    assert r["method"]["hubbard_u"] is False
    assert r["numerics"]["smearing_type"] == "Gaussian" and r["numerics"]["smearing_width_eV"] == 0.05
    assert r["results"]["magnetization_vector_muB"] == pytest.approx([1.1e-6, -3.1e-6, 2.2310872])
    assert r["results"]["total_magnetization_muB"] == pytest.approx(2.2310872, rel=1e-6)
    assert r["results"]["energy_sigma0_eV"] == pytest.approx(-8.29514161)
    assert r["results"]["total_energy_eV"] == pytest.approx(-8.29531207)
    assert validate(r, "calculation") == []


@pytest.mark.parametrize("incar,task", [
    ("ENCUT = 400\n", "scf"),                      # VASP default: NSW=0 -> IBRION=-1
    ("NSW = 100\n", "md"),                         # NSW>0, no IBRION -> IBRION=0
    ("IBRION = 2\nNSW = 50\nISIF = 3\n", "vc-relax"),
    ("ICHARG = 11\n", "nscf"),
])
def test_vasp_task_defaults_without_outcar(tmp_path, incar, task):
    (tmp_path / "INCAR").write_text(incar)
    assert parse_vasp(tmp_path).record["method"]["task"] == task


def test_relaxation_uses_contcar(tmp_path):
    for f in ("POSCAR", "KPOINTS", "OUTCAR"):
        shutil.copy(EX / "vasp_Fe_bcc" / f, tmp_path / f)
    (tmp_path / "INCAR").write_text("IBRION = 2\nNSW = 40\nISIF = 3\nGGA = PE\n")
    (tmp_path / "CONTCAR").write_text((EX / "vasp_Fe_bcc" / "POSCAR").read_text().replace("2.83", "2.84"))
    pc = parse_vasp(tmp_path)
    assert pc.record["method"]["task"] == "vc-relax"
    assert pc.structure_file == "CONTCAR"
    assert pc.record["structure"]["lattice_vectors_angstrom"][0] == [2.84, 0.0, 0.0]


def test_gzipped_outcar(tmp_path):
    for f in ("INCAR", "POSCAR", "KPOINTS"):
        shutil.copy(EX / "vasp_Fe_bcc" / f, tmp_path / f)
    with gzip.open(tmp_path / "OUTCAR.gz", "wt") as fh:
        fh.write((EX / "vasp_Fe_bcc" / "OUTCAR").read_text())
    r = parse_vasp(tmp_path).record
    assert r["results"]["total_energy_eV"] == pytest.approx(-16.46318822)
    assert r["software"]["version"] == "6.4.2"


def test_qe_xc_from_output():
    pc = parse_qe(EX / "qe_Si" / "si.scf.in", EX / "qe_Si" / "si.scf.out")
    assert pc.record["method"]["xc_functional"] == "PBE"


def test_equal_compares_lists_with_tolerance():
    assert _equal([[2.83, 0.0, 0.0]], [[2.8300001, 0.0, 0.0]])
    assert not _equal([16, 16, 16], [16, 16, 8])
    assert not _equal(True, 1)


# ---------------------------------------------------------------- LLM fallback
class Boom:
    def complete_json(self, system, user):
        raise ConnectionError("server unreachable")


def test_llm_failure_falls_back_to_parser_record():
    res = extract_calculation(EX / "vasp_Fe_bcc", Boom())
    assert res["llm_error"].startswith("ConnectionError")
    assert res["source"]["llm_used"] is False
    assert res["metadata"]["results"]["total_energy_eV"] == pytest.approx(-16.46318822)
    assert res["validation_errors"] == []


def test_cli_keeps_going_when_llm_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("matmeta.cli.OpenAICompatibleClient", lambda **kw: Boom())
    assert main(["calc", str(EX / "vasp_Fe_bcc"), "-o", str(tmp_path)]) == 0
    assert "parser-only" in capsys.readouterr().err
    assert json.loads((tmp_path / "vasp_Fe_bcc.json").read_text())["metadata"]["structure"]["formula"] == "Fe2"


# ---------------------------------------------------------------- LLM-only files
WOUT = (SOC / "wannier90.wout").read_text()


def _wannier_reply(hallucinate=False):
    rec = {
        "software": {"name": "Wannier90", "version": "3.1.0"},
        "system": {"num_wann": 18, "num_bands": 48, "kpoint_mesh": [12, 12, 12], "spinors": True},
        "disentanglement": {"used": True, "outer_window_min_eV": -10.0, "outer_window_max_eV": 30.0,
                            "frozen_window_min_eV": -10.0, "frozen_window_max_eV": 12.0,
                            "converged": True, "final_omega_i_A2": 31.25004812},
        "wannierisation": {"num_iter": 200, "converged": True, "omega_i_A2": 31.25004812,
                           "omega_d_A2": 0.004512338, "omega_od_A2": 1.847572962, "omega_total_A2": 33.10213342},
        "notes": None,
    }
    quotes = {
        "software.name": "|                   WANNIER90                       |",
        "software.version": "|            Release: 3.1.0        5th March 2020   |",
        "system.num_wann": "|  Number of Wannier Functions               :                18             |",
        "system.num_bands": "|  Number of input Bloch states              :                48             |",
        "system.kpoint_mesh": "|  Grid size =  12 x  12 x  12      Total points = 1728                      |",
        "system.spinors": "|  Use spinors                               :                 T             |",
        "disentanglement.used": "|  Using band disentanglement                :                 T             |",
        "disentanglement.outer_window_min_eV": "Outer: -10.00000000  to   30.00000000  (eV)",
        "disentanglement.outer_window_max_eV": "Outer: -10.00000000  to   30.00000000  (eV)",
        "disentanglement.frozen_window_min_eV": "Inner: -10.00000000  to   12.00000000  (eV)",
        "disentanglement.frozen_window_max_eV": "Inner: -10.00000000  to   12.00000000  (eV)",
        "disentanglement.converged": "<<< Disentanglement convergence criteria satisfied >>>",
        "disentanglement.final_omega_i_A2": "Final Omega_I    31.25004812 (Ang^2)",
        "wannierisation.num_iter": "|  Total number of iterations                :               200             |",
        "wannierisation.converged": "<<< Wannierisation convergence criteria satisfied >>>",
        "wannierisation.omega_i_A2": "Spreads (Ang^2)       Omega I      =    31.250048120",
        "wannierisation.omega_d_A2": "================       Omega D      =     0.004512338",
        "wannierisation.omega_od_A2": "Omega OD     =     1.847572962",
        "wannierisation.omega_total_A2": "Final Spread (Ang^2)       Omega Total  =    33.102133420",
    }
    if hallucinate:
        rec["wannierisation"]["omega_total_A2"] = 29.5  # wrong number, real line
        rec["system"]["num_bands"] = 64  # wrong number, invented line
        quotes["system.num_bands"] = "Number of bands : 64"
        quotes.pop("wannierisation.omega_od_A2")  # no quote at all
    return {"record": rec, "quotes": quotes}


def test_wannier_values_verified_against_quotes():
    res = extract_text(WOUT, MockClient(_wannier_reply()), "wannier")
    assert res["unverified"] == []
    assert res["provenance"]["wannierisation.omega_total_A2"] == "llm (verified)"
    assert res["provenance"]["system.kpoint_mesh"] == "llm (verified)"
    assert res["provenance"]["software.name"] == "llm (verified)"
    assert res["provenance"]["wannierisation.converged"] == "llm (quote found)"
    assert res["provenance"]["notes"] == "missing"
    assert validate(res["metadata"], "wannier") == []


def test_hallucinations_are_flagged_and_dropped_in_strict_mode():
    res = extract_text(WOUT, MockClient(_wannier_reply(hallucinate=True)), "wannier")
    prov = res["provenance"]
    assert prov["wannierisation.omega_total_A2"] == "llm (unverified: value not in quote)"
    assert prov["system.num_bands"] == "llm (unverified: quote not in file)"
    assert prov["wannierisation.omega_od_A2"] == "llm (unverified: no quote)"
    assert len(res["unverified"]) == 3
    strict = extract_text(WOUT, MockClient(_wannier_reply(hallucinate=True)), "wannier", strict=True)
    assert strict["metadata"]["system"]["num_bands"] is None
    assert strict["metadata"]["system"]["num_wann"] == 18  # verified values kept
    assert strict["provenance"]["system.num_bands"].startswith("dropped")


def test_quote_check_unit_conversion_and_whitespace():
    src = " ".join("Total energy  =   -22.83942715 Ry".split())
    assert check_quote(-22.83942715 * 13.605693122994, "Total energy =  -22.83942715   Ry", src) == "llm (verified)"
    assert check_quote("slurm", "#SBATCH --nodes=2", "#SBATCH --nodes=2") == "llm (verified)"


def test_condense_keeps_head_tail_and_keyword_lines():
    text = "\n".join([f"noise line {i}" for i in range(5000)] + ["Final Spread (Ang^2) Omega Total = 1.0"]
                     + [f"more noise {i}" for i in range(5000)])
    out, condensed = condense(text, ["final spread"], max_chars=2000, head=5, tail=5)
    assert condensed and "Omega Total = 1.0" in out and "lines skipped" in out
    assert len(out) < 2000


def test_invalid_file_reply_gives_empty_record():
    res = extract_text(WOUT, MockClient({"record": {"system": "oops"}}), "wannier")
    assert res["llm_validation_errors"]
    assert res["metadata"]["system"]["num_wann"] is None
    assert validate(res["metadata"], "wannier") == []


# ---------------------------------------------------------------- job scripts, extras, CLI
JOB = (SOC / "z_jobs_ibex").read_text()


def _job_reply():
    return {
        "record": {
            "scheduler": "slurm", "job_name": "fe_soc_wan",
            "resources": {"partition": "batch", "nodes": 2, "ntasks": 80, "ntasks_per_node": 40,
                          "cpus_per_task": None, "gpus": None, "memory": "360G", "time_limit": "24:00:00",
                          "constraint": "cascadelake"},
            "software": {"modules": ["intel/2020", "vasp/6.4.2", "wannier90/3.1.0"], "executable": "vasp_ncl",
                         "launcher": "srun"},
            "notes": None,
        },
        "quotes": {
            "scheduler": "#SBATCH --job-name=fe_soc_wan", "job_name": "#SBATCH --job-name=fe_soc_wan",
            "resources.partition": "#SBATCH --partition=batch", "resources.nodes": "#SBATCH --nodes=2",
            "resources.ntasks": "srun -n 80 vasp_ncl", "resources.ntasks_per_node": "#SBATCH --ntasks-per-node=40",
            "resources.memory": "#SBATCH --mem=360G", "resources.time_limit": "#SBATCH --time=24:00:00",
            "resources.constraint": "#SBATCH --constraint=cascadelake",
            "software.modules": "module load intel/2020 vasp/6.4.2 wannier90/3.1.0",
            "software.executable": "srun -n 80 vasp_ncl", "software.launcher": "srun -n 80 vasp_ncl",
        },
    }


def test_detection_and_jobscript():
    assert detect_kind(SOC / "wannier90.wout") == "wannier"
    assert detect_kind(SOC / "z_jobs_ibex") == "jobscript"
    assert detect_kind(SOC / "INCAR") == "calculation"
    res = extract_file(SOC / "z_jobs_ibex", MockClient(_job_reply()))
    assert res["source"]["schema"] == "jobscript"
    assert res["unverified"] == []


def _dispatching_client():
    def reply(system, user):
        if system != FILE_SYSTEM_PROMPT:  # the calculation prompt: agree with the parser
            return parse_vasp(SOC).record
        return _wannier_reply() if "Wannier90RunMetadata" in user else _job_reply()
    return MockClient(reply)


def test_calc_with_extras_attaches_wout_and_jobscript():
    assert [k for _, k in find_extras(SOC)] == ["wannier", "jobscript"]
    res = extract_calculation(SOC, _dispatching_client(), extras=True)
    assert res["conflicts"] == []
    assert [a["file"] for a in res["attachments"]] == ["wannier90.wout", "z_jobs_ibex"]
    assert res["attachments"][0]["metadata"]["system"]["num_wann"] == 18
    doc = to_jsonld(res, record_id="run:fe-soc")
    assert doc["attachments"][0]["@type"] == "mm:Wannier90Run"
    assert doc["attachments"][0]["wannierisation"]["omega_total_A2"] == {"value": 33.10213342, "unit": "square angstrom"}


def test_cli_file_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("matmeta.cli.OpenAICompatibleClient", lambda **kw: MockClient(_wannier_reply(True)))
    assert main(["file", str(SOC / "wannier90.wout"), "-o", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert err.count("[unverified]") == 3
    data = json.loads((tmp_path / "wannier90.wout.json").read_text())
    assert data["source"]["schema"] == "wannier"
