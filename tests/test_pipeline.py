import copy
import json
from pathlib import Path

from matmeta import extract_abstract, extract_calculation, to_jsonld
from matmeta.cli import main
from matmeta.extract import merge
from matmeta.llm import MockClient, parse_json_reply
from matmeta.parsers import parse_vasp
from matmeta.schema import validate

EX = Path(__file__).resolve().parents[1] / "examples"


def _llm_reply_for_fe():
    """A plausible LLM answer: agrees with the parser, adds a note, and hallucinates one number."""
    rec = copy.deepcopy(parse_vasp(EX / "vasp_Fe_bcc").record)
    rec["notes"] = "Ferromagnetic bcc Fe, PAW_PBE Fe_pv potential."
    rec["results"]["fermi_energy_eV"] = 9.4532  # value from an earlier SCF step -> must be flagged
    return rec


def test_parser_only_mode_is_offline():
    res = extract_calculation(EX / "vasp_Fe_bcc")
    assert res["source"]["llm_used"] is False
    assert res["validation_errors"] == []


def test_llm_merge_and_conflict_detection():
    client = MockClient(_llm_reply_for_fe())
    res = extract_calculation(EX / "vasp_Fe_bcc", client)
    md = res["metadata"]
    assert md["notes"].startswith("Ferromagnetic")  # LLM-only field kept
    assert res["provenance"]["notes"] == "llm"
    assert md["results"]["fermi_energy_eV"] == 9.4817  # parser wins
    assert res["conflicts"] == [{"field": "results.fermi_energy_eV", "parser": 9.4817, "llm": 9.4532}]
    assert res["provenance"]["numerics.plane_wave_cutoff_eV"] == "parser+llm"
    assert len(client.calls) == 1


def test_invalid_llm_reply_is_retried_then_discarded():
    bad = {"software": "VASP"}  # wrong type, missing keys
    client = MockClient(bad)
    res = extract_calculation(EX / "qe_Si" / "si.scf.in", client)
    assert len(client.calls) == 2  # one retry with the validation errors
    assert "failed validation" in client.calls[1][1]
    assert res["llm_validation_errors"]
    assert res["validation_errors"] == []  # final record still valid (parser values only)
    assert res["metadata"]["structure"]["formula"] == "Si2"


def test_merge_numeric_tolerance():
    _, prov, conf = merge({"a": 1.0000001}, {"a": 1.0})
    assert conf == [] and prov["a"] == "parser+llm"


def test_abstract_extraction():
    reply = {
        "materials": [{"name": "iron", "formula": "Fe", "dimensionality": "3D", "role": "ferromagnet"}],
        "methods": [{"name": "DFT (PBE + SOC)", "kind": "computational", "software": "VASP"}],
        "properties": [{"name": "spin magnetic moment", "value": "2.2", "unit": "Bohr magneton"}],
        "keywords": ["magnetism", "orbital Hall effect"],
    }
    res = extract_abstract((EX / "abstract.txt").read_text(), MockClient(reply))
    assert res["validation_errors"] == []
    assert validate(res["metadata"], "abstract") == []


def test_jsonld_drops_nulls_and_adds_units():
    res = extract_calculation(EX / "vasp_Fe_bcc")
    doc = to_jsonld(res, record_id="run:fe")
    assert doc["@id"] == "run:fe"
    assert doc["numerics"]["plane_wave_cutoff_eV"] == {"value": 500.0, "unit": "eV"}
    assert "hubbard_u" not in doc["method"]


def test_json_reply_parsing():
    assert parse_json_reply('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_reply('blah {"a": 2} blah') == {"a": 2}


def test_cli_offline(tmp_path, capsys):
    assert main(["calc", str(EX / "vasp_Fe_bcc"), "--no-llm", "-o", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "vasp_Fe_bcc.json").read_text())
    assert data["metadata"]["structure"]["formula"] == "Fe2"


def test_openai_compatible_client_against_local_server():
    """Exercise the real HTTP client against a tiny local server that mimics /chat/completions."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from matmeta.llm import OpenAICompatibleClient

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["body"], seen["auth"] = body, self.headers.get("Authorization")
            reply = {"choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = OpenAICompatibleClient(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k", model="m")
        assert client.complete_json("sys", "usr") == {"ok": True}
        assert seen["auth"] == "Bearer k"
        assert seen["body"]["model"] == "m"
        assert seen["body"]["response_format"] == {"type": "json_object"}
        assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]
    finally:
        srv.shutdown()
