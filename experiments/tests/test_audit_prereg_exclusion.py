"""Regression test: maturity audit must not treat preregistration files as receipts."""
import importlib.util


def _load():
    spec = importlib.util.spec_from_file_location(
        "ma", "bin/maturity_audit.py")
    ma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ma)
    return ma


def test_preregistration_excluded_from_receipt_tokens():
    ma = _load()
    ev = ("sealed preregistration experiments/preregistration_X.json; "
          "receipt `../../receipts/EXP-FP-0009.json`, hash-chained")
    toks = ma.extract_tokens(ev)
    assert not any("preregistration_" in n
                   for n in toks["receipts"] + toks["receipt_paths"])
    assert any(n.endswith("EXP-FP-0009.json") for n in toks["receipts"])


def test_real_receipt_still_checked():
    ma = _load()
    ev = "receipt `../../receipts/EXP-FP-0009.json`, hash-chained"
    toks = ma.extract_tokens(ev)
    assert toks["receipts"], "genuine receipt token must still be extracted"
