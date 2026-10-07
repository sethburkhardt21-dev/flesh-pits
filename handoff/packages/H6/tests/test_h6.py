"""H6 package test — memory_provenance hash-chaining.

Standalone proof from the packaged source:
  1. Bootstrap: a first write degrades to 'untrusted' even when the
     caller claims 'agent' (no prior provenance can vouch).
  2. Retention: a second write whose content_before hash matches the
     stored file_hash retains 'agent'.
  3. Tamper: content_before that does NOT match the stored hash degrades
     to 'untrusted' (hash chain broken).
  4. Rollback: the returned closure restores the previous record
     (compare-and-set on reservation_id).
  5. Allowlist: paths outside the policy are refused (returns None).
  6. Read-back: read_provenance returns the chained record; a hand-
     tampered store entry fails validation (reads back None).

Exit 0 on PASS, 1 on FAIL. Receipt -> ./receipts/. Stdlib only.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from memory_provenance import ProvenanceStore, sha256_hex

WS = "/tmp/h6test-ws"
REL = "memory/test.md"


def fresh_store(tmp):
    return ProvenanceStore(os.path.join(tmp, "prov.json")), tmp


def main():
    results, ok_all = {}, True
    with tempfile.TemporaryDirectory() as tmp:
        store, _ = fresh_store(tmp)

        # 1. bootstrap degrades
        rb = store.record_write_provenance(
            workspace_dir=WS, relative_path=REL,
            content_before="", content_after="hello",
            origin_class="agent", observed_at=1)
        rec = store.read_provenance(workspace_dir=WS, relative_path=REL)
        ok = rb is not None and rec["origin_class"] == "untrusted" \
            and rec["file_hash"] == sha256_hex("hello")
        results["bootstrap_degrades"] = {"ok": ok}
        print(f"  bootstrap: origin={rec['origin_class']} (need untrusted) "
              f"-> {'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

        # 2. retention on matching chain. The retention law requires a
        # previous 'agent' record; the public API can never mint the
        # first one (bootstrap always degrades) — that genesis path is
        # the documented INCONCLUSIVE design gap. Here we seed one
        # directly to test the retention RULE itself.
        from memory_provenance import (PROVENANCE_VERSION,
                                       normalize_workspace_key,
                                       normalize_relative_path, _store_key)
        ws_key = normalize_workspace_key(WS)
        rel_n = normalize_relative_path(REL, store._policy)
        store._records[_store_key(ws_key, rel_n)] = {
            "version": PROVENANCE_VERSION, "workspace_key": ws_key,
            "relative_path": rel_n, "file_hash": sha256_hex("hello"),
            "origin_class": "agent", "observed_at": 1,
            "reservation_id": "seeded-genesis"}
        store._save()
        store.record_write_provenance(
            workspace_dir=WS, relative_path=REL,
            content_before="hello", content_after="hello world",
            origin_class="agent", observed_at=2)
        rec = store.read_provenance(workspace_dir=WS, relative_path=REL)
        ok = rec["origin_class"] == "agent"
        results["retention"] = {"ok": ok}
        print(f"  retention: origin={rec['origin_class']} (need agent) -> "
              f"{'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

        # 3. tamper breaks the chain
        store.record_write_provenance(
            workspace_dir=WS, relative_path=REL,
            content_before="FORGED", content_after="evil",
            origin_class="agent", observed_at=3)
        rec = store.read_provenance(workspace_dir=WS, relative_path=REL)
        ok = rec["origin_class"] == "untrusted"
        results["tamper_breaks_chain"] = {"ok": ok}
        print(f"  tamper: origin={rec['origin_class']} (need untrusted) -> "
              f"{'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

        # 4. rollback restores
        rb2 = store.record_write_provenance(
            workspace_dir=WS, relative_path=REL,
            content_before="evil", content_after="evil2",
            origin_class="agent", observed_at=4)
        rb2()
        rec = store.read_provenance(workspace_dir=WS, relative_path=REL)
        ok = rec["file_hash"] == sha256_hex("evil")
        results["rollback"] = {"ok": ok}
        print(f"  rollback: restored hash matches 'evil' -> "
              f"{'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

        # 5. allowlist refusal
        r = store.record_write_provenance(
            workspace_dir=WS, relative_path="/etc/passwd",
            content_before="", content_after="x",
            origin_class="agent", observed_at=5)
        ok = r is None
        results["allowlist_refusal"] = {"ok": ok}
        print(f"  allowlist: outside-policy path refused -> "
              f"{'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

        # 6. store tamper breaks the chain, not the format check.
        # Swapping file_hash to another well-formed hash keeps the record
        # format-valid (read still returns it), but the hash chain is
        # broken: a continuity claim on the ORIGINAL content now degrades
        # to 'untrusted'. A malformed record (bad version) is rejected.
        with open(os.path.join(tmp, "prov.json")) as f:
            raw = json.load(f)
        key = next(iter(raw))
        raw[key]["file_hash"] = "f" * 64
        raw[key]["origin_class"] = "agent"
        with open(os.path.join(tmp, "prov.json"), "w") as f:
            json.dump(raw, f)
        store2 = ProvenanceStore(os.path.join(tmp, "prov.json"))
        store2.record_write_provenance(
            workspace_dir=WS, relative_path=REL,
            content_before="evil", content_after="evil3",
            origin_class="agent", observed_at=9)
        rec = store2.read_provenance(workspace_dir=WS, relative_path=REL)
        chain_broken = rec["origin_class"] == "untrusted"
        raw[key]["version"] = "tampered"
        with open(os.path.join(tmp, "prov.json"), "w") as f:
            json.dump(raw, f)
        store3 = ProvenanceStore(os.path.join(tmp, "prov.json"))
        malformed_rejected = store3.read_provenance(
            workspace_dir=WS, relative_path=REL) is None
        ok = chain_broken and malformed_rejected
        results["store_tamper_detected"] = {"ok": ok,
                                            "chain_broken": chain_broken,
                                            "malformed_rejected":
                                            malformed_rejected}
        print(f"  store tamper: chain broken={chain_broken}, malformed "
              f"rejected={malformed_rejected} -> {'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

    print("OVERALL ->", "PASS" if ok_all else "FAIL")
    receipt = {"package": "H6", "test": "test_h6.py", "checks": results,
               "pass": bool(ok_all),
               "note": ("'agent' provenance origin unreachable via the "
                        "public API — design gap, INCONCLUSIVE item"),
               "verdict": "PASS" if ok_all else "FAIL"}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
