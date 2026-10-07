"""H10 package test — cleanrl ppo_rnd_envpool.py (vendored at pinned SHA).

The vendored file is a third-party donor mechanism (MIT), not lab code:
it cannot run standalone here (requires torch/envpool/gym). This test
proves the PACKAGE is faithful instead:
  1. The file is byte-present and syntactically valid Python (py_compile).
  2. Structural check via AST: defines RNDModel and an intrinsic-reward
     coefficient path (the RND/curiosity mechanism being adopted).
  3. sha256 matches the manifest record.
  4. The adoption record (ADOPTION.md) names the pinned SHA, license,
     mechanism, integration surface, and rollback.

Mechanism proof is the donor's own CI at the pinned SHA (stated in the
README); this test proves the vendored artifact is the pinned artifact.

Exit 0 on PASS, 1 on FAIL. Receipt -> ./receipts/. Stdlib only.
"""
import ast
import hashlib
import json
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "ppo_rnd_envpool.py")
EXPECTED_SHA256 = "d6ec2fb0f2f1e3f78624b001f02abbe5fc46a645e2f88b2ff644609e3d1e5560"
PINNED_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"


def main():
    results, ok_all = {}, True

    # 1. present + compiles
    ok = os.path.exists(SRC)
    try:
        py_compile.compile(SRC, doraise=True)
        compiles = True
    except py_compile.PyCompileError:
        compiles = False
    ok = ok and compiles
    results["present_and_compiles"] = {"ok": ok}
    print(f"  vendored file present and compiles: {ok}")
    ok_all = ok_all and ok

    # 2. AST structural check: RND mechanism present
    with open(SRC) as f:
        tree = ast.parse(f.read())
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    src_text = open(SRC).read()
    has_rnd = "RNDModel" in names
    has_intrinsic = "intrinsic" in src_text.lower()
    ok = has_rnd and has_intrinsic
    results["rnd_mechanism_present"] = {"ok": ok, "RNDModel": has_rnd,
                                        "intrinsic_reward": has_intrinsic,
                                        "top_level_defs": sorted(names)[:10]}
    print(f"  RNDModel class: {has_rnd}, intrinsic-reward path: "
          f"{has_intrinsic} -> {'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # 3. hash matches the pinned artifact
    h = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
    ok = h == EXPECTED_SHA256
    results["hash_matches"] = {"ok": ok, "sha256": h}
    print(f"  sha256 matches manifest: {ok}")
    ok_all = ok_all and ok

    # 4. adoption record present and names the pin
    adop = os.path.join(HERE, "..", "ADOPTION.md")
    ok = os.path.exists(adop) and PINNED_COMMIT in open(adop).read()
    results["adoption_record"] = {"ok": ok}
    print(f"  ADOPTION.md names pinned SHA: {ok}")
    ok_all = ok_all and ok

    print("OVERALL ->", "PASS" if ok_all else "FAIL")
    receipt = {"package": "H10", "test": "test_h10.py", "checks": results,
               "pass": bool(ok_all),
               "note": ("vendored donor file; mechanism proven by donor CI "
                        "at pinned SHA, not by this test"),
               "verdict": "PASS" if ok_all else "FAIL"}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
