"""Tests for var_budget.py — the flesh-pits var/ byte-budget guard.

Includes the proving-ground refusal demonstration: a write that would
breach the live policy must be refused, and the refusal must name the
breached cap.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from var_budget import (BudgetExceeded, BudgetError, _match, assert_write_allowed,
                        check_write, parse_policy, validate_var_tree)

POLICY = {
    "version": 1,
    "maxTotalBytes": 1000,
    "maxFileBytes": 400,
    "maxDepth": 2,
    "pathByteLimits": [
        {"pattern": "frozen/**", "maxBytes": 600},
        {"pattern": "free.txt", "maxBytes": None},
    ],
}


def _tree(files):
    root = tempfile.mkdtemp()
    for rel, size in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x" * size)
    return root


def test_parse_policy_rejects_bad_version():
    bad = dict(POLICY, version=2)
    try:
        parse_policy(json.dumps(bad))
    except BudgetError:
        return
    raise AssertionError("bad version accepted")


def test_glob_match():
    assert _match("frozen/**", "frozen/a/b.txt")
    assert _match("frozen/**", "frozen/x.txt")
    assert not _match("frozen/**", "other/x.txt")
    assert _match("*.log", "checkpoint.log")
    assert not _match("*.log", "sub/checkpoint.log")


def test_validate_clean_tree():
    root = _tree({"a.txt": 100, "sub/b.txt": 200})
    assert validate_var_tree(root, POLICY) == []


def test_validate_total_breach():
    root = _tree({"a.txt": 400, "b.txt": 400, "c.txt": 300})
    v = validate_var_tree(root, POLICY)
    assert any(vv.startswith("total:") for vv in v), v


def test_validate_per_file_cap_with_override():
    # frozen/ override (600) beats the 400 default; other/ uses default.
    root = _tree({"frozen/big.bin": 500, "other/big.bin": 500})
    v = validate_var_tree(root, POLICY)
    assert any("other/big.bin" in vv for vv in v), v
    assert not any("frozen/big.bin" in vv for vv in v), v


def test_validate_null_override_uncapped():
    root = _tree({"free.txt": 900})
    v = validate_var_tree(root, POLICY)
    assert not any("free.txt" in vv and vv.startswith("file:") for vv in v), v


def test_validate_subtree_cap():
    pol = dict(POLICY, pathByteLimits=[
        {"pattern": "frozen/**", "maxBytes": None, "maxSubtreeBytes": 600}])
    root = _tree({"frozen/a.bin": 400, "frozen/b.bin": 300})
    v = validate_var_tree(root, pol)
    assert any(vv.startswith("subtree:") for vv in v), v


def test_check_write_refuses_subtree_breach():
    pol = dict(POLICY, pathByteLimits=[
        {"pattern": "frozen/**", "maxBytes": None, "maxSubtreeBytes": 600}])
    root = _tree({"frozen/a.bin": 500})
    v = check_write(root, pol, "frozen/new.bin", 200)
    assert any("maxSubtreeBytes" in vv for vv in v), v
    try:
        assert_write_allowed(root, pol, "frozen/new.bin", 200)
    except BudgetExceeded as e:
        assert "maxSubtreeBytes" in str(e)
        return
    raise AssertionError("frozen-subtree write was not refused")


def test_validate_depth_breach():
    root = _tree({"a/b/c/d.txt": 10})  # depth 3 > maxDepth 2
    v = validate_var_tree(root, POLICY)
    assert any(vv.startswith("depth:") for vv in v), v


def test_check_write_allows_small_write():
    root = _tree({"a.txt": 100})
    assert check_write(root, POLICY, "new/out.bin", 200) == []


def test_check_write_refuses_total_breach():
    root = _tree({"a.txt": 400, "b.txt": 400})  # 800 used, cap 1000
    v = check_write(root, POLICY, "new/big.bin", 500)  # would reach 1300
    assert any("maxTotalBytes" in vv for vv in v), v
    try:
        assert_write_allowed(root, POLICY, "new/big.bin", 500)
    except BudgetExceeded as e:
        assert "maxTotalBytes" in str(e)
        return
    raise AssertionError("over-budget write was not refused")


def test_check_write_refuses_per_file_breach():
    root = _tree({})
    v = check_write(root, POLICY, "new/big.bin", 401)
    assert any("per-file cap" in vv for vv in v), v


def test_check_write_rejects_traversal():
    root = _tree({})
    v = check_write(root, POLICY, "../escape.bin", 10)
    assert v, "path traversal not rejected"


def test_live_policy_parses():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "var-budget-policy.json")) as f:
        policy = parse_policy(f.read())
    assert policy["version"] == 1
    assert policy["maxTotalBytes"] == 167772160


def test_live_tree_within_budget():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "var-budget-policy.json")) as f:
        policy = parse_policy(f.read())
    var_root = os.path.join(os.path.dirname(here), "var")
    violations = validate_var_tree(var_root, policy)
    assert violations == [], violations


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS %s" % name)
        except Exception as e:
            failed += 1
            print("FAIL %s: %s" % (name, e))
    print("%d/%d passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
