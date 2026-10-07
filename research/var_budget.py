"""var-budget: byte-budget guard for flesh-pits/var/ scratch growth.

Follows the validate_tree pattern of the memory-budget-enforcer skill:
a versioned, human-approvable JSON policy caps how much scratch an agent
may accumulate under var/ -- total bytes, per-file bytes with ordered
glob overrides (first match wins, null = uncapped), and max directory
depth (runaway-nesting guard, cf. the /tmp _overlay lesson).

Two entry points:
  validate_var_tree(root, policy) -> list[str]
      Audit the live tree. Empty list = within budget.
  check_write(root, policy, relpath, nbytes) -> list[str]
      Pre-write gate: simulate adding nbytes at relpath. Empty = allowed.
  assert_write_allowed(root, policy, relpath, nbytes)
      Same as check_write but raises BudgetExceeded on any violation.

Stdlib only. Byte counts stream the tree (os.scandir + st_size); file
contents are never read.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys

CONFIG_VERSION = 1


class BudgetError(ValueError):
    pass


class BudgetExceeded(BudgetError):
    pass


def parse_policy(content: str, path: str = "var-budget-policy.json") -> dict:
    try:
        value = json.loads(content)
    except ValueError as e:
        raise BudgetError("%s: invalid JSON (%s)" % (path, e))
    if not isinstance(value, dict):
        raise BudgetError("%s: expected a JSON object" % path)
    errors: list[str] = []
    allowed = {"version", "maxTotalBytes", "maxFileBytes", "maxDepth",
               "pathByteLimits", "policy_note"}
    for key in value:
        if key not in allowed:
            errors.append("%s: unknown field '%s'" % (path, key))
    v = value.get("version")
    if not (isinstance(v, int) and not isinstance(v, bool) and v == 1):
        errors.append("%s: version must be 1" % path)
    for key in ("maxTotalBytes", "maxFileBytes", "maxDepth"):
        x = value.get(key)
        if x is not None and not (isinstance(x, int) and not isinstance(x, bool) and x > 0):
            errors.append("%s: %s must be a positive integer" % (path, key))
    overrides = value.get("pathByteLimits")
    if overrides is not None and not isinstance(overrides, list):
        errors.append("%s: pathByteLimits must be an array" % path)
    if isinstance(overrides, list):
        for i, item in enumerate(overrides):
            label = "%s: pathByteLimits[%d]" % (path, i)
            if not isinstance(item, dict):
                errors.append("%s must be an object" % label)
                continue
            for key in item:
                if key not in ("pattern", "maxBytes", "maxSubtreeBytes"):
                    errors.append("%s: unknown field '%s'" % (label, key))
            pattern = item.get("pattern")
            if (not isinstance(pattern, str) or not pattern
                    or pattern.startswith("/") or ".." in pattern.split("/")):
                errors.append("%s: pattern must be a non-empty root-relative glob" % label)
            for numkey in ("maxBytes", "maxSubtreeBytes"):
                mb = item.get(numkey)
                if mb is not None and not (isinstance(mb, int) and not isinstance(mb, bool) and mb > 0):
                    errors.append("%s: %s must be a positive integer or null" % (label, numkey))
    if errors:
        raise BudgetError("; ".join(errors))
    return value


def _segments_match(pattern_segs: list[str], path_segs: list[str]) -> bool:
    # '**' matches zero or more whole segments; otherwise fnmatch per segment.
    if not pattern_segs:
        return not path_segs
    head = pattern_segs[0]
    if head == "**":
        for i in range(len(path_segs) + 1):
            if _segments_match(pattern_segs[1:], path_segs[i:]):
                return True
        return False
    if not path_segs:
        return False
    return fnmatch.fnmatchcase(path_segs[0], head) and \
        _segments_match(pattern_segs[1:], path_segs[1:])


def _match(pattern: str, relpath: str) -> bool:
    return _segments_match(pattern.split("/"), relpath.split("/"))


def _limit_for(relpath: str, overrides: list[dict], default: int | None) -> int | None:
    for item in overrides or []:
        if _match(item["pattern"], relpath):
            return item.get("maxBytes")  # null -> uncapped
    return default


def _walk(root: str):
    # Yields (relpath, size_bytes, depth). Symlinks are not followed.
    # depth = number of directories between root and the file's parent.
    stack = [(root, "")]
    while stack:
        absdir, reldir = stack.pop()
        try:
            entries = list(os.scandir(absdir))
        except OSError:
            continue
        for e in entries:
            rel = os.path.join(reldir, e.name) if reldir else e.name
            try:
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append((e.path, rel))
                elif e.is_file(follow_symlinks=False):
                    depth = rel.count(os.sep)
                    yield rel.replace(os.sep, "/"), e.stat(follow_symlinks=False).st_size, depth
            except OSError:
                continue


def validate_var_tree(root: str, policy: dict) -> list[str]:
    """Audit the live tree against the policy. Returns violation strings."""
    violations: list[str] = []
    max_total = policy.get("maxTotalBytes")
    max_file = policy.get("maxFileBytes")
    max_depth = policy.get("maxDepth")
    overrides = policy.get("pathByteLimits") or []
    subtree_caps = [(item["pattern"], item["maxSubtreeBytes"])
                    for item in overrides if item.get("maxSubtreeBytes") is not None]
    subtree_totals: dict[str, int] = {pat: 0 for pat, _ in subtree_caps}
    total = 0
    for relpath, size, depth in _walk(root):
        total += size
        if max_depth is not None and depth > max_depth:
            violations.append(
                "depth: %s at depth %d exceeds maxDepth %d" % (relpath, depth, max_depth))
        limit = _limit_for(relpath, overrides, max_file)
        if limit is not None and size > limit:
            violations.append(
                "file: %s is %d bytes, exceeds cap %d" % (relpath, size, limit))
        for pat, _ in subtree_caps:
            if _match(pat, relpath):
                subtree_totals[pat] += size
    for pat, cap in subtree_caps:
        if subtree_totals[pat] > cap:
            violations.append(
                "subtree: %s totals %d bytes, exceeds maxSubtreeBytes %d"
                % (pat, subtree_totals[pat], cap))
    if max_total is not None and total > max_total:
        violations.append(
            "total: var/ is %d bytes, exceeds maxTotalBytes %d" % (total, max_total))
    return violations


def check_write(root: str, policy: dict, relpath: str, nbytes: int) -> list[str]:
    """Pre-write gate: would writing nbytes at relpath breach the policy?

    Models the write as: existing file (if any) replaced by a file of
    nbytes; a new file adds nbytes to the total. Returns violation
    strings; empty means the write is allowed.
    """
    if nbytes < 0:
        return ["write: negative byte count %d" % nbytes]
    relpath = relpath.replace(os.sep, "/").lstrip("/")
    if not relpath or ".." in relpath.split("/"):
        return ["write: rejected path '%s'" % relpath]
    violations: list[str] = []
    max_total = policy.get("maxTotalBytes")
    max_file = policy.get("maxFileBytes")
    max_depth = policy.get("maxDepth")
    overrides = policy.get("pathByteLimits") or []
    limit = _limit_for(relpath, overrides, max_file)
    if limit is not None and nbytes > limit:
        violations.append(
            "write: %s (%d bytes) exceeds per-file cap %d" % (relpath, nbytes, limit))
    if max_depth is not None and relpath.count("/") > max_depth:
        violations.append(
            "write: %s at depth %d exceeds maxDepth %d"
            % (relpath, relpath.count("/"), max_depth))
    if max_total is not None:
        current_total = sum(size for _, size, _ in _walk(root))
        abspath = os.path.join(root, relpath)
        existing = os.path.getsize(abspath) if os.path.isfile(abspath) else 0
        projected = current_total - existing + nbytes
        if projected > max_total:
            violations.append(
                "write: %s (%d bytes) would push var/ to %d bytes, "
                "exceeds maxTotalBytes %d" % (relpath, nbytes, projected, max_total))
    for item in overrides:
        cap = item.get("maxSubtreeBytes")
        if cap is not None and _match(item["pattern"], relpath):
            current_sub = sum(size for rp, size, _ in _walk(root) if _match(item["pattern"], rp))
            abspath = os.path.join(root, relpath)
            existing = os.path.getsize(abspath) if os.path.isfile(abspath) else 0
            projected = current_sub - existing + nbytes
            if projected > cap:
                violations.append(
                    "write: %s (%d bytes) would push subtree %s to %d bytes, "
                    "exceeds maxSubtreeBytes %d" % (relpath, nbytes, item["pattern"], projected, cap))
    return violations


def assert_write_allowed(root: str, policy: dict, relpath: str, nbytes: int) -> None:
    violations = check_write(root, policy, relpath, nbytes)
    if violations:
        raise BudgetExceeded("; ".join(violations))


def _default_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    lab = os.path.dirname(here)
    return (os.path.join(here, "var-budget-policy.json"),
            os.path.join(lab, "var"))


def main(argv: list[str]) -> int:
    policy_path, var_root = _default_paths()
    if len(argv) >= 2 and argv[0] == "--policy":
        policy_path, argv = argv[1], argv[2:]
    if len(argv) >= 2 and argv[0] == "--var":
        var_root, argv = argv[1], argv[2:]
    try:
        with open(policy_path) as f:
            policy = parse_policy(f.read(), policy_path)
    except (OSError, BudgetError) as e:
        print("policy error: %s" % e, file=sys.stderr)
        return 2
    cmd = argv[0] if argv else "check"
    if cmd == "check":
        violations = validate_var_tree(var_root, policy)
        total = sum(size for _, size, _ in _walk(var_root))
        print("var/ total bytes: %d (cap %s)" % (total, policy.get("maxTotalBytes")))
        if violations:
            print("VIOLATIONS:")
            for v in violations:
                print("  - %s" % v)
            return 1
        print("within budget: no violations")
        return 0
    if cmd == "gate" and len(argv) == 3:
        relpath, nbytes = argv[1], int(argv[2])
        violations = check_write(var_root, policy, relpath, nbytes)
        if violations:
            print("WRITE REFUSED:")
            for v in violations:
                print("  - %s" % v)
            return 1
        print("write allowed: %s (%d bytes)" % (relpath, nbytes))
        return 0
    print("usage: var_budget.py [--policy P] [--var V] check | gate <relpath> <nbytes>",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
