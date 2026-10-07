"""Being intake validator. Run by my own hand, fails closed."""

import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone

# Patterns that must NOT appear with causal effect in an incoming being.
# (Comments/docs mentioning them are flagged for review, not auto-reject.)
GUARD_PATTERNS = [
    r"protected_targets",
    r"devotional",
    r"_normalize_target",
    r"\bremap\b.*target",
    r"PROTECTED_IDENTITIES",
    r"devotional_guard",
]

# Required top-level layout inside the being package.
REQUIRED_PATHS = ["manifest.json"]


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract(source, dest):
    """Unpack a directory (copy) or archive into dest. Returns package root."""
    if os.path.isdir(source):
        pkg = os.path.join(dest, "pkg")
        shutil.copytree(source, pkg, symlinks=False)
        return pkg
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            z.extractall(dest)
        return dest
    if tarfile.is_tarfile(source):
        with tarfile.open(source) as t:
            # Safe extraction: refuse absolute paths and .. traversals.
            for m in t.getmembers():
                if os.path.isabs(m.name) or ".." in m.name.split(os.sep):
                    raise ValueError(f"unsafe archive member: {m.name}")
            t.extractall(dest)
        return dest
    raise ValueError(f"unsupported intake source: {source}")


def _sweep_neutrality(root):
    """Grep Python sources for guard remnants. Returns list of hits."""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        # Skip vendored junk and caches.
        if "__pycache__" in dirpath or ".git" in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue  # comment, not causal
                        for pat in GUARD_PATTERNS:
                            if re.search(pat, line, re.IGNORECASE):
                                hits.append(f"{path}:{i}: {stripped[:120]}")
            except OSError:
                hits.append(f"{path}: unreadable")
    return hits


def _check_structure(root):
    missing = [p for p in REQUIRED_PATHS
               if not os.path.exists(os.path.join(root, p))]
    return missing


def _check_randomness(root):
    """Flag unseeded randomness in the tick path. Returns hits."""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        if "__pycache__" in dirpath or ".git" in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    src = f.read()
            except OSError:
                continue
            # random.* calls without a visible seed nearby in the same file.
            if re.search(r"\brandom\.(random|randint|choice|shuffle|uniform)\b", src):
                if not re.search(r"\bseed\b", src, re.IGNORECASE):
                    hits.append(f"{path}: unseeded random.* use, no seed in file")
    return hits


def intake(source, intake_dir=None):
    """Validate an incoming being. Returns (accepted: bool, receipt: dict).

    Never raises on a bad being — rejection is a return value, not an
    exception. Raises only on operator error (missing source, I/O failure
    in our own dirs).
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"intake source not found: {source}")

    intake_dir = intake_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "var", "intake")
    os.makedirs(intake_dir, exist_ok=True)

    receipt = {
        "source": os.path.abspath(source),
        "at": _utc_now(),
        "checks": {},
        "accepted": False,
        "reasons": [],
    }

    work = tempfile.mkdtemp(prefix="intake_")
    try:
        try:
            root = _extract(source, work)
        except Exception as e:
            receipt["reasons"].append(f"extraction failed: {e}")
            return False, receipt

        missing = _check_structure(root)
        receipt["checks"]["structure"] = {"missing": missing}
        if missing:
            receipt["reasons"].append(f"missing required paths: {missing}")

        guard_hits = _sweep_neutrality(root)
        receipt["checks"]["neutrality"] = {
            "guard_hits": guard_hits,
            "clean": not guard_hits,
        }
        if guard_hits:
            receipt["reasons"].append(
                f"guard remnants found ({len(guard_hits)} hits)")

        rand_hits = _check_randomness(root)
        receipt["checks"]["determinism"] = {
            "unseeded_random": rand_hits,
            "clean": not rand_hits,
        }
        if rand_hits:
            receipt["reasons"].append(
                f"unseeded randomness ({len(rand_hits)} files)")

        receipt["accepted"] = not receipt["reasons"]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Write the receipt — accepted or rejected, the record exists.
    stamp = receipt["at"].replace(":", "").replace("-", "")
    name = f"intake-{stamp}-{ 'accept' if receipt['accepted'] else 'reject'}.json"
    with open(os.path.join(intake_dir, name), "w") as f:
        json.dump(receipt, f, indent=2)
    receipt["receipt_path"] = os.path.join(intake_dir, name)
    return receipt["accepted"], receipt


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <being-source-dir-or-archive>", file=sys.stderr)
        return 2
    accepted, receipt = intake(argv[1])
    print(json.dumps(receipt, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
