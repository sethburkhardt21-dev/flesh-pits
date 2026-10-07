"""Per-lane ID allocation registry — prevents concurrent-lane ID collisions.

INCIDENT (2026-10-07 ~12:40 UTC): two concurrent lanes minted EXP-FP-0050
without a shared allocation. Lane A's free-check at ~12:31 UTC was clean,
but an unidentified concurrent lane wrote receipts/EXP-FP-0050.json (hash
c55fdebe...) between ~12:31 and 12:39:48 UTC; lane A's
harness.write_receipt() (open(path, "w")) overwrote it at 12:40:29 UTC.
The lost content is unrecoverable (verified across the workspace, git
history, and agent transcripts). Documented in research/negative_results.md
("## 2026-10-07 — EXP-FP-0050"); the collision record is preserved in this
registry under "incidents".

PROCEDURE (binding for all lanes):
  1. Claim an ID family (e.g. "EXP-FP-008x") or an exact experiment ID via
     claim_id_family() / claim_id() BEFORE writing anything.
  2. Only then mint IDs inside the claimed family and write receipts.
  Never mint blind.

ATOMICITY: every claim runs under the file-lock skill
(~/workspace/skills/file-lock/file_lock.py) — the lock file is created
atomically (O_CREAT|O_EXCL), so exactly one of two concurrent claimants
wins; the loser gets IDClaimError. Claims are read-modify-write inside the
lock, so no two lanes can hold the same family.

FAIL CLOSED: if experiments/ID_REGISTRY does not exist, claims RAISE rather
than creating a fresh empty registry — a missing registry means the lab's
allocation truth is absent, and minting against an empty view would repeat
the collision. Build (or rebuild) it with build_registry().

Registry layout (experiments/ID_REGISTRY, JSON):
  {
    "version": 1,
    "created_utc": "...",
    "families": {
      "EXP-FP-008x": {"lane": "ecr-replication", "purpose": "...",
                      "claimed_utc": "...", "claimed_by_pid": 1234,
                      "notes": [...]},
    },
    "ids": {
      "EXP-FP-0006": {"source": "receipt", "file": "receipts/EXP-FP-0006.json",
                      "family": null, "lane": null},
      ...
    },
    "incidents": [
      {"date": "2026-10-07", "ids": ["EXP-FP-0050"], ...},
    ],
  }

Stdlib only (plus the file-lock skill module).
"""

import datetime
import json
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/skills/file-lock"))
from file_lock import FileLock  # noqa: E402

REGISTRY_FILENAME = "ID_REGISTRY"
VERSION = 1

# Phase 7 pre-allocations (owner-standing, 2026-10-07).
PREALLOCATED_FAMILIES = {
    "EXP-FP-008x": {"lane": "ecr-replication",
                    "purpose": "Phase 7 pre-allocation: ECR replication lane"},
    "EXP-FP-009x": {"lane": "transfer-dynamics",
                    "purpose": "Phase 7 pre-allocation: transfer-dynamics lane"},
    "EXP-FP-010x": {"lane": "calibration",
                    "purpose": "Phase 7 pre-allocation: calibration lane"},
}


class IDClaimError(Exception):
    """A claim failed: family/ID already allocated, registry missing, or
    the registry lock is held by another claimant."""


# -- paths ---------------------------------------------------------------

def _default_experiments_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _registry_path(experiments_dir=None):
    d = experiments_dir or _default_experiments_dir()
    return os.path.join(d, REGISTRY_FILENAME)


def _lock_path(registry_path):
    return registry_path + ".lock"


# -- registry IO (callers must hold the lock) -----------------------------

def _empty_registry():
    return {
        "version": VERSION,
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "families": {},
        "ids": {},
        "incidents": [],
    }


def _save_locked(registry_path, reg):
    """Atomic write (temp + os.replace) so a crashed writer never leaves a
    truncated registry."""
    tmp = registry_path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, registry_path)


def _load_locked(registry_path):
    with open(registry_path) as f:
        reg = json.load(f)
    if reg.get("version") != VERSION:
        raise IDClaimError(
            f"registry version {reg.get('version')} != {VERSION}: "
            "refusing to claim against an unknown schema")
    return reg


def load_registry(experiments_dir=None):
    """Read-only load of the registry (no lock; for inspection)."""
    with open(_registry_path(experiments_dir)) as f:
        return json.load(f)


# -- family matching -------------------------------------------------------

def _family_regex(family):
    """Family pattern: 'EXP-FP-008x' matches IDs like EXP-FP-0080..0089.

    Trailing 'x' means one-or-more digits."""
    if not family or not family.endswith("x"):
        raise IDClaimError(
            f"invalid family {family!r}: must end with 'x' (e.g. 'EXP-FP-008x')")
    return re.compile("^" + re.escape(family[:-1]) + r"\d+$")


def family_contains(family, experiment_id):
    return bool(_family_regex(family).match(experiment_id))


# -- claims -----------------------------------------------------------------

def _claim_locked(lock_path):
    lock = FileLock(lock_path)
    if not lock.acquire():
        raise IDClaimError(
            f"could not acquire registry lock {lock_path}: held by another "
            "claimant — retry, do not mint blind")
    return lock


def claim_id_family(family, lane, purpose="", claimed_by=None,
                    experiments_dir=None):
    """Atomically claim an ID family for a lane.

    Raises IDClaimError if the family is already claimed by a DIFFERENT
    lane, or if the registry does not exist (fail closed). Re-claiming by
    the SAME lane is idempotent and returns the existing record.
    """
    _family_regex(family)  # validates format, raises on garbage
    registry_path = _registry_path(experiments_dir)
    if not os.path.exists(registry_path):
        raise IDClaimError(
            f"registry {registry_path} missing: allocation truth absent, "
            "refusing to mint blind — run build_registry() first")
    lock = _claim_locked(_lock_path(registry_path))
    try:
        reg = _load_locked(registry_path)
        existing = reg["families"].get(family)
        if existing is not None:
            if existing["lane"] != lane:
                raise IDClaimError(
                    f"family {family} already claimed by lane "
                    f"{existing['lane']!r} (claimed {existing['claimed_utc']})")
            return existing
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        notes = []
        for exp_id, info in reg["ids"].items():
            if family_contains(family, exp_id):
                notes.append(
                    f"ID {exp_id} already in use ({info.get('source')}: "
                    f"{info.get('file')}) at claim time")
        record = {
            "lane": lane,
            "purpose": purpose,
            "claimed_utc": now,
            "claimed_by_pid": claimed_by if claimed_by is not None
            else os.getpid(),
            "notes": notes,
        }
        reg["families"][family] = record
        _save_locked(registry_path, reg)
        return record
    finally:
        lock.release()


def claim_id(experiment_id, lane, experiments_dir=None):
    """Atomically claim an exact experiment ID for a lane.

    Raises IDClaimError if the ID is already allocated (claimed, or in use
    by a preregistration/receipt scan) or falls inside a family claimed by a
    different lane. If it falls inside the SAME lane's family, the exact ID
    is recorded as minted under that family.
    """
    registry_path = _registry_path(experiments_dir)
    if not os.path.exists(registry_path):
        raise IDClaimError(
            f"registry {registry_path} missing: allocation truth absent, "
            "refusing to mint blind — run build_registry() first")
    lock = _claim_locked(_lock_path(registry_path))
    try:
        reg = _load_locked(registry_path)
        existing = reg["ids"].get(experiment_id)
        if existing is not None:
            raise IDClaimError(
                f"ID {experiment_id} already allocated "
                f"(source: {existing.get('source')}, "
                f"file: {existing.get('file')}, lane: {existing.get('lane')})")
        home_family = None
        for family, info in reg["families"].items():
            if family_contains(family, experiment_id):
                if info["lane"] != lane:
                    raise IDClaimError(
                        f"ID {experiment_id} falls inside family {family}, "
                        f"claimed by lane {info['lane']!r}")
                home_family = family
                break
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = {
            "source": "claimed",
            "file": None,
            "family": home_family,
            "lane": lane,
            "claimed_utc": now,
        }
        reg["ids"][experiment_id] = record
        _save_locked(registry_path, reg)
        return record
    finally:
        lock.release()


def is_allocated(experiment_id, experiments_dir=None):
    """(allocated: bool, info: dict|None) — inspection helper, no lock."""
    reg = load_registry(experiments_dir)
    info = reg["ids"].get(experiment_id)
    if info is not None:
        return True, info
    for family, finfo in reg["families"].items():
        if family_contains(family, experiment_id):
            return True, {"source": "family", "family": family,
                          "lane": finfo["lane"]}
    return False, None


# -- seeding -----------------------------------------------------------------

def _scan_in_use(experiments_dir):
    """Every ID minted so far: experiments/preregistration_*.json and
    receipts/*.json. Returns {id: info}."""
    ids = {}
    pre = experiments_dir  # preregistration files live in experiments/ itself
    for fn in sorted(os.listdir(pre)):
        if not (fn.startswith("preregistration_") and fn.endswith(".json")):
            continue
        p = os.path.join(pre, fn)
        try:
            with open(p) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        exp_id = d.get("experiment_id") or d.get("id")
        if exp_id:
            ids.setdefault(exp_id, {"source": "preregistration",
                                    "file": f"experiments/{fn}",
                                    "family": None, "lane": None})
    lab_root = os.path.dirname(experiments_dir)
    rec_dir = os.path.join(lab_root, "receipts")
    if os.path.isdir(rec_dir):
        for fn in sorted(os.listdir(rec_dir)):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(rec_dir, fn)
            try:
                with open(p) as f:
                    d = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            exp_id = d.get("experiment_id")
            if exp_id:
                ids.setdefault(exp_id, {"source": "receipt",
                                        "file": f"receipts/{fn}",
                                        "family": None, "lane": None})
    return ids


def _collision_incident():
    return {
        "date": "2026-10-07",
        "time_utc": "~12:40",
        "ids": ["EXP-FP-0050"],
        "lost_receipt_hash": "c55fdebe…",
        "surviving_receipt_hash": "75f474d1…",
        "cause": (
            "Two concurrent lanes minted EXP-FP-0050 with no shared "
            "allocation registry. Lane A's free-check at ~12:31 UTC was "
            "clean, but an unidentified concurrent lane wrote "
            "receipts/EXP-FP-0050.json (hash c55fdebe…) between ~12:31 and "
            "12:39:48 UTC; lane A's harness.write_receipt() "
            "(open(path, 'w')) overwrote it at 12:40:29 UTC. Lost content "
            "unrecoverable — verified across the workspace, git history, "
            "and agent transcripts."),
        "collateral": (
            "EXP-FP-0040.json's prev_receipt_hash now dangles (points at "
            "c55fdebe…); preserved as-is per lab law — never rewritten. "
            "verify_chain flags it."),
        "fix": (
            "2026-10-07 systemic fix: per-lane ID allocation registry "
            "(experiments/ID_REGISTRY) with atomic family/ID claims under "
            "the file-lock skill (claim before writing, never mint blind); "
            "harness.write_receipt() fails closed on existing paths with an "
            "explicit supersede path recording superseded_previous_hash + "
            "supersede_reason. Incident also recorded in "
            "research/negative_results.md ('## 2026-10-07 — EXP-FP-0050')."),
    }


def build_registry(experiments_dir=None, overwrite=False):
    """Seed experiments/ID_REGISTRY from live allocation truth.

    Scans experiments/preregistration_*.json and receipts/*.json for IDs in
    use, records the Phase 7 family pre-allocations, and embeds the
    2026-10-07 collision incident. Atomic under the registry lock.
    Raises IDClaimError if the registry exists and overwrite is False
    (never clobber seeded allocation truth silently).
    """
    experiments_dir = experiments_dir or _default_experiments_dir()
    registry_path = _registry_path(experiments_dir)
    lock_path = _lock_path(registry_path)
    os.makedirs(experiments_dir, exist_ok=True)
    lock = _claim_locked(lock_path)
    try:
        if os.path.exists(registry_path) and not overwrite:
            raise IDClaimError(
                f"registry {registry_path} already exists: refusing to "
                "rebuild over seeded allocation truth (pass overwrite=True "
                "to rebuild deliberately)")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        reg = _empty_registry()
        reg["created_utc"] = now
        reg["ids"] = _scan_in_use(experiments_dir)
        for family, meta in PREALLOCATED_FAMILIES.items():
            reg["families"][family] = {
                "lane": meta["lane"],
                "purpose": meta["purpose"],
                "claimed_utc": now,
                "claimed_by_pid": os.getpid(),
                "notes": ["seeded by build_registry from Phase 7 "
                          "pre-allocations (owner-standing)"],
            }
        reg["incidents"] = [_collision_incident()]
        _save_locked(registry_path, reg)
        return reg
    finally:
        lock.release()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Flesh Pits per-lane ID allocation registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="seed the registry from live truth")
    b.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("claim-family",
                       help="claim an ID family for a lane")
    c.add_argument("family")
    c.add_argument("lane")
    c.add_argument("--purpose", default="")
    i = sub.add_parser("claim-id", help="claim an exact experiment ID")
    i.add_argument("experiment_id")
    i.add_argument("lane")
    s = sub.add_parser("status", help="show allocation for an ID")
    s.add_argument("experiment_id")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            reg = build_registry(overwrite=args.overwrite)
            print(f"registry built: {_registry_path()} "
                  f"({len(reg['ids'])} IDs, {len(reg['families'])} families, "
                  f"{len(reg['incidents'])} incidents)")
        elif args.cmd == "claim-family":
            rec = claim_id_family(args.family, args.lane,
                                  purpose=args.purpose)
            print(f"claimed {args.family} for lane {args.lane}")
            if rec["notes"]:
                for n in rec["notes"]:
                    print(f"  note: {n}")
        elif args.cmd == "claim-id":
            claim_id(args.experiment_id, args.lane)
            print(f"claimed {args.experiment_id} for lane {args.lane}")
        elif args.cmd == "status":
            ok, info = is_allocated(args.experiment_id)
            print(f"{args.experiment_id}: "
                  f"{'ALLOCATED ' + json.dumps(info, sort_keys=True) if ok else 'FREE'}")
    except IDClaimError as e:
        print(f"CLAIM REFUSED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
