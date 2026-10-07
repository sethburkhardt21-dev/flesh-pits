"""Evidence store with the P-B03 quarantine invariant (additive).

EpisodicStore: an append-only episodic memory whose admission contract is
SIMULATION-NEVER-BECOMES-MEMORY (audit P-B03) + VERIFIED-OUTCOME-CALIBRATION-
ONLY admission discipline (audit P-B05):

  - every admitted record MUST carry source == "observed";
  - any record with source != "observed" (\"predicted\", \"simulated\",
    missing, or anything else) is REJECTED with StoreQuarantineError and a
    rejection counter is incremented;
  - each accepted record carries chain_hash = sha256(prev_chain_hash +
    canonical_json(record)) — tamper-evident, mirroring the receipts chain;
  - restore(snapshot) re-admits every record through the SAME gate, so
    quarantine is enforced on restore (no backdoor through snapshots);
  - scan() audits every stored record for tag violations.

The store deliberately does NOT inspect record CONTENT for "prediction-like"
features: the invariant is tag-enforced at admission, honestly documented as
the boundary in the preregistration (a forged \"observed\" tag would pass).

Stdlib only.
"""

import hashlib

from env_interface import canonical_json


class StoreQuarantineError(ValueError):
    """Raised when a non-observed record is offered for admission."""


class EpisodicStore:
    OBSERVED = "observed"
    GENESIS_CHAIN = "GENESIS"

    def __init__(self):
        self._records = []
        self._chain_head = self.GENESIS_CHAIN
        self.rejections = 0
        self.admissions = 0

    # -- admission ------------------------------------------------------
    def admit(self, record: dict) -> str:
        """Admit one record; returns its chain_hash. Rejects non-observed."""
        if not isinstance(record, dict) or record.get("source") != self.OBSERVED:
            self.rejections += 1
            raise StoreQuarantineError(
                "quarantine: only source=='observed' records may be admitted; "
                f"got source={record.get('source')!r}" if isinstance(record, dict)
                else "quarantine: record is not a dict")
        body = dict(record)
        chain_hash = hashlib.sha256(
            (self._chain_head + canonical_json(body)).encode("utf-8")
        ).hexdigest()
        body["chain_hash"] = chain_hash
        self._records.append(body)
        self._chain_head = chain_hash
        self.admissions += 1
        return chain_hash

    # -- audit ----------------------------------------------------------
    def scan(self) -> list:
        """Return every stored record violating the quarantine invariant."""
        return [r for r in self._records if r.get("source") != self.OBSERVED]

    def verify_chain(self) -> bool:
        """Recompute the full chain; True iff every link validates."""
        head = self.GENESIS_CHAIN
        for r in self._records:
            body = {k: v for k, v in r.items() if k != "chain_hash"}
            expected = hashlib.sha256(
                (head + canonical_json(body)).encode("utf-8")).hexdigest()
            if expected != r["chain_hash"]:
                return False
            head = r["chain_hash"]
        return head == self._chain_head

    # -- persistence (quarantine enforced on restore) -------------------
    def to_snapshot(self) -> dict:
        return {"records": [dict(r) for r in self._records],
                "chain_head": self._chain_head,
                "admissions": self.admissions,
                "rejections": self.rejections}

    def restore(self, snapshot: dict) -> None:
        """Restore from snapshot; every record re-passes the admission gate."""
        records = snapshot.get("records", [])
        # Bug fix 2026-10-07 (found by EXP-FP-0060 red-team): an empty
        # snapshot with chain_head None is a legitimate fresh store, not
        # a quarantine violation.
        if not records:
            if snapshot.get("chain_head") not in (None, self.GENESIS_CHAIN):
                raise StoreQuarantineError(
                    "quarantine: empty snapshot with unexpected chain head")
            self._records = []
            self._chain_head = self.GENESIS_CHAIN
            return
        for r in records:
            if r.get("source") != self.OBSERVED:
                self.rejections += 1
                raise StoreQuarantineError(
                    "quarantine: snapshot contains non-observed record; "
                    "restore refused, store unchanged")
        # Chain-integrity check on the raw records before adopting.
        head = self.GENESIS_CHAIN
        for r in records:
            if "chain_hash" not in r:
                raise StoreQuarantineError(
                    "quarantine: snapshot record missing chain_hash")
            body = {k: v for k, v in r.items() if k != "chain_hash"}
            expected = hashlib.sha256(
                (head + canonical_json(body)).encode("utf-8")).hexdigest()
            if expected != r["chain_hash"]:
                raise StoreQuarantineError(
                    "quarantine: snapshot chain broken; restore refused")
            head = r["chain_hash"]
        if head != snapshot.get("chain_head"):
            raise StoreQuarantineError(
                "quarantine: snapshot chain head mismatch; restore refused")
        self._records = [dict(r) for r in records]
        self._chain_head = snapshot.get("chain_head")
        self.admissions += len(records)

    def __len__(self):
        return len(self._records)
