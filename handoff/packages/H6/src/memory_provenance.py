"""Hash-chained provenance for memory writes.

Clean-room reimplementation of the audited design:
- Every memory write records provenance: sha256 of the new content,
  origin class ('agent' | 'untrusted'), observed-at timestamp, session ids,
  and the normalized relative path.
- Origin retention rule (CORRECTED — repair lane B2): 'agent' origin is
  retained ONLY when a previous provenance record EXISTS with 'agent'
  origin AND the pre-write content hash matches it. A first write — no
  previous record — can never retain 'agent'; it degrades to 'untrusted'
  even when the caller claims 'agent' origin. The previous behavior
  (first write retained 'agent' with no previous provenance) contradicted
  this rule and the documented retention law; it was a bootstrap hole.
- Compare-and-set rollback: record returns a rollback closure that restores
  the previous record only if the reservation id still matches.
- Parameterizable path allowlist (PathPolicy): the default policy keeps
  the audited behavior — MEMORY.md / memory.md / USER.md at the root,
  users/<id>/USER.md, and memory/*.md with the dream subtrees
  (memory/dreaming/, memory/.dreams/) excluded. Absolute paths, '..'
  traversal, and non-Markdown paths are rejected. Consumers with a
  different memory layout pass their own PathPolicy; the mechanism
  (hash chain, retention rule, CAS rollback) is layout-independent.
- The workspace key is the sha256 of the physical (symlink-resolved)
  workspace directory, so alias paths cannot split trust records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass

PROVENANCE_VERSION = 1
MAX_ENTRIES = 50_000


@dataclass(frozen=True)
class PathPolicy:
    """Parameterizable path allowlist for memory-provenance.

    Strip/generalization parameters (the complete list):
      root_filenames    — exact filenames allowed at the workspace root
      memory_prefix     — subtree prefix covered by the Markdown rule
      markdown_suffix   — required suffix inside that subtree
      excluded_prefixes — subtree prefixes excluded from the Markdown rule
      user_file_pattern — regex for per-user identity files
    """
    root_filenames: tuple = ("MEMORY.md", "memory.md", "USER.md")
    memory_prefix: str = "memory/"
    markdown_suffix: str = ".md"
    excluded_prefixes: tuple = ("memory/dreaming/", "memory/.dreams/")
    user_file_pattern: str = (
        r"^users/[A-Za-z0-9][A-Za-z0-9_-]{0,127}/USER\.md$"
    )


DEFAULT_PATH_POLICY = PathPolicy()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_relative_path(relative_path: str,
                            policy: PathPolicy = DEFAULT_PATH_POLICY
                            ) -> str | None:
    """Strict allowlist under the given policy.

    Returns the normalized path or None if rejected.
    """
    normalized = relative_path.replace("\\", "/")
    # Collapse repeated slashes so "memory//a.md" and "memory/a.md" are the
    # same store key (review finding 2026-09-28: two spellings of one path
    # would otherwise not match).
    normalized = re.sub(r"/{2,}", "/", normalized)
    if (not normalized or normalized.startswith("/")
            or any(seg == ".." for seg in normalized.split("/"))):
        return None
    if normalized in policy.root_filenames:
        return normalized
    if re.fullmatch(policy.user_file_pattern, normalized):
        return normalized
    if (normalized.startswith(policy.memory_prefix)
            and normalized.endswith(policy.markdown_suffix)):
        if any(normalized.startswith(p) for p in policy.excluded_prefixes):
            return None
        return normalized
    return None


def normalize_workspace_key(workspace_dir: str) -> str:
    resolved = os.path.abspath(workspace_dir)
    try:
        resolved = os.path.realpath(resolved)
    except OSError:
        pass
    return sha256_hex(resolved.replace("\\", "/"))


def _store_key(workspace_key: str, relative_path: str) -> str:
    return f"{workspace_key}:{sha256_hex(relative_path)}"


class ProvenanceStore:
    """JSON-backed provenance store (one file per workspace use)."""

    def __init__(self, path: str,
                 path_policy: PathPolicy = DEFAULT_PATH_POLICY):
        self.path = path
        self._policy = path_policy
        self._records: dict[str, dict] = {}
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as fh:
                self._records = json.load(fh)

    def _save(self) -> None:
        tmp = self.path + f".{os.getpid()}.tmp"  # PID-unique: not a fixed shared temp
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def _valid(self, record: dict | None, workspace_key: str,
               relative_path: str) -> dict | None:
        if not isinstance(record, dict):
            return None
        if (record.get("version") != PROVENANCE_VERSION
                or record.get("workspace_key") != workspace_key
                or record.get("relative_path") != relative_path
                or not re.fullmatch(r"[0-9a-f]{64}", record.get("file_hash", ""))
                or record.get("origin_class") not in ("agent", "untrusted")
                or not isinstance(record.get("observed_at"), int)
                or not record.get("reservation_id")):
            return None
        return record

    def record_write_provenance(self, *, workspace_dir: str, relative_path: str,
                                content_before: str, content_after: str,
                                origin_class: str, observed_at: int,
                                session_id: str | None = None,
                                session_key: str | None = None):
        """Record provenance for a memory write.

        Returns a rollback closure (compare-and-set on reservation_id), or
        None if the path is rejected by the allowlist.

        Retention law: 'agent' is kept only when a previous record exists
        with 'agent' origin AND its file_hash equals sha256(content_before).
        Otherwise — including every first write — the record is 'untrusted'.
        """
        normalized = normalize_relative_path(relative_path, self._policy)
        if normalized is None:
            return None
        workspace_key = normalize_workspace_key(workspace_dir)
        store_key = _store_key(workspace_key, normalized)
        previous = self._valid(self._records.get(store_key),
                               workspace_key, normalized)
        # CORRECTED bootstrap semantics: previous MUST exist. A first write
        # with no previous provenance degrades to 'untrusted' — it can never
        # retain 'agent', no matter what origin_class the caller claims.
        retained = (origin_class == "agent"
                    and previous is not None
                    and previous["origin_class"] == "agent"
                    and previous["file_hash"] == sha256_hex(content_before))
        reservation_id = uuid.uuid4().hex
        record = {
            "version": PROVENANCE_VERSION,
            "workspace_key": workspace_key,
            "relative_path": normalized,
            "file_hash": sha256_hex(content_after),
            "origin_class": "agent" if retained else "untrusted",
            "observed_at": observed_at,
            "reservation_id": reservation_id,
        }
        if session_id:
            record["session_id"] = session_id
        if session_key:
            record["session_key"] = session_key
        if len(self._records) >= MAX_ENTRIES and store_key not in self._records:
            raise OverflowError("provenance store full: rejecting new entries")
        self._records[store_key] = record
        self._save()

        previous_snapshot = previous

        def rollback() -> None:
            current = self._records.get(store_key)
            if current is not None and current.get("reservation_id") != reservation_id:
                return  # someone else wrote after us: do not clobber
            if previous_snapshot is not None:
                self._records[store_key] = previous_snapshot
            else:
                self._records.pop(store_key, None)
            self._save()

        return rollback

    def clear_provenance(self, *, workspace_dir: str, relative_path: str,
                         content_before: str) -> None:
        """Delete provenance only if the pre-write hash still matches."""
        normalized = normalize_relative_path(relative_path, self._policy)
        if normalized is None:
            return
        store_key = _store_key(normalize_workspace_key(workspace_dir), normalized)
        current = self._records.get(store_key)
        if current and current.get("file_hash") == sha256_hex(content_before):
            del self._records[store_key]
            self._save()

    def read_provenance(self, *, workspace_dir: str, relative_path: str) -> dict | None:
        normalized = normalize_relative_path(relative_path, self._policy)
        if normalized is None:
            return None
        workspace_key = normalize_workspace_key(workspace_dir)
        store_key = _store_key(workspace_key, normalized)
        stored = self._valid(self._records.get(store_key), workspace_key, normalized)
        if stored is None:
            return None
        return {k: v for k, v in stored.items()
                if k in ("file_hash", "origin_class", "observed_at",
                         "session_id", "session_key")}

    def list_provenance(self, *, workspace_dir: str) -> list[dict]:
        workspace_key = normalize_workspace_key(workspace_dir)
        prefix = f"{workspace_key}:"
        out = []
        for key, record in self._records.items():
            if not key.startswith(prefix):
                continue
            stored = self._valid(record, workspace_key, record.get("relative_path", ""))
            if stored:
                out.append({"relative_path": stored["relative_path"],
                            "provenance": self.read_provenance(
                                workspace_dir=workspace_dir,
                                relative_path=stored["relative_path"])})
        return out
