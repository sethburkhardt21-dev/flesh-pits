"""Batch consolidation cycle (MALICE M1, S-01).

Reads a snapshot of the experience store and emits a compacted summary
layer plus a promotion list. It is a *batch* operation only: it takes a
caller-supplied list of episode dicts and returns new structures. It
never runs inline on the store's write path, never mutates its input,
and never deletes or rewrites source episodes.

Pipeline (one cycle): decay -> merge -> prune.

- decay:  each entry's relevance decays multiplicatively
           (relevance *= decay_rate).
- merge:   greedy grouping. Entries are visited in deterministic order
           (relevance descending, id ascending). Each unmerged entry
           becomes a group seed; every later unmerged entry whose cosine
           similarity to the *seed* vector exceeds `merge_threshold`
           joins the group. The merged vector is the arithmetic mean of
           the member vectors; merged relevance is the sum of member
           relevances; non-vector fields are taken from the
           highest-relevance member (ties by id).
- prune:   merged entries whose relevance falls below `prune_threshold`
           are dropped from the *summary layer only*. Source episodes
           are untouched; this module has no delete path.

Summaries are deterministic given the same input: tie-breaking is by
id order, never by hash iteration order. Every summary starts
UNVERIFIED; an independent verifier flips it to VERIFIED via
`CycleResult.mark_verified`. The raw episodes a summary replaces are
the caller's to retain or release -- this module never deletes them.

Stdlib only. No clocks, no randomness, no network.
"""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "CycleConfig",
    "CycleRefused",
    "CycleResult",
    "ConsolidationCycle",
    "adapt_episode",
    "cosine_similarity",
]

#: Status of a summary before any independent verification has happened.
UNVERIFIED = "unverified"
#: Status after an independent verifier has attested the summary.
VERIFIED = "verified"


class CycleRefused(ValueError):
    """Bad configuration, bad entry shape, or a contract violation.
    Nothing is consolidated."""


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors in [-1, 1].

    Zero-length, mismatched-length, or zero-norm inputs score 0.0
    (orthogonal by convention) rather than raising: an entry with no
    usable vector is simply not similar to anything.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class CycleConfig:
    """Tuning for one consolidation cycle. Fail-closed validation."""

    #: Multiplicative relevance decay per cycle. Range (0, 1].
    decay_rate: float = 0.99
    #: Cosine similarity above which an entry joins a merge group. (0, 1].
    merge_threshold: float = 0.9
    #: Entries below this relevance leave the summary layer. [0, 1).
    prune_threshold: float = 0.1
    #: Relevance floor for freshly seeded entries. [0, 1].
    relevance_floor: float = 0.2
    #: Minimum episode count before a cycle does any work.
    min_episodes: int = 1
    #: Hard cap on summaries emitted per cycle; the rest are overflow.
    max_summaries: int = 256
    #: Promotion list length (top-k by relevance).
    promotion_k: int = 16

    def __post_init__(self) -> None:
        if not (0.0 < self.decay_rate <= 1.0):
            raise CycleRefused("decay_rate must be in (0, 1]")
        if not (0.0 < self.merge_threshold <= 1.0):
            raise CycleRefused("merge_threshold must be in (0, 1]")
        if not (0.0 <= self.prune_threshold < 1.0):
            raise CycleRefused("prune_threshold must be in [0, 1)")
        if not (0.0 <= self.relevance_floor <= 1.0):
            raise CycleRefused("relevance_floor must be in [0, 1]")
        if not isinstance(self.min_episodes, int) or self.min_episodes < 0:
            raise CycleRefused("min_episodes must be a non-negative int")
        if not isinstance(self.max_summaries, int) or self.max_summaries < 1:
            raise CycleRefused("max_summaries must be a positive int")
        if not isinstance(self.promotion_k, int) or self.promotion_k < 0:
            raise CycleRefused("promotion_k must be a non-negative int")


def _check_finite_number(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise CycleRefused(f"{what} must be a number, got {type(x).__name__}")
    if not math.isfinite(x):
        raise CycleRefused(f"{what} must be finite")
    return float(x)


def _validate_entry(entry: Mapping[str, Any], index: int,
                    seed_floor: float) -> dict[str, Any]:
    """Deep-copy and validate one input entry.

    Required: "id" (non-empty str), "vector" (list of finite numbers).
    Optional: "relevance" (finite number in [0, 1]); when absent the
    entry is seeded at `seed_floor`. Any other keys are carried through
    and preserved on merge.
    """
    if not isinstance(entry, Mapping):
        raise CycleRefused(f"entry[{index}] must be a mapping")
    eid = entry.get("id")
    if not isinstance(eid, str) or not eid:
        raise CycleRefused(f"entry[{index}] needs a non-empty str 'id'")
    vector = entry.get("vector")
    if not isinstance(vector, (list, tuple)) or not vector:
        raise CycleRefused(f"entry[{eid!r}] needs a non-empty 'vector'")
    vector = [_check_finite_number(x, f"entry[{eid!r}].vector[{i}]")
              for i, x in enumerate(vector)]
    raw_rel = entry.get("relevance", seed_floor)
    relevance = _check_finite_number(raw_rel, f"entry[{eid!r}].relevance")
    if not 0.0 <= relevance <= 1.0:
        raise CycleRefused(f"entry[{eid!r}].relevance must be in [0, 1]")
    out = copy.deepcopy(dict(entry))
    out["id"] = eid
    out["vector"] = vector
    out["relevance"] = max(relevance, 0.0)
    return out


def adapt_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt an experience-store episode record to a cycle entry.

    Episode shape (per the experience-store contract): "episode_id",
    "context_vector", "salience", plus free-form payload dicts that are
    carried through untouched. Returns a fresh entry dict.
    """
    if not isinstance(episode, Mapping):
        raise CycleRefused("episode must be a mapping")
    eid = episode.get("episode_id")
    if not isinstance(eid, str) or not eid:
        raise CycleRefused("episode needs a non-empty str 'episode_id'")
    vector = episode.get("context_vector")
    if not isinstance(vector, (list, tuple)) or not vector:
        raise CycleRefused(f"episode[{eid!r}] needs a non-empty "
                           f"'context_vector'")
    vector = [_check_finite_number(x, f"episode[{eid!r}].context_vector[{i}]")
              for i, x in enumerate(vector)]
    salience = episode.get("salience", 1.0)
    salience = _check_finite_number(salience, f"episode[{eid!r}].salience")
    if not 0.0 <= salience <= 1.0:
        raise CycleRefused(f"episode[{eid!r}].salience must be in [0, 1]")
    entry = copy.deepcopy(dict(episode))
    entry["id"] = eid
    entry["vector"] = vector
    entry["relevance"] = salience
    return entry


def _summary_id(member_ids: Sequence[str]) -> str:
    """Deterministic summary id from the sorted member id set."""
    digest = hashlib.sha256(
        "\x00".join(sorted(member_ids)).encode("utf-8")).hexdigest()
    return "sum_" + digest[:16]


@dataclass
class CycleResult:
    """Output of one batch cycle. All lists are caller-owned."""

    #: One dict per surviving group, each UNVERIFIED until an
    #: independent verifier calls mark_verified().
    summaries: list[dict[str, Any]] = field(default_factory=list)
    #: Summaries beyond max_summaries (lowest relevance first). They are
    #: reported, never silently dropped; the raw episodes remain.
    overflow: list[dict[str, Any]] = field(default_factory=list)
    #: Top-k entry ids by relevance (tie: member-id order). Replay/
    #: promotion candidates for the next cadence.
    promotion: list[str] = field(default_factory=list)
    #: The caller-supplied cycle tick this result was computed for.
    tick: int = 0
    #: True when the cycle actually ran (False: below min_episodes).
    ran: bool = True

    def mark_verified(self, summary_id: str, evidence: str) -> None:
        """An *independent* verifier attests one summary.

        Only UNVERIFIED summaries can be flipped; the evidence note is
        stored on the summary. Raw episodes stay retained by the caller
        until this has happened -- the cycle itself never releases them.
        """
        if not isinstance(summary_id, str) or not summary_id:
            raise CycleRefused("summary_id must be a non-empty str")
        if not isinstance(evidence, str) or not evidence:
            raise CycleRefused("evidence must be a non-empty str")
        for summary in self.summaries:
            if summary.get("summary_id") == summary_id:
                if summary.get("status") == VERIFIED:
                    raise CycleRefused(
                        f"summary {summary_id!r} is already verified")
                summary["status"] = VERIFIED
                summary["verification_evidence"] = evidence
                return
        raise CycleRefused(f"unknown summary_id {summary_id!r}")

    def unverified(self) -> list[dict[str, Any]]:
        """Summaries still awaiting independent verification."""
        return [s for s in self.summaries if s.get("status") == UNVERIFIED]


class ConsolidationCycle:
    """Batch consolidation over an experience-store snapshot.

    Usage: pass a list of episode dicts (or `adapt_episode` outputs)
    plus the current cycle tick. The cycle returns a CycleResult; the
    input list is never mutated and no source episode is ever deleted.

    This object deliberately holds no reference to any store and
    registers no callbacks: it cannot be wired into a hot path except
    by a scheduler explicitly calling run() with a snapshot.
    """

    def __init__(self, config: CycleConfig | None = None) -> None:
        self.config = config or CycleConfig()
        self.cycles_run = 0

    # -- one cycle --------------------------------------------------------

    def run(self, entries: Sequence[Mapping[str, Any]], *,
            tick: int) -> CycleResult:
        """Run one batch cycle over a snapshot of entries.

        `tick` is caller-supplied (an injected clock in tests); it is
        stamped on the result so a scheduler can order cycles. The
        input sequence is deep-copied up front: callers keep full
        ownership of their episodes.
        """
        if not isinstance(tick, int) or tick < 0:
            raise CycleRefused("tick must be a non-negative int")
        staged = [_validate_entry(e, i, self.config.relevance_floor)
                  for i, e in enumerate(entries)]
        # Deterministic visit order: relevance desc, id asc. Never hash
        # order: dict/set iteration never decides a tie.
        staged.sort(key=lambda e: (-e["relevance"], e["id"]))

        if len(staged) < self.config.min_episodes:
            return CycleResult(tick=tick, ran=False)

        cfg = self.config
        # 1. decay (multiplicative, in place on the *copies* only)
        for e in staged:
            e["relevance"] = e["relevance"] * cfg.decay_rate

        # 2. greedy merge against the group seed
        merged_flags = [False] * len(staged)
        groups: list[list[int]] = []
        for i, seed in enumerate(staged):
            if merged_flags[i]:
                continue
            group = [i]
            merged_flags[i] = True
            for j in range(i + 1, len(staged)):
                if merged_flags[j]:
                    continue
                if cosine_similarity(seed["vector"],
                                     staged[j]["vector"]) > cfg.merge_threshold:
                    group.append(j)
                    merged_flags[j] = True
            groups.append(group)

        summaries: list[dict[str, Any]] = []
        for group in groups:
            members = [staged[i] for i in group]
            dim = len(members[0]["vector"])
            mean = [sum(m["vector"][d] for m in members) / len(members)
                    for d in range(dim)]
            total_relevance = sum(m["relevance"] for m in members)
            best = min(members,
                       key=lambda m: (-m["relevance"], m["id"]))
            summary = {
                "summary_id": _summary_id([m["id"] for m in members]),
                "member_ids": sorted(m["id"] for m in members),
                "merged_count": len(members),
                "vector": mean,
                "relevance": total_relevance,
                "status": UNVERIFIED,
                "tick": tick,
            }
            for key, value in best.items():
                if key not in ("id", "vector", "relevance"):
                    summary.setdefault(key, value)
            summaries.append(summary)

        # 3. prune (summary layer only; source episodes untouched)
        summaries = [s for s in summaries
                     if s["relevance"] >= cfg.prune_threshold]

        # promotion: top-k by relevance over surviving summaries,
        # tie-break by member-id order for determinism.
        ordered = sorted(
            summaries,
            key=lambda s: (-s["relevance"], s["member_ids"]))
        promotion = [s["summary_id"]
                     for s in ordered[:cfg.promotion_k]] if cfg.promotion_k else []

        # budget: cap the summary layer; overflow is reported, not lost.
        kept = ordered[:cfg.max_summaries]
        overflow = ordered[cfg.max_summaries:]

        self.cycles_run += 1
        return CycleResult(summaries=kept, overflow=overflow,
                           promotion=promotion, tick=tick, ran=True)

    # -- convenience: store-shaped input ----------------------------------

    def run_over_episodes(self, episodes: Sequence[Mapping[str, Any]], *,
                          tick: int) -> CycleResult:
        """Run one cycle over raw experience-store episode dicts."""
        return self.run([adapt_episode(e) for e in episodes], tick=tick)
