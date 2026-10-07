"""Memory consolidation cycle: decay -> merge -> prune -> replay.

Clean-room port of the consolidation mechanism in
``MemoryConsolidationManager`` (donor: models/memory/optimized_store.py).
Stdlib only. Vectors are plain sequences of floats; entries are dicts.

Deliberately NOT ported: the legacy merge variant (a bug-reproduction
ablation that drops fields), the emotional/temporal hierarchical indices
(product coupling), and the OptimizedMemoryStore product shell.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; zero vectors score 0.0 (donor-faithful)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


class ConsolidationCycle:
    """Relevance-decayed memory consolidation with dedup merging.

    Each entry is a dict with at least ``"id"`` and ``"vector"``
    (a sequence of floats); ``"relevance"`` defaults to 1.0 and any
    extra fields are preserved verbatim (and carried over from the
    highest-relevance member when entries merge).
    """

    def __init__(
        self,
        decay_rate: float = 0.99,
        merge_threshold: float = 0.9,
        prune_threshold: float = 0.1,
        consolidation_threshold: int = 100,
    ) -> None:
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError("decay_rate must be in (0, 1]")
        if not -1.0 <= merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be in [-1, 1]")
        if prune_threshold < 0.0:
            raise ValueError("prune_threshold must be >= 0")
        if consolidation_threshold < 1:
            raise ValueError("consolidation_threshold must be >= 1")
        self.decay_rate = decay_rate
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.consolidation_threshold = consolidation_threshold
        self._consolidation_count = 0

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count

    def check_consolidation(self, entries: List[Dict[str, Any]] | None) -> bool:
        """True when the entry count reaches the consolidation threshold."""
        if entries is None:
            return False
        return len(entries) >= self.consolidation_threshold

    def seed_relevance(self, entry: Dict[str, Any], priority: float) -> None:
        """Seed relevance as max(0.2, priority) (donor-faithful floor)."""
        entry["relevance"] = max(0.2, priority)

    def increment_relevance(self, entry: Dict[str, Any], amount: float = 0.1) -> None:
        """Increment relevance on retrieval. (Present in the donor but
        unwired there — grep-verified dead code; kept here as live API.)"""
        entry["relevance"] = entry.get("relevance", 1.0) + amount

    def consolidate(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One full cycle: decay relevance, merge near-duplicates, prune.

        Merge is greedy O(n^2) on cosine similarity > merge_threshold:
        merged vector = mean of member vectors, relevance = sum of member
        relevances, all other fields taken from the highest-relevance
        member, plus ``"merged_count"``.
        """
        if not entries:
            return entries

        # 1. Decay relevance.
        for entry in entries:
            entry.setdefault("relevance", 1.0)
            entry["relevance"] *= self.decay_rate

        # 2. Merge near-duplicates.
        entries = self._merge_similar(entries)

        # 3. Prune low-relevance entries.
        entries = [e for e in entries if e.get("relevance", 0.0) >= self.prune_threshold]

        self._consolidation_count += 1
        return entries

    def _merge_similar(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(entries) < 2:
            return entries
        merged = [False] * len(entries)
        result: List[Dict[str, Any]] = []
        for i in range(len(entries)):
            if merged[i]:
                continue
            group = [i]
            for j in range(i + 1, len(entries)):
                if merged[j]:
                    continue
                if _cosine(entries[i]["vector"], entries[j]["vector"]) > self.merge_threshold:
                    group.append(j)
                    merged[j] = True
            if len(group) == 1:
                result.append(entries[i])
            else:
                dim = len(entries[group[0]]["vector"])
                avg = [
                    sum(entries[g]["vector"][d] for g in group) / len(group)
                    for d in range(dim)
                ]
                total = sum(entries[g].get("relevance", 1.0) for g in group)
                best = max(group, key=lambda g: entries[g].get("relevance", 1.0))
                merged_entry = dict(entries[best])
                merged_entry["vector"] = avg
                merged_entry["relevance"] = total
                merged_entry["merged_count"] = len(group)
                result.append(merged_entry)
        return result

    def get_replay_batch(
        self, entries: List[Dict[str, Any]], k: int = 16
    ) -> List[Dict[str, Any]]:
        """Top-k entries by relevance, for experience replay."""
        if not entries or k <= 0:
            return []
        return sorted(entries, key=lambda e: e.get("relevance", 0.0), reverse=True)[:k]
