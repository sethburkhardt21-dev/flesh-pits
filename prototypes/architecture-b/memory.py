"""Episodic memory substrate — Architecture B.

Clean-room port of the experience_store ALGORITHM (donor: the_consciousness_ai
MemoryCore via sandbox/being/memory_port/experience_store.py — read-only
reference, not modified):

  store: every experience appended to the recent list with provenance;
         the vector index upserts ONLY when attention >= threshold (0.7,
         donor-exact).
  retrieve: cosine similarity against each experience's ``state`` field ONLY
            (never the concatenated vector); dimension mismatch scores 0.0
            rather than padding/truncating; search pool bounded to the most
            recent max(k*10, 100).

Provenance per record: id, tick, episode, env name, obs vector, action,
reward, error norms, precision snapshot, attention, narrative-free
(mechanism only — no language content).

The world model queries this store for:
  (a) retrieval correction of the next-obs prediction (mean stored residual
      of the k nearest neighbours, similarity-weighted), and
  (b) predicted per-action error magnitudes for active inference.

Deterministic. Stdlib only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class EpisodicStore:
    """Attention-gated episodic store with provenance and honest retrieval."""

    SEARCH_POOL_MIN = 100
    ATTENTION_THRESHOLD = 0.7  # donor-exact

    def __init__(self, attention_threshold: float = ATTENTION_THRESHOLD) -> None:
        if not 0.0 <= attention_threshold <= 1.0:
            raise ValueError("attention_threshold must be in [0, 1]")
        self.attention_threshold = attention_threshold
        self.recent: List[Dict[str, Any]] = []
        self._indexed = 0

    @property
    def indexed_count(self) -> int:
        return self._indexed

    def __len__(self) -> int:
        return len(self.recent)

    # -- encoding ---------------------------------------------------------
    def store(self, obs_vec: List[float], action: int, reward: float,
              e0: List[float], e1: List[float], precision: List[float],
              attention: float, tick: int, episode: int,
              env_name: str) -> str:
        """Encode one experience with full provenance.

        Always appended to the recent list; the retrieval index upserts
        only when attention >= threshold. Returns the memory id when
        indexed, else "".
        """
        obs_vec = [float(x) for x in obs_vec]
        mem_id = f"mem_{len(self.recent)}"
        self.recent.append({
            "id": mem_id,
            "tick": int(tick),
            "episode": int(episode),
            "env": str(env_name),
            "state": obs_vec,                       # query lives in state space
            "action": int(action),
            "reward": float(reward),
            "e0": [float(x) for x in e0],            # stored sensory residual
            "e1": [float(x) for x in e1],            # stored L1 residual
            "e0_norm": math.sqrt(sum(x * x for x in e0)),
            "e1_norm": math.sqrt(sum(x * x for x in e1)),
            "precision": [float(x) for x in precision],
            "attention": float(attention),
        })
        if attention >= self.attention_threshold:
            self._indexed += 1
            return mem_id
        return ""

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, query: List[float], k: int = 5,
                 action: Optional[int] = None) -> List[Dict[str, Any]]:
        """k most similar experiences by cosine over ``state`` only.

        If action is given, only experiences with that action are eligible
        (used for per-action error prediction in active inference).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self.recent:
            return []
        query = [float(x) for x in query]
        pool_size = max(k * 10, self.SEARCH_POOL_MIN)
        pool = self.recent[-pool_size:]
        scored = []
        for exp in pool:
            if action is not None and exp["action"] != action:
                continue
            stored = exp["state"]
            if len(stored) != len(query):
                scored.append((0.0, exp))
                continue
            scored.append((_cosine(query, stored), exp))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{"id": exp["id"], "score": score, "exp": exp}
                for score, exp in scored[:k]]

    def retrieval_correction(self, query: List[float], k: int = 5) -> Dict:
        """Similarity-weighted mean stored e0 residual of k neighbours.

        Used to correct the world model's next-obs prediction. Returns the
        correction vector, mean similarity, and neighbour count. Zero
        correction when the store is empty (honest, not invented).
        """
        nbrs = self.retrieve(query, k=k)
        if not nbrs:
            return {"correction": None, "mean_similarity": 0.0, "n": 0}
        dim = len(nbrs[0]["exp"]["e0"])
        corr = [0.0] * dim
        wsum = 0.0
        for n in nbrs:
            w = max(0.0, n["score"])
            wsum += w
            for c in range(dim):
                corr[c] += w * n["exp"]["e0"][c]
        if wsum > 0:
            corr = [c / wsum for c in corr]
        mean_sim = sum(n["score"] for n in nbrs) / len(nbrs)
        return {"correction": corr, "mean_similarity": mean_sim,
                "n": len(nbrs)}

    def predicted_error_for_action(self, query: List[float],
                                   action: int, k: int = 5) -> Optional[float]:
        """Mean stored e1_norm of k similar experiences with this action.

        Data-driven expected-error estimate for active inference. None when
        no eligible neighbours exist (caller falls back to the running table).
        """
        nbrs = self.retrieve(query, k=k, action=action)
        if not nbrs:
            return None
        return sum(n["exp"]["e1_norm"] for n in nbrs) / len(nbrs)

    # -- persistence --------------------------------------------------------
    def snapshot(self) -> Dict:
        return {
            "attention_threshold": self.attention_threshold,
            "indexed": self._indexed,
            "recent": self.recent,
        }

    def restore(self, state: Dict) -> None:
        self.attention_threshold = state["attention_threshold"]
        self._indexed = state["indexed"]
        self.recent = state["recent"]
