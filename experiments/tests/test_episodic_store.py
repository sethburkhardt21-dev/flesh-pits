"""Unit tests for the quarantined episodic store (P-B03 invariant).

Run: python3 -m pytest tests/test_episodic_store.py -v   (from experiments/)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from episodic_store import EpisodicStore, StoreQuarantineError  # noqa: E402


def obs_rec(**kw):
    d = {"source": "observed", "episode": 0, "tick": 0,
         "observation": {"hand": 0.5}}
    d.update(kw)
    return d


def test_admit_observed_ok():
    s = EpisodicStore()
    h = s.admit(obs_rec())
    assert len(s) == 1 and s.admissions == 1
    assert s.scan() == []
    assert s.verify_chain()
    assert h == s.to_snapshot()["chain_head"]


def test_reject_predicted_simulated_missing():
    s = EpisodicStore()
    for bad in ({"source": "predicted"}, {"source": "simulated"},
                {"episode": 1}, {"source": None}, {"source": ""}, "x"):
        with pytest.raises(StoreQuarantineError):
            s.admit(bad)
    assert len(s) == 0 and s.rejections == 6
    assert s.verify_chain()


def test_chain_tamper_detected():
    s = EpisodicStore()
    s.admit(obs_rec(tick=0))
    s.admit(obs_rec(tick=1))
    s._records[0]["observation"] = {"hand": 0.999}
    assert not s.verify_chain()
    assert s.scan() == []  # tag audit still clean; chain audit catches it


def test_restore_rejects_predicted_snapshot():
    s = EpisodicStore()
    s.admit(obs_rec(tick=0))
    snap = s.to_snapshot()
    tampered = {"records": [dict(r) for r in snap["records"]],
                "chain_head": snap["chain_head"]}
    tampered["records"][0]["source"] = "predicted"
    fresh = EpisodicStore()
    with pytest.raises(StoreQuarantineError):
        fresh.restore(tampered)
    assert len(fresh) == 0


def test_restore_rejects_broken_chain():
    s = EpisodicStore()
    s.admit(obs_rec(tick=0))
    snap = s.to_snapshot()
    broken = {"records": [dict(r) for r in snap["records"]],
              "chain_head": snap["chain_head"]}
    broken["records"][0]["observation"] = {"hand": 0.1}
    fresh = EpisodicStore()
    with pytest.raises(StoreQuarantineError):
        fresh.restore(broken)
    assert len(fresh) == 0


def test_restore_honest_snapshot_roundtrip():
    s = EpisodicStore()
    for t in range(5):
        s.admit(obs_rec(tick=t))
    snap = s.to_snapshot()
    fresh = EpisodicStore()
    fresh.restore(snap)
    assert len(fresh) == 5
    assert fresh.verify_chain()
    assert fresh.scan() == []
