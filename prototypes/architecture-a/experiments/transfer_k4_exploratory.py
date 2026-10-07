"""EXPLORATORY (post-verdict) phase-split + warm-start analyses for TRANSFER-K4-RESTORE.

Not part of the preregistered gates. Appends labeled exploratory records to
the receipt chain.
"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "experiments", "envs"))

import transfer_k4_restore as t  # noqa: E402
from tick import WorkspaceTick  # noqa: E402
from env_interface import derive_seed  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402

SEEDS = (11, 22, 33, 44)
RECEIPT_PATH = os.path.join(HERE, "..", "receipts", "transfer",
                            "TRANSFER-K4-RESTORE.ndjson")


def run_phase_split(seed, frozen, preload_gains=None, unfreeze_after_load=False):
    """Returns (phase0_total, phase1_total, final_gains)."""
    wk = WorkspaceTick(t.CHANNELS, t.cue_indexed_specialists(t.CHANNELS),
                       capacity=3, frozen_gains=frozen, gain_lr=t.GAIN_LR,
                       ignition_kwargs={"theta": t.THETA})
    if preload_gains is not None:
        for c, g in preload_gains.items():
            wk.arbitrator.gains[c] = float(g)
        if not unfreeze_after_load:
            wk.arbitrator.freeze()
    env = ChangingRule()
    p0 = p1 = 0.0
    for run_index in range(t.EPISODES):
        obs = env.reset(derive_seed(seed, run_index, "env"))
        chan_env = t.ChannelEnv(env)
        for _ in range(t.STEPS):
            wk.step({"cue": int(obs["cue"]),
                     "last_reward": float(obs["last_reward"]),
                     "last_action": int(obs["last_action"])}, chan_env)
            r = wk.rewards[-1]
            if run_index < 5:
                p0 += r
            else:
                p1 += r
            obs = chan_env.last_obs2
            if chan_env.last_done:
                break
    return p0, p1, dict(wk.arbitrator.gains)


def receipt_hash(rec):
    body = {k: v for k, v in rec.items() if k != "receipt_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def main():
    with open(os.path.join(HERE, "..", "receipts", "transfer",
                           "transferred_gains.json")) as f:
        saved = json.load(f)["permitted_state_per_seed"]
    t_wall = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(RECEIPT_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]

    def emit(kind, payload):
        rec = {"experiment_id": "TRANSFER-K4-RESTORE", "kind": kind,
               "t_wall": t_wall}
        rec.update(payload)
        rec["prev_hash"] = records[-1]["receipt_hash"]
        rec["receipt_hash"] = receipt_hash(rec)
        records.append(rec)
        return rec

    rows = []
    for seed in SEEDS:
        g = saved[str(seed)]["gains"]
        p0_T, p1_T, _ = run_phase_split(seed, True, preload_gains=g)
        p0_F, p1_F, _ = run_phase_split(seed, True)
        p0_W, p1_W, gW = run_phase_split(seed, True, preload_gains=g,
                                        unfreeze_after_load=True)
        rows.append({"seed": seed,
                     "transfer_phase0": p0_T, "transfer_phase1": p1_T,
                     "frozen_phase0": p0_F, "frozen_phase1": p1_F,
                     "warmstart_phase0": p0_W, "warmstart_phase1": p1_W,
                     "warmstart_final_gains": {k: round(v, 3)
                                               for k, v in gW.items()}})
    emit("exploratory_phase_split", {
        "label": "EXPLORATORY — not a preregistered gate",
        "question": ("Does the frozen transferred gain vector help in phase 1 "
                     "(post-flip, the world it was trained on) and hurt in "
                     "phase 0 (pre-flip)?"),
        "rows": rows,
        "warmstart_note": ("warmstart = transferred gains loaded UNFROZEN: "
                           "tests transfer-as-initialization (relearning "
                           "speed) vs frozen transfer-as-policy."),
    })
    with open(RECEIPT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    for r in rows:
        s = r["seed"]
        print(f"seed {s}: phase0 T={r['transfer_phase0']:.0f} F={r['frozen_phase0']:.0f} "
              f"W={r['warmstart_phase0']:.0f} | phase1 T={r['transfer_phase1']:.0f} "
              f"F={r['frozen_phase1']:.0f} W={r['warmstart_phase1']:.0f}")


if __name__ == "__main__":
    main()
