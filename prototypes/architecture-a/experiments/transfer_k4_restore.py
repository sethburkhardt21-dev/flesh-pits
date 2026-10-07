"""TRANSFER-K4-RESTORE — directive §11 model-independence decomposition, honest variant.

ARCHITECTURAL FINDING (verified by own hand, documented in preregistration):
The K4 learned-gains loop (tick.py / attention.py / consumers.py / envs.py)
contains NO foundation model. Specialists are pure functions
(envs.make_specialists: stimulus = float(obs)); arbitrate()/update_gains()
touch only floats; feedback utility is {action: reward}. None of the ladder's
contamination dimensions (model_id, revision_sha, quant_file, sampling_params,
chat_template) exist in the loop. Therefore the task's literal arm —
"train with model A (0.5B), transfer to model B (1.5B), measure K4 win" —
is VACUOUS BY CONSTRUCTION: loading two models and never calling them, then
reporting "behavior survives replacement," would be pretend machinery,
forbidden by the 2026-10-07 owner law (nothing ever fake, only real) and the
Estate Formation Charter (no simulated evidence). It is NOT run.

HONEST VARIANT (this script): the directive §11 question decomposed for a
model-free mechanism —
  learned behavior = f(persisted gains) + g(transient estimator state)
                   + h(model weights = 0, by construction)
Transfer ONLY the permitted persistent state (gains JSON) across a loop
restart into a fresh loop (fresh z-score estimators, fresh ignition strength,
fresh reward EMA, fresh RNG-derived envs), freeze the transferred gains
(no further learning), and measure whether the K4 win (R >= 1.30 vs
frozen-at-1.0 baseline) survives. This separates:
  - external persistent cognition (the gains file) .......... TRANSFERRED
  - transient learned state (z-score mean/var, EMA, strength)  EXCLUDED
  - model weights ........................................... ABSENT (none)
  - prompt artifacts ......................................... ABSENT (none)

Protocol mirrors the proven K4-CANONICAL-RERUN receipt:
  canonical changing_rule env (ENV_INTERFACE v1.0.0), cue-indexed adapter
  (channels c{cue}a{action}), gain_lr=0.15, theta=0.45, capacity=3,
  10 episodes x 40 steps = 400 ticks/seed, seeds {11,22,33,44}.
Cue-indexed adapter reconstructed from the receipt config (the test-battery
worker's script was not preserved in the tree): specialist stimulus is 1.0
when the observed cue matches the channel's cue, else 0.0; the channel's
action index is stepped in the env. Train-phase R_learned >= 1.30 on >=3/4
seeds is the adapter-fidelity reproduction gate (canonical rerun got
1.44/1.60/1.70/1.48).

Conditions per seed:
  LEARNED   fresh loop, gains learn online (frozen=False)   -> total L
  TRANSFER  fresh loop, gains := persisted G, then freeze() -> total T
  FROZEN    fresh loop, gains pinned at 1.0 (frozen=True)    -> total F
Metrics: R_learned = L/F, R_transfer = T/F.

Preregistered gates:
  G1 reproduction: R_learned >= 1.30 on >= 3/4 seeds. If FAILS, adapter
     reconstruction is unfaithful and the transfer verdict is
     UNINTERPRETABLE (report BLOCKED, no transfer claims).
  G2 transfer:     R_transfer >= 1.30 on >= 3/4 seeds -> H1: learned
     behavior survives in external persistent gains alone.
     Null H0: R_transfer < 1.30 on >= 2/4 seeds.
  G3 flag:         R_transfer >= R_learned on >= 3/4 seeds -> endpoint gains
     suffice; online learning adds nothing beyond its endpoint (report,
     do not celebrate).
Determinism check: FROZEN run twice per seed; totals must be bit-identical.
"""
import copy
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                       # architecture-a
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "experiments", "envs"))

from tick import WorkspaceTick                                   # noqa: E402
from env_interface import CONTRACT_VERSION, derive_seed, config_hash  # noqa: E402
from changing_rule import ChangingRule                           # noqa: E402

SEEDS = (11, 22, 33, 44)
EPISODES = 10
STEPS = 40          # 400 ticks per seed, matches canonical rerun
GAIN_LR = 0.15
THETA = 0.45
MARGIN = 1.30
NEED = 3
CHANNELS = ["c0a0", "c0a1", "c1a0", "c1a1"]

RECEIPT_PATH = os.path.join(HERE, "..", "receipts", "transfer",
                            "TRANSFER-K4-RESTORE.ndjson")
STATE_PATH = os.path.join(HERE, "..", "receipts", "transfer",
                          "transferred_gains.json")


def cue_indexed_specialists(channels):
    specs = []
    for ch in channels:
        cue = int(ch[1])

        def fn(obs, tick, _cue=cue, _ch=ch):
            stim = 1.0 if int(obs.get("cue", -1)) == _cue else 0.0
            return stim, {"summary": f"cue-indexed:{_ch}", "signal": stim}

        specs.append((ch, fn))
    return specs


class ChannelEnv:
    """Adapter: tick actions are channel names; env takes int actions.

    Captures the post-step observation so the driver loop does not
    double-step the environment.
    """

    def __init__(self, env):
        self._env = env
        self.last_obs2 = None
        self.last_done = False

    def step(self, channel):
        action = int(channel[3])  # "c0a1" -> 1
        obs2, reward, done, _info = self._env.step(action)
        self.last_obs2 = obs2
        self.last_done = bool(done)
        return float(reward)


def run_condition(seed, frozen, preload_gains=None):
    """One 400-tick run. Returns (total_reward, final_gains, n_gain_updates)."""
    wk = WorkspaceTick(CHANNELS, cue_indexed_specialists(CHANNELS),
                       capacity=3, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA})
    if preload_gains is not None:
        for c, g in preload_gains.items():
            wk.arbitrator.gains[c] = float(g)
        wk.arbitrator.freeze()
    env = ChangingRule()
    total = 0.0
    for run_index in range(EPISODES):
        obs = env.reset(derive_seed(seed, run_index, "env"))
        chan_env = ChannelEnv(env)
        for _ in range(STEPS):
            wk.step({"cue": int(obs["cue"]),
                     "last_reward": float(obs["last_reward"]),
                     "last_action": int(obs["last_action"])}, chan_env)
            total += wk.rewards[-1]
            obs = chan_env.last_obs2  # set by the tick's env.step
            if chan_env.last_done:
                break
    return total, dict(wk.arbitrator.gains), len(wk.arbitrator.gain_history)


def chain_write(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def receipt_hash(rec):
    body = {k: v for k, v in rec.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def main():
    t_wall = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []

    def emit(kind, payload):
        rec = {"experiment_id": "TRANSFER-K4-RESTORE", "kind": kind,
               "t_wall": t_wall}
        rec.update(payload)
        rec["prev_hash"] = records[-1]["receipt_hash"] if records else "GENESIS"
        rec["receipt_hash"] = receipt_hash(rec)
        records.append(rec)
        return rec

    # ---- run header per runtime-ladder law ---------------------------
    run_header = {
        "model_id": None,
        "revision_sha": None,
        "quant_file": None,
        "quant_file_sha256": None,
        "chat_template_hash": None,
        "sampling_params": None,
        "runtime_adapter": "SymbolicBaseline",
        "runtime_adapter_version": "flesh-pits architecture-a tick.py (stdlib only)",
        "n_ctx": None,
        "seed": list(SEEDS),
        "backend_rung": 5,
        "backend_note": ("No foundation model participates in the K4 loop "
                         "(verified by code inspection: tick.py, attention.py, "
                         "consumers.py, envs.py). The model-A->model-B arm is "
                         "inapplicable by construction; this is the honest "
                         "transfer-across-restart variant."),
    }
    config = {"adapter": "cue-indexed (reconstructed)", "channels": CHANNELS,
              "gain_lr": GAIN_LR, "theta": THETA, "capacity": 3,
              "episodes_per_seed": EPISODES, "steps_per_episode": STEPS,
              "total_ticks_per_seed": EPISODES * STEPS,
              "env": "changing_rule", "env_version": "1.0.0",
              "contract_version": CONTRACT_VERSION,
              "note": ("Cue-indexed adapter reconstructed from "
                       "K4-CANONICAL-RERUN.ndjson config; original worker "
                       "script not preserved in tree.")}
    emit("preregistration", {
        "run_header": run_header,
        "hypothesis": ("Transferred gains alone (frozen at persisted values, "
                       "all other state fresh) achieve R_transfer >= 1.30 vs "
                       "frozen-at-1.0 on >= 3/4 seeds: learned behavior "
                       "survives in external persistent state."),
        "null_hypothesis": ("R_transfer < 1.30 on >= 2/4 seeds: the K4 win "
                            "depends on transient estimator state or the "
                            "online trajectory, not on transferable gains."),
        "preregistered_metric": "total_reward_ratio_transferred_over_frozen",
        "decision_rule": ("PASS transfer iff R_transfer >= 1.30 on >= 3 of 4 "
                          "seeds {11,22,33,44}."),
        "seeds": list(SEEDS),
        "config": config,
        "config_hash": config_hash(config),
        "gates": {
            "G1_reproduction": "R_learned >= 1.30 on >= 3/4 seeds (adapter fidelity)",
            "G2_transfer": "R_transfer >= 1.30 on >= 3/4 seeds -> H1",
            "G3_flag": "R_transfer >= R_learned on >= 3/4 seeds -> endpoint sufficiency",
        },
        "tolerance": 1e-09,
        "metric_direction": "higher_better",
        "permitted_persistent_state": {
            "transferred": ["gains {channel: float}", "gain_lr", "gain_cap",
                            "theta", "channels", "trained_on", "seed"],
            "excluded": ["RunningZScoreBid running mean/var per channel",
                         "tick._reward_ema", "ignition.strength",
                         "buffer contents", "env RNG state",
                         "gain_history (audit only, not loaded)"],
        },
        "vacuous_arm_rejected": {
            "arm": "train gains with Qwen2.5-0.5B, transfer to Qwen2.5-1.5B",
            "reason": ("K4 loop has no model contamination dimension; loading "
                       "models without calling them and reporting 'survives' "
                       "would be pretend machinery. Rejected per 2026-10-07 "
                       "owner law (nothing ever fake, only real)."),
        },
    })

    per_seed = []
    for seed in SEEDS:
        total_L, gains_L, n_updates = run_condition(seed, frozen=False)
        total_T, _gains_T, _ = run_condition(seed, frozen=True,
                                            preload_gains=gains_L)
        total_F, _gF, _ = run_condition(seed, frozen=True)
        total_F2, _gF2, _ = run_condition(seed, frozen=True)
        assert total_F == total_F2, f"nondeterminism at seed {seed}"
        r_learned = total_L / total_F if total_F else float("nan")
        r_transfer = total_T / total_F if total_F else float("nan")
        per_seed.append({"seed": seed, "learned_total": total_L,
                         "transferred_total": total_T, "frozen_total": total_F,
                         "learned_gains": gains_L,
                         "n_gain_updates": n_updates,
                         "r_learned": r_learned, "r_transfer": r_transfer,
                         "transfer_win": r_transfer >= MARGIN,
                         "determinism_check": total_F == total_F2})
        emit("seed_result", {
            "seed": seed,
            "learned_total": total_L, "transferred_total": total_T,
            "frozen_total": total_F,
            "learned_gains": gains_L,
            "transferred_gains_loaded": gains_L,
            "r_learned": round(r_learned, 4),
            "r_transfer": round(r_transfer, 4),
            "win": r_transfer >= MARGIN,
            "learned_gain_updates": n_updates,
            "determinism_frozen_bitidentical": True,
        })

    n_g1 = sum(1 for s in per_seed if s["r_learned"] >= MARGIN)
    n_g2 = sum(1 for s in per_seed if s["transfer_win"])
    n_g3 = sum(1 for s in per_seed if s["r_transfer"] >= s["r_learned"])
    g1 = n_g1 >= NEED
    g2 = n_g2 >= NEED if g1 else None

    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"experiment_id": "TRANSFER-K4-RESTORE",
                   "permitted_state_per_seed":
                   {str(s["seed"]): payload_for(s) for s in per_seed}},
                  f, indent=1, sort_keys=True)

    verdict = ("UNINTERPRETABLE (G1 reproduction failed — adapter "
               "reconstruction unfaithful)") if not g1 else (
               "H1 SUPPORTED — learned behavior survives transfer in "
               "external persistent gains alone" if g2 else
               "H0 — transfer fails: K4 win needs transient state/trajectory")
    emit("verdict", {
        "G1_reproduction_pass": g1, "G1_seeds": n_g1,
        "G2_transfer_pass": g2, "G2_seeds": n_g2,
        "G3_endpoint_sufficiency_seeds": n_g3,
        "r_learned": [round(s["r_learned"], 3) for s in per_seed],
        "r_transfer": [round(s["r_transfer"], 3) for s in per_seed],
        "verdict": verdict,
        "interpretation": (
            "What transferred: gains JSON (external persistent cognition). "
            "What did not transfer: z-score estimators, reward EMA, ignition "
            "strength, buffer, RNG (all fresh). Model weights: none exist in "
            "this loop. Prompt artifacts: none. A G2 PASS means the learned "
            "behavior is fully externalized and model-independent by "
            "construction — it would survive ANY backend, including no model."),
    })
    chain_write(records, RECEIPT_PATH)
    print(json.dumps({"G1": g1, "G2": g2, "G3_seeds": n_g3,
                      "r_learned": [round(s['r_learned'], 3) for s in per_seed],
                      "r_transfer": [round(s['r_transfer'], 3) for s in per_seed],
                      "verdict": verdict,
                      "receipt": RECEIPT_PATH}, indent=1))


def payload_for(s):
    return {"gains": s["learned_gains"], "gain_lr": GAIN_LR, "gain_cap": 2.0,
            "theta": THETA, "channels": CHANNELS,
            "trained_on": "changing_rule/1.0.0", "seed": s["seed"],
            "n_gain_updates": s["n_gain_updates"]}


if __name__ == "__main__":
    main()
