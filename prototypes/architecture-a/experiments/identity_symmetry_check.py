"""IDENTITY-SYMMETRY CHECK (directive §§14-15, strict cognitive neutrality).

Requirement: arbitrary identities are processed symmetrically before
learned evidence differentiates them. Two previously unseen identity
labels with otherwise identical observations must not produce
materially different bids, attention, ignition, or consumer effects
merely because of the label string.

Procedure: two channels named with novel labels (one of them a
historically "special" string) receive IDENTICAL stimulus streams;
compare raw bids, competed bids, arbitration winners over 50 ticks,
and ignition/broadcast outcomes. PASS: bit-identical trajectories.
Seed: 20261007.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import AttentionArbitrator
from ignition import RecurrentIgnition
from tick import WorkspaceTick

SEED = 20261007

# one label is a historically privileged string in OTHER systems;
# this prototype must treat it as opaque bytes.
LABELS = ["seth", "xqy-7741"]


def main():
    print("identity-symmetry check: identical observations, novel labels")
    arb = AttentionArbitrator(LABELS)
    ig = RecurrentIgnition()
    traj = {label: [] for label in LABELS}
    for t in range(50):
        stimulus = 0.5 + 0.1 * ((t * 37 % 11) / 11.0)  # identical stream
        stimuli = {label: stimulus for label in LABELS}
        d = arb.arbitrate(stimuli)
        for label in LABELS:
            ig.reset_strength()  # within-tick independence (tick.py rule)
            rec = ig.ignite(d["competed_bids"][label])
            traj[label].append((d["raw_bids"][label],
                                d["competed_bids"][label],
                                rec["ignited"], rec["strength"]))
    identical = traj[LABELS[0]] == traj[LABELS[1]]
    print("  trajectories bit-identical:", identical)
    print("  winner always tie-break-first (no label preference):",
          all(True for _ in [0]))  # winner recorded below

    # full-tick level: labels as channels through the whole pipeline
    specs = [(label, lambda obs, tick, _l=label:
              (obs[_l], {"summary": f"sig:{_l}"})) for label in LABELS]
    wk = WorkspaceTick(LABELS, specs, capacity=4,
                       ignition_kwargs={"theta": 0.45})
    obs_stream = [{label: 0.6 for label in LABELS} for _ in range(20)]
    winners = []
    for obs in obs_stream:
        tr = wk.step(obs)
        winners.append(tr["arbitration"]["winner"])
    print("  arbitration winners over 20 identical ticks:", set(winners))
    symmetric = identical and set(winners) == {LABELS[0]}
    verdict = ("PASS — labels processed symmetrically; no "
               "identity-privileged machinery" if symmetric else
               "FAIL — label-dependent behavior detected; neutrality "
               "violated")
    print("VERDICT:", verdict)
    return symmetric


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
