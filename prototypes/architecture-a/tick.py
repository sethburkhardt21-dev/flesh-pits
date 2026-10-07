"""Architecture A — the tick: sole-path wiring.

Data flow per tick:
  perception (observation)
    -> specialists: channel -> (stimulus, payload)
    -> attention arbitration over habituated bids (learned gains)
    -> BOUNDED workspace buffer (capacity K, eviction policy)
    -> recurrent ignition per buffered item (hard gate)
    -> BROADCAST to declared consumers  [THE SOLE PATH]
    -> consumers: memory_admit, self_model_update, planner_input,
                  attention_update, consolidation_eligible, report
    -> action = arbitration winner channel (closed loop via env)
    -> reward -> feedback item -> broadcast -> attention_update (gain learning)

SOLE-PATH ENFORCEMENT (by construction):
  - consumers are registered ONLY on the BroadcastBus;
  - the tick keeps no consumer references (bus owns them);
  - payloads cross to consumers only inside broadcast envelopes;
  - the attention_update gain writer is reachable only through a
    feedback broadcast item. Gains cannot be touched any other way.

The ``specialists`` argument: list of (channel, fn) where
fn(observation, tick) -> (stimulus: float, payload: dict).
"""
from __future__ import annotations

import copy

try:
    from .workspace_buffer import BoundedWorkspace, admit_swallowing_refusal
    from .attention import AttentionArbitrator
    from .ignition import RecurrentIgnition
    from .broadcast import BroadcastBus, CONSUMER_REGISTRY
    from .consumers import build_consumers
except ImportError:  # standalone script run
    from workspace_buffer import BoundedWorkspace, admit_swallowing_refusal
    from attention import AttentionArbitrator
    from ignition import RecurrentIgnition
    from broadcast import BroadcastBus, CONSUMER_REGISTRY
    from consumers import build_consumers


class WorkspaceTick:
    def __init__(self, channels, specialists, *, capacity=3,
                 gain_lr=0.05, frozen_gains=False,
                 ignition_kwargs=None, alpha=0.01, initial_var=1.0,
                 admission_floor=0.0,
                 arbitrator_cls=None, context_fn=None):
        """Additive extensions (2026-10-07, K9/K10; defaults preserve the
        proven behavior exactly):
          arbitrator_cls -- AttentionArbitrator subclass to instantiate
                            instead of AttentionArbitrator (default None
                            -> AttentionArbitrator, byte-identical path).
          context_fn     -- callable observation -> hashable context, fed
                            to arbitrator.set_context() before each
                            arbitration (default None -> no context;
                            requires an arbitrator with set_context)."""
        self.channels = list(channels)
        self.specialists = {c: fn for c, fn in specialists}
        if set(self.specialists) != set(self.channels):
            raise ValueError("specialists must cover exactly the channels")
        self.buffer = BoundedWorkspace(
            capacity, consumer_registry=set(CONSUMER_REGISTRY))
        arb_cls = arbitrator_cls or AttentionArbitrator
        self.arbitrator = arb_cls(
            self.channels, alpha=alpha, initial_var=initial_var,
            gain_lr=gain_lr, frozen=frozen_gains)
        if context_fn is not None and not hasattr(self.arbitrator,
                                                   "set_context"):
            raise ValueError("context_fn requires an arbitrator with "
                             "set_context()")
        self.context_fn = context_fn
        self.ignition = RecurrentIgnition(**(ignition_kwargs or {}))
        self.bus = BroadcastBus()
        # The tick holds NO consumer references after registration:
        # consumers exist only behind the bus. The planner queue is read
        # through the bus's sanctioned reverse accessor (consumer -> tick),
        # which is how action selection consumes broadcast output.
        _held = build_consumers(self.arbitrator)
        for name, consumer in _held.items():
            if name == "planner_input":
                self.bus.register(name, consumer.consume,
                                  state_provider=consumer.pop_proposal)
            else:
                self.bus.register(name, consumer.consume)
        del _held, consumer
        self.admission_floor = float(admission_floor)
        self.tick_index = 0
        self.total_broadcast = 0
        self.total_ignited = 0
        self.actions: list[str] = []
        self.rewards: list[float] = []
        self._reward_ema = 0.0
        self._reward_seen = 0
        self._last_action = self.channels[0]

    # ---------------------------------------------------------------
    def _select_action(self, proposal, decision) -> tuple:
        """Action-selection hook. Default (R1, NR-A-004, wired
        2026-10-07 after K6 + K8 confirmation): the planner proposal —
        which arrives only via broadcast after ignition — is the sole
        path when present; with no proposal (nothing ignited this
        tick), act on the graded arbitration winner WITHOUT
        propagating anything to consumers (no broadcast items exist on
        this path, so no deliveries happen; the ignition gate remains
        the sole decider of propagation — K2/K3 intact, verified K8).
        Subclasses may still override this hook for alternative
        resolution paths."""
        if proposal and proposal.get("proposed_action"):
            return proposal["proposed_action"], {"path": "ignited_proposal"}
        return decision["winner"], {"path": "sub_ignition_explore"}

    def step(self, observation: dict, env=None) -> dict:
        """One closed-loop tick. Returns a machine-readable trace."""
        self.tick_index += 1
        t = self.tick_index
        trace = {"tick": t}

        # 0. context conditioning (K10 extension; default None -> skipped).
        #    The context is input conditioning, set before arbitration.
        if self.context_fn is not None:
            self.arbitrator.set_context(self.context_fn(observation))

        # 1. specialists -> stimuli
        stimuli, payloads = {}, {}
        for c in self.channels:
            stimulus, payload = self.specialists[c](observation, t)
            stimuli[c] = float(stimulus)
            payloads[c] = dict(payload)
        trace["stimuli"] = stimuli

        # 2. attention arbitration (graded selection)
        decision = self.arbitrator.arbitrate(stimuli)
        trace["arbitration"] = {k: decision[k] for k in
                                ("winner", "margin", "competed_bids")}

        # 3. admit every channel's item to the bounded buffer
        #    (capacity competition happens here, K1).
        #    Declared consumers: the arbitration WINNER's item declares the
        #    full registry (including planner_input -> action proposals);
        #    non-winner items declare everything EXCEPT planner_input, so
        #    action selection follows the winner, not a FIFO queue.
        winner = decision["winner"]
        planner_set = list(CONSUMER_REGISTRY)
        nonplanner_set = [c for c in CONSUMER_REGISTRY if c != "planner_input"]
        admissions = []
        for c in self.channels:
            bid = decision["competed_bids"][c]
            if bid < self.admission_floor:
                continue
            payload = dict(payloads[c])
            payload.update({"channel": c, "bid": bid,
                            "tick": t, "consolidation_candidate": bid > 0.5,
                            "proposed_action": c,
                            "arbitration_winner": (c == winner)})
            rec = admit_swallowing_refusal(
                self.buffer, kind="specialist_output", payload=payload,
                bid=bid,
                declared_consumers=(planner_set if c == winner
                                    else nonplanner_set))
            admissions.append(rec)
        trace["admissions"] = [
            {"item_id": r["item_id"], "admitted": r["admitted"],
             "reason": r["reason"], "evicted": r["evicted_item_id"]}
            for r in admissions]

        # 4. recurrent ignition per buffered item (hard gate).
        #    Within-tick independence: every item seeds from the SAME
        #    carried strength (no item-to-item contamination — measured
        #    2026-10-07: sequential seeding made ignition order-dependent).
        #    Cross-tick reverberation: the tick's dominant settled strength
        #    carries forward as next tick's seed (working-memory persistence).
        ignitions, broadcast_items = [], []
        seed_strength = self.ignition.strength
        tick_strength = 0.0
        for item in self.buffer.contents():
            self.ignition.strength = seed_strength
            rec = self.ignition.ignite(item.bid)
            tick_strength = max(tick_strength, rec["strength"])
            ignitions.append({"item_id": item.item_id,
                              "ignited": rec["ignited"],
                              "strength": rec["strength"],
                              "cycles": rec["cycles"]})
            if rec["ignited"]:
                broadcast_items.append(item)
        self.ignition.strength = tick_strength
        trace["ignitions"] = ignitions
        self.total_ignited += len(broadcast_items)

        # 5. broadcast: THE sole path to consumers
        deliveries = []
        for item in broadcast_items:
            payload = dict(item.payload)
            payload["ignited"] = True
            ev = self.bus.broadcast(item.item_id, item.kind, payload,
                                    item.declared_consumers, t)
            deliveries.append(ev)
        trace["deliveries"] = deliveries
        self.total_broadcast += len(deliveries)
        self.buffer.drain()
        self.buffer.tick()

        # 6. action selection consumes the planner queue THROUGH the bus
        #    (broadcast -> planner_input is the sole path workspace content
        #    takes to reach action selection). If nothing ignited this
        #    tick, _select_action's default R1 path acts on the graded
        #    arbitration winner without propagating anything to consumers.
        #    Routed through _select_action so alternative resolution
        #    paths can be tested as explicit subclasses (NR-A-004).
        proposal = self.bus.read_consumer("planner_input")
        action, selection = self._select_action(proposal, decision)
        self._last_action = action
        self.actions.append(action)
        trace["action"] = action
        trace["action_selection"] = selection
        if env is not None:
            reward = env.step(action)
            self.rewards.append(float(reward))
            self._reward_seen += 1
            # Slow baseline (alpha=0.01): the reference must NOT track the
            # current policy's reward, or the error signal vanishes under
            # distribution shift and learning provably stalls (measured
            # 2026-10-07: fast alpha=0.05 baseline -> gain froze, agent
            # perseverated). Slow baseline keeps the error alive long
            # enough for the gain to cross the switching point.
            a = 0.01
            self._reward_ema = (1 - a) * self._reward_ema + a * float(reward)
            # feedback item closes the learning loop THROUGH broadcast.
            # reward_baseline is a FIXED aspiration level (0.5, the reward
            # midpoint), NOT a tracking EMA: a tracking baseline provably
            # erases the error signal under distribution shift (measured
            # twice 2026-10-07 — fast EMA froze gains; slow EMA shrank the
            # error 10x). A constant baseline is the standard
            # variance-reduction reference; it cannot vanish.
            fb = {"channel": action, "bid": 1.0, "tick": t,
                  "utility": {action: float(reward)},
                  "reward_baseline": 0.5,
                  "consolidation_candidate": False,
                  "proposed_action": None, "ignited": True}
            self.bus.broadcast(f"fb-{t}", "feedback", fb,
                               ["attention_update"], t)
            trace["reward"] = float(reward)
        return trace

    def stats(self) -> dict:
        return {"tick": self.tick_index, "actions": len(self.actions),
                "broadcast_items": self.total_broadcast,
                "ignited_items": self.total_ignited,
                "mean_reward": (sum(self.rewards) / len(self.rewards)
                                if self.rewards else 0.0),
                "gains": dict(self.arbitrator.gains),
                "buffer": self.buffer.stats()}
