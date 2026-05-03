"""
rl_policy.py — Reinforcement Learning policy for the poker bot.

Algorithm: DQN (Deep Q-Network)

WHY DQN INSTEAD OF REINFORCE FOR POKER
───────────────────────────────────────
REINFORCE is an on-policy Monte Carlo algorithm: it waits until the end of a
hand, computes a single scalar reward, and then nudges all the log-probabilities
of actions taken in that hand in proportion to that reward.  This has three
serious weaknesses in the poker setting:

1.  High variance.  A hand's outcome is dominated by luck (card run-outs,
    opponent holdings) rather than the quality of any single decision.
    Attributing the entire outcome equally to every action in the hand
    produces a noisy signal that trains slowly and unstably.

2.  Sparse, delayed rewards.  Chip changes only materialise at showdown, so
    every intermediate decision receives no informative gradient until the hand
    ends.  DQN explicitly handles this via the Bellman equation, assigning
    temporal credit through bootstrapped Q-value estimates.

3.  Sample inefficiency.  REINFORCE discards each trajectory after one update.
    DQN stores transitions in a replay buffer and replays each experience
    multiple times, yielding far better data efficiency — crucial when real
    hands are expensive to generate.

DQN's additional stability mechanisms:
  - Experience replay (ReplayBuffer): breaks correlated transitions and lets
    the network learn from a shuffled mix of past situations.
  - Target network: a periodically-frozen copy of the Q-network provides
    stable Bellman targets, preventing the oscillating "moving goalposts"
    problem of naive Q-learning.
  - Epsilon-greedy exploration with decay: ensures the policy explores
    broadly early on, then gradually shifts to exploitation as Q-values improve.

Architecture:
  Input  : 22 normalised features from the game context
  Hidden : 128 → 64 neurons, ReLU + Dropout(0.1)
  Output : 9 Q-values (one per action)

Action space:
  0  fold
  1  check / call
  2  raise ~25 % pot  (block / probe bet)
  3  raise ~50 % pot  (half pot)
  4  raise ~75 % pot  (3/4 pot)
  5  raise ~100 % pot (pot-sized)
  6  raise ~150 % pot (overbet)
  7  raise ~200 % pot (big overbet)
  8  all-in

Bet sizing noise:
  Each raise action adds ±15 % random noise to the fraction at execution time
  so opponents cannot pin down exact frequencies from bet sizes alone.
  e.g. a "50 % pot" action may come out anywhere from 42 % to 57 % pot.

Install PyTorch if not present:
    pip3 install torch --index-url https://download.pytorch.org/whl/cpu
"""

import os
import math
import random
import collections

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES      = 22    # 19 original + call_ev + can_check flag + short-stack flag
N_ACTIONS       = 9    # fold, chk/call, raise×6 sizes, all-in

# Base fractions for raise actions 2–7 (±15% noise applied at execution)
RAISE_FRACTIONS = {2: 0.25, 3: 0.50, 4: 0.75, 5: 1.00, 6: 1.50, 7: 2.00}
RAISE_NOISE     = 0.15  # ±15 % random sizing variation

# DQN hyper-parameters
GAMMA           = 0.99        # discount factor
BATCH_SIZE      = 64          # transitions per gradient update
REPLAY_CAPACITY = 100_000     # max replay buffer size
MIN_REPLAY      = 1_000       # min transitions before first update
UPDATE_EVERY    = 4           # gradient update every N transitions
TARGET_UPDATE   = 1_000       # copy main→target every N transitions
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.9995      # per-step multiplicative decay

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "rl_model.pt")


# ── Replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    """Fixed-capacity circular replay buffer for DQN transitions."""

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self._buf = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Append a single (s, a, r, s', done) transition."""
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, n: int):
        """
        Draw n transitions uniformly at random.
        Returns five tensors: states, actions, rewards, next_states, dones.
        """
        batch = random.sample(self._buf, n)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones),
        )

    def __len__(self):
        return len(self._buf)


# ── Q-network ─────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class _QNet(nn.Module):
        """
        Outputs a Q-value for each of the N_ACTIONS discrete actions.
        Architecture mirrors the old _PolicyNet so both networks have the
        same representational capacity.
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(N_FEATURES, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, N_ACTIONS),
            )

        def forward(self, x):
            return self.net(x)
else:
    _QNet = None


# ── Policy wrapper ────────────────────────────────────────────────────────────

class RLPolicy:
    """
    Neural network poker policy with DQN (Deep Q-Network).

    Training (train.py):
        policy = RLPolicy(lr=3e-4)
        policy.load()

        for hand in training_hands:
            policy.start_episode()
            action = policy.select_action(ctx, game_state)   # each decision
            policy.finish_episode(reward)                     # end of hand

        policy.save()

    Inference (bot.py):
        policy = RLPolicy()
        if policy.load():
            action = policy.select_action_greedy(ctx, game_state)
    """

    def __init__(self, lr: float = 3e-4, temperature: float = 1.0):
        # `temperature` is kept for API compatibility; DQN does not use it.
        self.temperature = temperature
        self._episodes   = 0       # hands with ≥1 postflop RL decision
        self._steps      = 0       # total transitions pushed to replay
        self.epsilon     = EPSILON_START

        # Transition storage for multi-step within a single hand
        self._prev_state  = None
        self._prev_action = None

        if not TORCH_AVAILABLE:
            print("[RL] PyTorch not found — RL policy disabled.")
            print("[RL] Install:  pip3 install torch --index-url "
                  "https://download.pytorch.org/whl/cpu")
            self._ready = False
            return

        self._net     = _QNet()
        self._target  = _QNet()
        # Initialise target weights identically to the main network
        self._target.load_state_dict(self._net.state_dict())
        self._target.eval()

        self._optimizer = optim.Adam(self._net.parameters(), lr=lr)
        self._replay    = _ReplayBuffer(REPLAY_CAPACITY)
        self._ready     = True

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, ctx: dict, game_state: dict) -> dict:
        """
        Epsilon-greedy action (training).  Stores transitions in the replay
        buffer and triggers a Bellman update when enough data has accumulated.
        Returns None if policy not ready.
        """
        if not self._ready:
            return None

        state = _featurize(ctx)

        # Push the previous step's transition (reward=0, not terminal)
        if self._prev_state is not None:
            self._push_transition(
                self._prev_state, self._prev_action, 0.0, state, False
            )

        # Epsilon-greedy action selection
        mask = _action_mask(game_state)
        if random.random() < self.epsilon:
            # Random valid action
            valid_indices = [i for i, m in enumerate(mask) if m]
            action_idx = random.choice(valid_indices)
        else:
            self._net.eval()
            with torch.no_grad():
                q_vals = self._net(
                    torch.FloatTensor(state).unsqueeze(0)
                ).squeeze(0)
            # Mask invalid actions with a large negative value
            mask_tensor = torch.tensor(
                [0.0 if m else -1e9 for m in mask], dtype=torch.float32
            )
            action_idx = (q_vals + mask_tensor).argmax().item()

        self._prev_state  = state
        self._prev_action = action_idx
        return _idx_to_action(action_idx, game_state)

    def select_action_greedy(self, ctx: dict, game_state: dict) -> dict:
        """
        Deterministic (greedy) action for live play.
        No transition stored; no epsilon exploration.
        Returns None if policy not ready.
        """
        if not self._ready:
            return None

        self._net.eval()
        state = _featurize(ctx)
        mask  = _action_mask(game_state)

        with torch.no_grad():
            q_vals = self._net(
                torch.FloatTensor(state).unsqueeze(0)
            ).squeeze(0)

        mask_tensor = torch.tensor(
            [0.0 if m else -1e9 for m in mask], dtype=torch.float32
        )
        action_idx = (q_vals + mask_tensor).argmax().item()
        return _idx_to_action(action_idx, game_state)

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def start_episode(self):
        """Clear per-hand state at the start of each hand."""
        self._prev_state  = None
        self._prev_action = None

    def finish_episode(self, reward: float):
        """
        Push the terminal transition with the hand's actual reward (done=True)
        so the Bellman backup knows there is no future value beyond this point.
        """
        if not self._ready:
            return

        if self._prev_state is not None:
            terminal_next = [0.0] * N_FEATURES
            self._push_transition(
                self._prev_state, self._prev_action, reward, terminal_next, True
            )

        self._episodes   += 1
        self._prev_state  = None
        self._prev_action = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = None):
        """Checkpoint network weights, optimiser state, and counters."""
        if not self._ready:
            return
        path = path or _DEFAULT_MODEL
        torch.save({
            "net":       self._net.state_dict(),
            "target":    self._target.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "episodes":  self._episodes,
            "steps":     self._steps,
            "epsilon":   self.epsilon,
        }, path)

    def load(self, path: str = None) -> bool:
        """
        Restore a checkpoint.  Returns True on success, False if no file found.
        """
        if not self._ready:
            return False
        path = path or _DEFAULT_MODEL
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._net.load_state_dict(ckpt["net"])
        if "target" in ckpt:
            self._target.load_state_dict(ckpt["target"])
        if "optimizer" in ckpt:
            self._optimizer.load_state_dict(ckpt["optimizer"])
        self._episodes = ckpt.get("episodes", 0)
        self._steps    = ckpt.get("steps",    0)
        self.epsilon   = ckpt.get("epsilon",  EPSILON_MIN)
        print(
            f"[RL] Loaded DQN model — "
            f"{self._episodes:,} episodes, ε={self.epsilon:.3f}"
        )
        return True

    def reset_weights(self):
        """
        Re-initialise all network parameters and clear the replay buffer.
        Use before retraining from scratch.
        """
        if not self._ready:
            return
        for layer in self._net.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self._target.load_state_dict(self._net.state_dict())
        self._replay    = _ReplayBuffer(REPLAY_CAPACITY)
        self._episodes  = 0
        self._steps     = 0
        self.epsilon    = EPSILON_START
        self._prev_state  = None
        self._prev_action = None
        print("[RL] Weights reset — ready to retrain from scratch.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def episodes_trained(self) -> int:
        return self._episodes

    # ── Internal DQN machinery ────────────────────────────────────────────────

    def _push_transition(self, state, action, reward, next_state, done):
        """
        Push one transition, decay epsilon, and conditionally trigger
        a Bellman update and/or a target-network sync.
        """
        self._replay.push(state, action, reward, next_state, done)
        self._steps += 1

        # Decay exploration rate
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        # Gradient update (only once we have enough data)
        if len(self._replay) >= MIN_REPLAY and self._steps % UPDATE_EVERY == 0:
            self._update()

        # Sync target network
        if self._steps % TARGET_UPDATE == 0:
            self._target.load_state_dict(self._net.state_dict())

    def _update(self):
        """
        One mini-batch Bellman update:
            Q(s,a) ← r + γ · max_a' Q_target(s',a')  if not done
            Q(s,a) ← r                                 if done
        Loss: MSE between predicted Q(s,a) and the target above.
        """
        states, actions, rewards, next_states, dones = \
            self._replay.sample(BATCH_SIZE)

        self._net.train()

        # Q(s, a) for the actions actually taken
        q_sa = self._net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bootstrapped target from the frozen target network
        with torch.no_grad():
            next_q = self._target(next_states).max(1)[0]
            next_q[dones] = 0.0          # terminal states have no future value
            target_vals = rewards + GAMMA * next_q

        loss = F.mse_loss(q_sa, target_vals)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
        self._optimizer.step()


# ── Feature extraction ────────────────────────────────────────────────────────

def _featurize(ctx: dict) -> list:
    """
    Convert preprocessed context → 22-element normalised feature vector.

    Index  Feature                     Range   Notes
    ─────  ─────────────────────────── ──────  ──────────────────────────────
      0    hand_strength               0–1
      1    pot_odds                    0–1
      2    spr / 20  (capped)          0–1
      3    position == early           0/1
      4    position == middle          0/1
      5    position == late            0/1
      6    street == preflop           0/1
      7    street == flop              0/1
      8    street == turn              0/1
      9    street == river             0/1
     10    board_wetness               0–1
     11    board has_ace               0/1
     12    board is_paired             0/1
     13    board is_monotone           0/1
     14    num_opponents / 5           0–1
     15    was_preflop_raiser          0/1
     16    tournament equity_premium   0–1
     17    pot / (pot + stack)         0–1
     18    amount_owed / stack         0–1
     19    call_ev = hs - pot_odds     -1–1   +ve → calling is +EV; fixes pot-odds blindness
     20    can_check flag              0/1    explicit signal to prevent fold-when-can-check
     21    short_stack = spr ≤ 2       0/1    triggers all-in recognition at low SPR
    """
    pos   = ctx.get("position", "middle")
    st    = ctx.get("street",   "preflop")
    tex   = ctx.get("board_texture", {})
    stack = max(1, ctx.get("your_stack",   1))
    pot   = max(1, ctx.get("pot",          1))
    owed  = ctx.get("amount_owed", 0)
    hs    = float(ctx.get("hand_strength", 0.5))
    po    = float(ctx.get("pot_odds",      0.0))
    spr   = float(ctx.get("spr", 10))

    return [
        hs,
        po,
        min(1.0, spr / 20.0),
        1.0 if pos == "early"  else 0.0,
        1.0 if pos == "middle" else 0.0,
        1.0 if pos == "late"   else 0.0,
        1.0 if st == "preflop" else 0.0,
        1.0 if st == "flop"    else 0.0,
        1.0 if st == "turn"    else 0.0,
        1.0 if st == "river"   else 0.0,
        float(ctx.get("board_wetness", 0.0)),
        1.0 if tex.get("has_ace")     else 0.0,
        1.0 if tex.get("is_paired")   else 0.0,
        1.0 if tex.get("is_monotone") else 0.0,
        min(1.0, float(ctx.get("num_opponents", 1)) / 5.0),
        1.0 if ctx.get("was_preflop_raiser") else 0.0,
        float(ctx.get("tournament_info", {}).get("equity_premium", 0.0)),
        float(pot) / (pot + stack),
        min(1.0, float(owed) / stack),
        # ── 3 new features ──────────────────────────────────────────────────
        max(-1.0, min(1.0, hs - po)),          # 19: call_ev
        1.0 if ctx.get("can_check") else 0.0,  # 20: can_check flag
        1.0 if spr <= 2.0 else 0.0,            # 21: short-stack flag
    ]


# ── Action helpers ────────────────────────────────────────────────────────────

def _action_mask(gs: dict) -> list:
    """
    Returns a 9-element validity mask for the expanded action space.

    Hard constraints applied (in addition to legality):

    1. Fold-when-can-check guard (existing):
       Fold is invalid when can_check=True — you cannot fold a free check.

    2. Pot-odds call guard (new):
       Fold is also invalid when facing a bet with clearly positive call EV
       (hand_strength - pot_odds > 0.12).  Forces the model to call or raise
       rather than throw away a mathematically profitable spot.

    3. Deep-stacked flop all-in guard (new):
       All-in is invalid on the flop when SPR > 6.  Deep-stacked flop shoves
       scare opponents into folding, leaving value on the table.  Forces the
       model to use a sized bet instead and keep opponents in the pot.

    4. Garbage river overbet guard (new):
       Raise 150 % and 200 % are invalid on the river when hand_strength < 0.25
       and can_check=True.  Prevents constant pot-sized bluffs with complete
       air — small bluffs (raise25–100 %) are still permitted.
    """
    owed      = gs.get("amount_owed",  0)
    stack     = gs.get("your_stack",   0)
    min_raise = gs.get("min_raise_to", 0)
    pot       = max(1, gs.get("pot",   1))
    can_check = gs.get("can_check",    False)
    street    = gs.get("street",       "")
    hs        = gs.get("hand_strength", 0.5)   # injected by train.py / bot.py
    can_raise = stack > owed and min_raise <= stack
    spr       = stack / pot

    # ── Pot odds: compute call_ev from gs fields ──────────────────────────────
    po       = owed / max(1, owed + pot) if owed > 0 else 0.0
    call_ev  = hs - po

    # ── Fix 1 + 2: fold validity ──────────────────────────────────────────────
    fold_ok  = not can_check                   # never fold a free check
    if owed > 0 and call_ev > 0.12:
        fold_ok = False                        # never fold a clearly +EV call

    # ── Fix 3: all-in validity on deep-stacked flop ───────────────────────────
    allin_ok = stack > 0
    if street == "flop" and spr > 6:
        allin_ok = False                       # size down, keep opponents in

    # ── Fix 4: block large overbets with weak hands on river ─────────────────
    overbet_ok = can_raise
    if street == "river" and hs < 0.25 and can_check:
        overbet_ok = False                     # no 150/200 % bluffs with air

    return [
        fold_ok,        # fold
        True,           # check/call
        can_raise,      # raise ~25 % pot
        can_raise,      # raise ~50 % pot
        can_raise,      # raise ~75 % pot
        can_raise,      # raise ~100 % pot
        overbet_ok,     # raise ~150 % pot (overbet)
        overbet_ok,     # raise ~200 % pot (big overbet)
        allin_ok,       # all-in
    ]


def _idx_to_action(idx: int, gs: dict) -> dict:
    """
    Convert action index → action dict.

    Raise actions (2–7) apply ±RAISE_NOISE random sizing so opponents cannot
    reverse-engineer the bot's ranges from bet sizes alone.  Overbets (idx 6,7)
    are only executed when the stack supports them; otherwise they fall back to
    all-in.

    Short-stack guard: if SPR ≤ 2 and a raise action is chosen, go all-in
    directly — fractional raises are meaningless when commitment is near-total.
    """
    pot       = max(1, gs.get("pot",           1))
    stack     = gs.get("your_stack",   0)
    min_raise = gs.get("min_raise_to", 0)
    owed      = gs.get("amount_owed",  0)
    can_check = gs.get("can_check",    False)
    spr       = stack / max(1, pot)   # stack-to-pot ratio

    if idx == 0:
        return {"action": "fold"}
    if idx == 1:
        return {"action": "check"} if can_check else {"action": "call"}
    if idx == 8:
        return {"action": "all_in"}

    # Short-stack: fractional raises are meaningless when SPR ≤ 2, just shove
    if spr <= 2.0:
        return {"action": "all_in"}

    # Apply ±RAISE_NOISE random variation to the base fraction
    base_fraction = RAISE_FRACTIONS[idx]
    noise         = random.uniform(1.0 - RAISE_NOISE, 1.0 + RAISE_NOISE)
    fraction      = base_fraction * noise

    amount = max(int(pot * fraction) + owed, min_raise)
    amount = min(amount, stack)
    if amount >= stack:
        return {"action": "all_in"}
    return {"action": "raise", "amount": amount}
