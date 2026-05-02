"""
preflop_ranges.py  —  Layer 2a
GTO-frequency-based preflop decisions loaded from gto_ranges.json.

All 169 hand types are defined in the JSON for each position and scenario.
Edit gto_ranges.json directly to adjust any frequency — no code changes needed.

Decision tree (in order):
  1. Short stack (SPR < push_fold threshold)  → push-or-fold
  2. Facing a 3-bet+                          → 4-bet or call/fold (vs_3bet table)
  3. Facing one raise                         → 3-bet / call / fold (vs_open table)
  4. No prior raises                          → open-raise / fold  (open table)

Mixed strategies: frequencies are sampled randomly each decision,
so the bot plays a mixed GTO strategy rather than a pure deterministic one.

Adaptive / Bayesian drift:
  After 15+ hands of data, exploit_profile() from OpponentModel shifts the
  sampled frequencies toward an exploitative strategy:
    - vs tight players  → steal more from late position
    - vs loose players  → tighten opens, bluff 3-bet less
    - vs calling station → value-raise more, skip light 3-bets
  Confidence ramps 0→1 over 15–40 hands so early play stays close to GTO.

VPIP-based open adjustment:
  When opponent's VPIP is known, open-raise frequencies are nudged:
    VPIP < 20% (tight) → widen late-position opens by up to +10%
    VPIP > 40% (loose) → tighten opens by up to -8% to avoid inflating pots
                         with marginal hands vs a wide calling range
"""

import json
import os
import random


_FOLD    = {"action": "fold"}
_CHECK   = {"action": "check"}
_CALL    = {"action": "call"}
_ALL_IN  = {"action": "all_in"}


class PreflopRanges:

    def __init__(self, ranges_file: str = "gto_ranges.json"):
        self._data = self._load(ranges_file)

    # ── Public interface ──────────────────────────────────────────────────────

    def decide(self, ctx: dict, opp_model=None) -> dict:
        hand      = self._classify(ctx["your_cards"])
        position  = ctx["position"]
        spr       = ctx["spr"]
        n_raises  = ctx.get("preflop_raises", 0)
        tourn     = ctx.get("tournament_info", {})

        # Gather exploit profile if we have opponent data
        profile   = None
        opp_vpip  = None
        if opp_model is not None:
            hero_seat = ctx.get("hero_seat")
            if hero_seat is not None:
                profile  = opp_model.exploit_profile(hero_seat)
                opp_vpip = opp_model.vpip_rate(hero_seat)

        # ── 1. Short-stack push-fold ──────────────────────────────────────────
        push_threshold = tourn.get("push_fold_bbs", 10)
        bb_approx      = ctx["pot"] / 1.5 if ctx["pot"] > 0 else 10
        effective_bbs  = ctx["your_stack"] / bb_approx if bb_approx > 0 else 99
        if spr < 3 or effective_bbs <= push_threshold:
            return self._push_fold(hand, spr, ctx)

        # ── 2. Facing a 3-bet ─────────────────────────────────────────────────
        if n_raises >= 2:
            return self._vs_3bet(hand, ctx, profile)

        # ── 3. Facing one raise ───────────────────────────────────────────────
        if n_raises == 1:
            return self._vs_open(hand, position, ctx, profile)

        # ── 4. Open opportunity ───────────────────────────────────────────────
        return self._open(hand, position, ctx, profile, opp_vpip)

    # ── Scenario handlers ─────────────────────────────────────────────────────

    def _open(self, hand: str, position: str, ctx: dict,
              profile: dict = None, opp_vpip: float = None) -> dict:
        freq = list(self._lookup("open", position, hand))   # mutable copy

        # ── Bayesian drift: adjust open frequency ─────────────────────────────
        if profile is not None and profile["confidence"] > 0:
            steal_adj = profile.get("steal_freq_adj", 0.0)

            # Apply steal adjustment only in late position where steals matter
            if position == "late" and steal_adj != 0.0:
                freq = self._nudge_raise(freq, steal_adj * profile["confidence"])

        # ── VPIP-based open adjustment ────────────────────────────────────────
        if opp_vpip is not None and position == "late":
            if opp_vpip < 0.20:
                # Tight opponent: widen opens (they won't defend wide)
                freq = self._nudge_raise(freq, +0.10)
            elif opp_vpip > 0.40:
                # Loose opponent: tighten opens (don't inflate pots marginal)
                freq = self._nudge_raise(freq, -0.08)

        act = self._sample(freq)
        if act == "raise":
            return {"action": "raise", "amount": self._open_size(ctx)}
        if act == "call":
            return _CHECK if ctx["amount_owed"] == 0 else _CALL
        return _CHECK if ctx["can_check"] else _FOLD

    def _vs_open(self, hand: str, position: str, ctx: dict,
                 profile: dict = None) -> dict:
        freq = list(self._lookup("vs_open", position, hand))

        # ── Bayesian drift ────────────────────────────────────────────────────
        if profile is not None and profile["confidence"] > 0:
            conf = profile["confidence"]
            arch = _profile_arch(profile)

            if arch == "calling_station":
                # Less light 3-betting — shift some raise → call
                freq = self._shift_raise_to_call(freq, 0.08 * conf)
            elif arch == "tight_passive":
                # More 3-betting — they fold to 3-bets a lot
                freq = self._nudge_raise(freq, 0.06 * conf)
            elif arch == "loose_aggressive":
                # Only 3-bet for value — they re-raise light 3-bets
                freq = self._shift_raise_to_fold(freq, 0.06 * conf)

        act = self._sample(freq)
        if act == "raise":
            return {"action": "raise", "amount": self._3bet_size(ctx)}
        if act == "call":
            return _CALL
        return _CHECK if ctx["can_check"] else _FOLD

    def _vs_3bet(self, hand: str, ctx: dict, profile: dict = None) -> dict:
        freq = list(self._lookup_flat("vs_3bet", hand))

        # ── Bayesian drift: tighten 4-bet range vs LAG ───────────────────────
        if profile is not None and profile["confidence"] > 0:
            arch = _profile_arch(profile)
            if arch == "loose_aggressive":
                # They 5-bet light — only 4-bet the nuts
                freq = self._shift_raise_to_call(freq, 0.05 * profile["confidence"])

        act = self._sample(freq)
        if act == "raise":
            return _ALL_IN                          # 4-bet is a shove
        if act == "call":
            return _CALL
        return _CHECK if ctx["can_check"] else _FOLD

    def _push_fold(self, hand: str, spr: float, ctx: dict) -> dict:
        """
        Short-stack strategy: shove or fold.
        Shove range widens as stack shrinks.
        """
        if spr < 2:
            position = "late"   # use widest range when desperate
        else:
            position = ctx["position"]

        freq = self._lookup("open", position, hand)
        if freq[0] > 0:        # hand is in our raising range
            return _ALL_IN
        return _CHECK if ctx["can_check"] else _FOLD

    # ── Sizing helpers ────────────────────────────────────────────────────────

    def _open_size(self, ctx: dict) -> int:
        target = max(ctx["min_raise_to"], int(ctx["pot"] * 0.75))
        return min(target, ctx["your_stack"])

    def _3bet_size(self, ctx: dict) -> int:
        target = max(ctx["min_raise_to"], int(ctx["current_bet"] * 3))
        return min(target, ctx["your_stack"])

    # ── Range lookup helpers ──────────────────────────────────────────────────

    def _lookup(self, section: str, position: str, hand: str) -> list:
        """Return [raise_freq, call_freq, fold_freq] for the hand."""
        sec  = self._data.get(section, {})
        pos  = sec.get(position, sec.get("middle", {}))
        return pos.get(hand, [0.0, 0.0, 1.0])

    def _lookup_flat(self, section: str, hand: str) -> list:
        sec = self._data.get(section, {})
        return sec.get(hand, [0.0, 0.0, 1.0])

    def _sample(self, freq: list) -> str:
        """Randomly sample raise / call / fold according to frequencies."""
        r   = random.random()
        cum = 0.0
        for action, f in zip(("raise", "call", "fold"), freq):
            cum += f
            if r < cum:
                return action
        return "fold"

    # ── Frequency nudge helpers ───────────────────────────────────────────────

    def _nudge_raise(self, freq: list, delta: float) -> list:
        """
        Shift `delta` probability into (positive) or out of (negative) the
        raise bucket, compensating from/to the fold bucket.
        Result is clamped and re-normalised to sum to 1.0.
        """
        r, c, f = freq
        r = max(0.0, min(1.0, r + delta))
        f = max(0.0, 1.0 - r - c)
        total = r + c + f
        if total == 0:
            return [0.0, 0.0, 1.0]
        return [r / total, c / total, f / total]

    def _shift_raise_to_call(self, freq: list, delta: float) -> list:
        """Move `delta` from raise bucket into call bucket."""
        r, c, f = freq
        move = min(delta, r)
        r -= move
        c  = min(1.0, c + move)
        f  = max(0.0, 1.0 - r - c)
        total = r + c + f
        if total == 0:
            return [0.0, 0.0, 1.0]
        return [r / total, c / total, f / total]

    def _shift_raise_to_fold(self, freq: list, delta: float) -> list:
        """Move `delta` from raise bucket into fold bucket."""
        r, c, f = freq
        move = min(delta, r)
        r -= move
        f  = min(1.0, f + move)
        total = r + c + f
        if total == 0:
            return [0.0, 0.0, 1.0]
        return [r / total, c / total, f / total]

    # ── Card classification ───────────────────────────────────────────────────

    def _classify(self, cards: list) -> str:
        """Convert ["As","Kh"] → "AKo" / "AKs" / "AA"."""
        rank_order = "23456789TJQKA"
        rank_map   = {r: i for i, r in enumerate(rank_order)}
        r1, s1     = cards[0][0], cards[0][1]
        r2, s2     = cards[1][0], cards[1][1]
        if rank_map[r1] < rank_map[r2]:
            r1, r2 = r2, r1
            s1, s2 = s2, s1
        if r1 == r2:
            return r1 + r2
        return r1 + r2 + ("s" if s1 == s2 else "o")

    # ── JSON loader ───────────────────────────────────────────────────────────

    def _load(self, filename: str) -> dict:
        search = [
            os.path.join(os.path.dirname(__file__), filename),
            filename,
        ]
        for path in search:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        return {}   # fallback: everything folds


# ── Module-level helper ───────────────────────────────────────────────────────

def _profile_arch(profile: dict) -> str:
    """
    Infer dominant archetype from an exploit_profile dict by inspecting
    which adjustments are most active.
    """
    val_adj   = profile.get("value_threshold_adj", 0.0)
    bluff_adj = profile.get("bluff_threshold_adj", 0.0)
    size_mult = profile.get("bet_size_multiplier", 1.0)

    if bluff_adj > 0.10 and size_mult > 1.10:
        return "calling_station"
    if bluff_adj > 0.05 and val_adj > 0.02:
        return "loose_aggressive"
    if val_adj > 0.03 and size_mult < 0.95:
        return "tight_passive"
    if val_adj > 0.03 and size_mult >= 0.95:
        return "tight_aggressive"
    return "unknown"
