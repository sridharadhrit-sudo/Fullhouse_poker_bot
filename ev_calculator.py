"""
ev_calculator.py  —  Layer 2b
Expected Value based decisions for post-flop streets.

EV(call) = (hand_strength × pot_after_call) - amount_owed
If EV > 0 → calling is profitable.

Multi-way adjustment: equity requirement scales up with number of opponents.
Raise sizing uses pot-percentage bets adjusted for SPR.

Adaptive:
  raise_threshold_adj from OpponentModel.exploit_profile() shifts how
  aggressively we raise for value — e.g. lower vs calling stations,
  higher vs loose-aggressive players who raise back.
"""


class EVCalculator:

    # Raise if equity is this far above the break-even threshold
    RAISE_THRESHOLD = 0.22

    def decide(self, ctx: dict, opp_model=None) -> dict:
        hs   = ctx["hand_strength"]
        pot  = ctx["pot"]
        owed = ctx["amount_owed"]
        n    = ctx.get("num_opponents", 1)

        # ── Adaptive raise threshold ──────────────────────────────────────────
        raise_threshold = self.RAISE_THRESHOLD
        if opp_model is not None:
            hero_seat = ctx.get("hero_seat")
            if hero_seat is not None:
                profile         = opp_model.exploit_profile(hero_seat)
                raise_threshold = max(0.08, self.RAISE_THRESHOLD
                                      + profile.get("raise_threshold_adj", 0.0))

        # Multi-way: each additional opponent costs ~5% equity
        pot_odds = owed / (pot + owed) if owed > 0 else 0.0
        adjusted_pot_odds = min(0.9, pot_odds + max(0, (n - 1) * 0.05))

        # ── Decision logic ────────────────────────────────────────────────────
        if hs >= adjusted_pot_odds + raise_threshold:
            return {"action": "raise", "amount": self._value_raise(ctx)}

        elif hs >= adjusted_pot_odds:
            if owed == 0:
                return {"action": "check"}
            return {"action": "call"}

        elif ctx["can_check"]:
            return {"action": "check"}

        else:
            return {"action": "fold"}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _value_raise(self, ctx: dict) -> int:
        """
        Size value bets at 60–75% pot.
        Adjusts for SPR: deeper stacks → smaller relative bet.
        Short-stacked: just shove.
        """
        spr = ctx["spr"]
        if spr > 10:
            fraction = 0.6
        elif spr > 4:
            fraction = 0.75
        else:
            return ctx["your_stack"]   # shove

        target = int(ctx["pot"] * fraction) + ctx["amount_owed"]
        target = max(target, ctx["min_raise_to"])
        return min(target, ctx["your_stack"])
