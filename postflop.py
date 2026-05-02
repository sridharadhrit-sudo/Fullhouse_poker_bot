"""
postflop.py  —  Layer 2c
Post-flop situational logic with board texture reads and c-bet tracking.

Decision order:
  1. C-bet / double-barrel (preflop raiser, no bet facing us)
  2. Strong hand  → value bet (opponent-adjusted sizing)
  3. Medium hand  → pot control (check or call cheap)
  4. Draw         → semi-bluff
  5. Weak         → give up

Adaptive thresholds:
  exploit_profile() from OpponentModel shifts STRONG_HAND, MEDIUM_HAND and
  BLUFF_EQUITY per-session based on observed VPIP / WTSD / archetype.
  Confidence ramps from 0 → 1 over 15–40 hands so early play stays GTO.
"""


class PostflopLogic:

    STRONG_HAND  = 0.72
    MEDIUM_HAND  = 0.55
    BLUFF_EQUITY = 0.38

    def decide(self, ctx: dict, opp_model) -> dict:
        hs       = ctx["hand_strength"]
        board    = ctx["community_cards"]
        opp_type = opp_model.classify_table(ctx["players"])
        texture  = ctx.get("board_texture", {})

        # ── Adaptive thresholds from exploit profile ──────────────────────────
        hero_seat = ctx.get("hero_seat")
        profile   = (opp_model.exploit_profile(hero_seat)
                     if hero_seat is not None
                     else {"value_threshold_adj": 0.0,
                           "bluff_threshold_adj": 0.0,
                           "bet_size_multiplier": 1.0})

        strong_thresh = self.STRONG_HAND  + profile["value_threshold_adj"]
        medium_thresh = self.MEDIUM_HAND  + profile["value_threshold_adj"]
        bluff_thresh  = self.BLUFF_EQUITY + profile["bluff_threshold_adj"]
        size_mult     = profile["bet_size_multiplier"]

        # ── Check-raise opportunity ───────────────────────────────────────────
        # Hero checked earlier this street; opponent has since bet (amount_owed > 0)
        if ctx.get("hero_checked_this_street") and ctx["amount_owed"] > 0:
            cr = self._maybe_check_raise(ctx, opp_type, texture, profile)
            if cr is not None:
                return cr

        # ── River: dedicated handler with polarised sizing ────────────────────
        if ctx["street"] == "river":
            return self._river_decide(ctx, opp_type, texture, profile,
                                       strong_thresh, medium_thresh, bluff_thresh,
                                       size_mult)

        # ── C-bet / barrel (preflop raiser, no bet to face) ──────────────────
        if ctx.get("was_preflop_raiser") and ctx["amount_owed"] == 0:
            cbet = self._maybe_cbet(ctx, opp_type, texture, profile)
            if cbet is not None:
                return cbet

        # ── Strong hand ───────────────────────────────────────────────────────
        if hs >= strong_thresh:
            return self._value_bet(ctx, opp_type, texture, size_mult)

        # ── Medium hand ───────────────────────────────────────────────────────
        if hs >= medium_thresh:
            return self._pot_control(ctx)

        # ── Draw / semi-bluff ─────────────────────────────────────────────────
        if hs >= bluff_thresh and self._has_draw(board, ctx["your_cards"]):
            return self._semi_bluff(ctx)

        # ── Weak: give up ─────────────────────────────────────────────────────
        if ctx["can_check"]:
            return {"action": "check"}
        return {"action": "fold"}

    # ── Check-raise logic ─────────────────────────────────────────────────────

    def _maybe_check_raise(self, ctx: dict, opp_type: str, texture: dict,
                            profile: dict) -> dict:
        """
        Check-raise with very strong hands (trapping) or semi-bluff draws on flop.
        Returns None to fall through to normal call/fold logic.

        Sizing: 2.5× the opponent's bet (standard check-raise size).
        """
        hs      = ctx["hand_strength"]
        street  = ctx["street"]
        strong  = self.STRONG_HAND + profile.get("value_threshold_adj", 0.0)
        bet_in  = ctx["amount_owed"]
        pot     = ctx["pot"]

        # Very strong hand: always check-raise for value
        if hs >= strong + 0.06:   # threshold slightly higher than normal value-bet
            amount = int((pot + bet_in) * 2.5)
            amount = max(amount, ctx["min_raise_to"])
            amount = min(amount, ctx["your_stack"])
            return {"action": "raise", "amount": amount}

        # Semi-bluff check-raise on the flop only (draws have equity to make up for fold equity)
        if (street == "flop"
                and hs >= self.BLUFF_EQUITY + profile.get("bluff_threshold_adj", 0.0)
                and self._has_draw(ctx["community_cards"], ctx["your_cards"])
                and ctx["pot_odds"] > 0.30           # bad odds to call straight up
                and opp_type not in ("calling_station", "loose_aggressive")):
            amount = int((pot + bet_in) * 2.2)
            amount = max(amount, ctx["min_raise_to"])
            amount = min(amount, ctx["your_stack"])
            return {"action": "raise", "amount": amount}

        return None

    # ── River-specific handler ─────────────────────────────────────────────────

    def _river_decide(self, ctx: dict, opp_type: str, texture: dict,
                      profile: dict, strong_thresh: float, medium_thresh: float,
                      bluff_thresh: float, size_mult: float) -> dict:
        """
        River-specific decisions. Key principle: polarised sizing.
          Strong hands  → large bet (75–100 % pot) to maximise value
          Medium hands  → blocking bet (25–33 %) or check, never call big bets
          Bluff hands   → large bet only with good blockers vs foldable opponents
          Weak hands    → check or fold
        """
        hs           = ctx["hand_strength"]
        pot          = ctx["pot"]
        owed         = ctx["amount_owed"]
        can_check    = ctx["can_check"]
        blocker      = ctx.get("blocker_score",    0.0)
        range_adv    = ctx.get("range_advantage",  0.0)

        # ── Facing a bet ──────────────────────────────────────────────────────
        if owed > 0:
            pot_odds = ctx["pot_odds"]
            if hs >= strong_thresh:
                # Strong hand facing a bet: re-raise (value raise)
                raise_to = min(ctx["your_stack"], int(pot * 1.5))
                raise_to = max(raise_to, ctx["min_raise_to"])
                return {"action": "raise", "amount": raise_to}
            if hs >= medium_thresh:
                # Medium hand: call only if getting good odds
                return {"action": "call"} if pot_odds < 0.28 else {"action": "fold"}
            # Weak / bluff-catcher: call only with very good odds
            return {"action": "call"} if pot_odds < 0.20 else {"action": "fold"}

        # ── No bet to face (we act first or opponent checked) ─────────────────

        # Sizing multipliers for river (larger than earlier streets)
        base_fracs = {
            "calling_station":  0.85,
            "loose_aggressive": 0.90,
            "tight_passive":    0.65,
            "tight_aggressive": 0.75,
        }
        value_frac = max(0.65, min(1.00,
                        base_fracs.get(opp_type, 0.75) * size_mult))

        # Texture bump: wet/monotone board → charge for draws that hit
        if texture.get("wetness", 0) > 0.6 or texture.get("is_monotone"):
            value_frac = min(1.00, value_frac + 0.10)

        # Strong hand → polarised large value bet
        if hs >= strong_thresh:
            bet = int(pot * value_frac)
            bet = max(bet, ctx["min_raise_to"])
            bet = min(bet, ctx["your_stack"])
            return {"action": "raise", "amount": bet}

        # Medium hand → small blocking bet (25–33 %) or check vs aggressive opponents
        if hs >= medium_thresh:
            if opp_type == "loose_aggressive":
                return {"action": "check"}   # check and call vs LAG
            block_bet = int(pot * 0.28)
            block_bet = max(block_bet, ctx["min_raise_to"])
            block_bet = min(block_bet, ctx["your_stack"])
            return {"action": "raise", "amount": block_bet}

        # Bluff zone: only bluff with good blockers, range advantage, vs foldable opponents
        bluff_ok = (
            blocker  >= 0.35
            and range_adv >= 0.0
            and opp_type in ("tight_passive", "tight_aggressive")
            and hs >= bluff_thresh - 0.05
        )
        if bluff_ok:
            bluff_bet = int(pot * 0.75)
            bluff_bet = max(bluff_bet, ctx["min_raise_to"])
            bluff_bet = min(bluff_bet, ctx["your_stack"])
            return {"action": "raise", "amount": bluff_bet}

        # Weak or no good bluff spot → check or fold
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # ── C-bet / barrel logic ──────────────────────────────────────────────────

    def _maybe_cbet(self, ctx: dict, opp_type: str, texture: dict,
                    profile: dict = None):
        """
        Continuation bet as the preflop raiser.

        Flop:
          Dry board   (wetness < 0.35) → bet freely, small size (33% pot)
          Medium board (0.35-0.65)    → need decent equity, 50% pot
          Wet board   (> 0.65)        → need strong equity, 66% pot

        Turn (double-barrel):
          Requires was_flop_aggressor + meaningful equity.
          Tighter thresholds — opponent called flop and likely has something.

        Board texture adjustments:
          Paired board  → bluff more (opponent's range has fewer made hands)
          Ace-high board → bet more (our EP/MP range has more aces)
          Monotone board → reduce bluff frequency (flush draws dominate)

        Exploit profile adjustments:
          bluff_threshold_adj raises/lowers the equity cutoff for c-bet bluffs.

        Returns None to fall through to normal hand-strength routing.
        """
        if profile is None:
            profile = {"bluff_threshold_adj": 0.0, "bet_size_multiplier": 1.0}

        hs      = ctx["hand_strength"]
        wetness = ctx.get("board_wetness", 0.5)
        street  = ctx["street"]
        n_opps  = ctx.get("num_opponents", 1)

        # Multi-way: only c-bet with real equity
        if n_opps >= 3 and hs < 0.50:
            return None

        # Board texture modifiers on equity threshold
        tex_adj = 0.0
        if texture.get("is_paired"):
            tex_adj -= 0.05   # paired board → bluff more credible
        if texture.get("has_ace"):
            tex_adj -= 0.04   # ace board → range advantage
        if texture.get("is_monotone"):
            tex_adj += 0.08   # monotone → tighten up without the flush

        # Exploit overlay: calling stations → raise bluff threshold on c-bets
        tex_adj += profile.get("bluff_threshold_adj", 0.0) * 0.5

        # ── Flop ─────────────────────────────────────────────────────────────
        if street == "flop":
            if wetness < 0.35:
                cutoff, fraction = 0.32 + tex_adj, 0.33
            elif wetness < 0.65:
                cutoff, fraction = 0.42 + tex_adj, 0.50
            else:
                cutoff, fraction = 0.50 + tex_adj, 0.66

            if hs < cutoff:
                return None

        # ── Turn (double-barrel) ──────────────────────────────────────────────
        elif street == "turn":
            if not ctx.get("was_flop_aggressor"):
                return None   # only barrel if we c-bet the flop

            if opp_type == "tight_passive":
                cutoff = 0.40 + tex_adj
            elif opp_type == "calling_station":
                cutoff = 0.55 + tex_adj
            else:
                cutoff = 0.45 + tex_adj

            if hs < cutoff:
                return None

            fraction = 0.55 if wetness < 0.5 else 0.65

        else:
            return None   # river: handled by value_bet / bluff logic

        # ── Opponent-type sizing adjustment ───────────────────────────────────
        if opp_type == "calling_station":
            fraction = min(0.85, fraction * 1.25)
        elif opp_type == "tight_passive":
            fraction = max(0.25, fraction * 0.85)

        # Apply exploit size multiplier
        fraction = max(0.25, min(0.90,
                       fraction * profile.get("bet_size_multiplier", 1.0)))

        bet = int(ctx["pot"] * fraction)
        bet = max(bet, ctx["min_raise_to"])
        bet = min(bet, ctx["your_stack"])
        return {"action": "raise", "amount": bet}

    # ── Draw detection ────────────────────────────────────────────────────────

    def _has_draw(self, board: list, hole: list) -> bool:
        return self._has_flush_draw(board, hole) or self._has_straight_draw(board, hole)

    def _has_flush_draw(self, board: list, hole: list) -> bool:
        suits = [c[1] for c in board + hole]
        return any(suits.count(s) >= 4 for s in "shdc")

    def _has_straight_draw(self, board: list, hole: list) -> bool:
        rank_order = "23456789TJQKA"
        rank_map   = {r: i for i, r in enumerate(rank_order)}
        all_ranks  = {rank_map[c[0]] for c in board + hole}
        hole_ranks = {rank_map[c[0]] for c in hole}
        if 12 in all_ranks:
            all_ranks  |= {0}
        if 12 in hole_ranks:
            hole_ranks |= {0}
        sorted_ranks = sorted(all_ranks)
        for i in range(len(sorted_ranks) - 3):
            window = sorted_ranks[i:i + 4]
            span   = window[-1] - window[0]
            if span in (3, 4) and len(set(window)) == 4:
                if hole_ranks & set(window):
                    return True
        return False

    # ── Bet sizing strategies ─────────────────────────────────────────────────

    def _value_bet(self, ctx: dict, opp_type: str, texture: dict,
                   size_mult: float = 1.0) -> dict:
        """
        Value bet sizing by opponent type + board texture + exploit multiplier.

        Paired/dry board  → slightly smaller (opponent has fewer strong hands
                            to call with, so thin value matters)
        Wet/draw board    → larger (charge draws, protect equity)
        """
        if ctx["amount_owed"] > 0:
            raise_to = min(ctx["your_stack"], int(ctx["pot"] * 1.5))
            raise_to = max(raise_to, ctx["min_raise_to"])
            return {"action": "raise", "amount": raise_to}

        base_fractions = {
            "calling_station":  0.75,
            "loose_aggressive": 0.80,
            "tight_passive":    0.55,
            "tight_aggressive": 0.62,
        }
        fraction = base_fractions.get(opp_type, 0.65)

        # Texture adjustment
        if texture.get("is_paired"):
            fraction -= 0.05   # thinner value on paired boards
        if texture.get("wetness", 0) > 0.6:
            fraction += 0.08   # protect against draws
        if texture.get("is_monotone"):
            fraction += 0.10   # charge flush draws hard

        # Range advantage: when we have range advantage, bet larger to exploit it
        range_adv = ctx.get("range_advantage", 0.0)
        if range_adv > 0.15:
            fraction = min(0.90, fraction + 0.08)
        elif range_adv < -0.10:
            fraction = max(0.33, fraction - 0.05)

        # Exploit multiplier (e.g. 1.20 vs calling station → size up)
        fraction = max(0.33, min(0.90, fraction * size_mult))

        bet = int(ctx["pot"] * fraction)
        bet = max(bet, ctx["min_raise_to"])
        bet = min(bet, ctx["your_stack"])
        return {"action": "raise", "amount": bet}

    def _pot_control(self, ctx: dict) -> dict:
        if ctx["can_check"]:
            return {"action": "check"}
        if ctx["pot_odds"] < 0.25:
            return {"action": "call"}
        return {"action": "fold"}

    def _semi_bluff(self, ctx: dict) -> dict:
        if ctx["amount_owed"] > 0:
            if ctx["pot_odds"] < 0.35:
                return {"action": "call"}
            return {"action": "fold"}
        bet = int(ctx["pot"] * 0.5)
        bet = max(bet, ctx["min_raise_to"])
        bet = min(bet, ctx["your_stack"])
        return {"action": "raise", "amount": bet}
