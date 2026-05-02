"""
preprocessor.py  —  Layer 1
Converts raw game_state into an enriched context dict used by all modules.

Fields added beyond raw game_state:
  hand_strength      float   equity vs all active opponents (Monte Carlo)
  pot_odds           float   break-even call threshold
  spr                float   stack-to-pot ratio
  position           str     "early" | "middle" | "late"
  num_opponents      int     active opponents at the table
  preflop_raises     int     number of raises made preflop so far
  hero_seat          int     our seat number (None if undetectable)
  was_preflop_raiser bool    True if we raised preflop this hand
  was_flop_aggressor bool    True if we bet/raised on the flop
  board_wetness      float   0.0 (dry) → 1.0 (flushy+connected)
  board_texture      dict    detailed board analysis (see _board_texture)
  tournament_info    dict    stage/bubble_factor from TournamentManager
  street_idx         int     0-3 numeric street
"""

import eval7


class Preprocessor:

    def process(self, gs: dict, tournament=None) -> dict:
        ctx = dict(gs)

        # ── Opponents ─────────────────────────────────────────────────────────
        num_opps          = self._count_opponents(gs["players"])
        ctx["num_opponents"] = num_opps

        # ── Hand strength ─────────────────────────────────────────────────────
        ctx["hand_strength"] = self._hand_strength(
            gs["your_cards"], gs["community_cards"], num_opps
        )

        # ── Pot odds & SPR ────────────────────────────────────────────────────
        owed             = gs["amount_owed"]
        pot              = gs["pot"]
        ctx["pot_odds"]  = owed / (pot + owed) if owed > 0 else 0.0
        ctx["spr"]       = gs["your_stack"] / pot if pot > 0 else float("inf")

        # ── Position ──────────────────────────────────────────────────────────
        ctx["position"]  = self._position(gs["players"])

        # ── Preflop context ───────────────────────────────────────────────────
        action_log = gs.get("action_log", [])
        ctx["preflop_raises"] = sum(
            1 for a in action_log
            if a.get("street", "preflop") == "preflop"
            and a.get("action") == "raise"
        )

        hero_seat                = self._find_hero_seat(gs["players"])
        ctx["hero_seat"]         = hero_seat
        ctx["was_preflop_raiser"] = self._was_aggressor(action_log, hero_seat, "preflop")

        # ── C-bet / barrel tracking ───────────────────────────────────────────
        ctx["was_flop_aggressor"] = self._was_aggressor(action_log, hero_seat, "flop")

        # ── Board texture ─────────────────────────────────────────────────────
        board                   = gs["community_cards"]
        ctx["board_wetness"]    = self._board_wetness(board)
        ctx["board_texture"]    = self._board_texture(board)

        # ── Tournament context ────────────────────────────────────────────────
        ctx["tournament_info"] = tournament.info() if tournament else {
            "mode": "cash", "stage": "cash",
            "bubble_factor": 1.0, "equity_premium": 0.0,
            "stack_depth": "deep", "push_fold_bbs": 10,
        }

        # ── Street index ──────────────────────────────────────────────────────
        ctx["street_idx"] = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}[
            gs["street"]
        ]

        return ctx

    # ── Opponent count ────────────────────────────────────────────────────────

    def _count_opponents(self, players: list) -> int:
        active = [
            p for p in players
            if p.get("is_active", True)
            and not (p.get("is_me") or p.get("is_hero") or p.get("you"))
        ]
        return max(1, len(active))

    # ── Hand strength (Monte Carlo) ───────────────────────────────────────────

    def _hand_strength(self, hole: list, board: list, num_opponents: int = 1) -> float:
        iters = 500 if not board else (1000 if len(board) < 5 else 200)
        return self._mc_equity(hole, board, iters, num_opponents)

    def _mc_equity(self, hole: list, board: list, iterations: int,
                   num_opponents: int = 1) -> float:
        try:
            deck        = eval7.Deck()
            hole_cards  = [eval7.Card(c) for c in hole]
            board_cards = [eval7.Card(c) for c in board]
            for c in hole_cards + board_cards:
                deck.cards.remove(c)

            remaining_board = 5 - len(board_cards)
            safe_opps = min(num_opponents,
                            max(1, (len(deck.cards) - remaining_board) // 2))
            wins = 0
            for _ in range(iterations):
                deck.shuffle()
                idx       = 0
                opp_hands = []
                for _ in range(safe_opps):
                    opp_hands.append(deck.cards[idx:idx + 2])
                    idx += 2
                full_board = board_cards + deck.cards[idx:idx + remaining_board]
                my_score   = eval7.evaluate(hole_cards + full_board)
                if all(my_score > eval7.evaluate(o + full_board) for o in opp_hands):
                    wins += 1
            return wins / iterations
        except Exception:
            return 0.5

    # ── Position ──────────────────────────────────────────────────────────────

    def _find_hero_seat(self, players: list):
        for p in players:
            if p.get("is_me") or p.get("is_hero") or p.get("you"):
                return p.get("seat")
        return None

    def _position(self, players: list) -> str:
        active = [p for p in players if p.get("is_active", True)]
        n      = len(active)
        if n == 0:
            return "middle"

        dealer_seat = None
        hero_seat   = None
        for p in active:
            if p.get("is_dealer") or p.get("is_button") or p.get("position") == "BTN":
                dealer_seat = p.get("seat")
            if p.get("is_me") or p.get("is_hero") or p.get("you"):
                hero_seat = p.get("seat")

        if dealer_seat is None or hero_seat is None:
            return "middle"

        active_seats = sorted(p["seat"] for p in active)
        try:
            dealer_idx = active_seats.index(dealer_seat)
            hero_idx   = active_seats.index(hero_seat)
        except ValueError:
            return "middle"

        seats_from_btn = (hero_idx - dealer_idx) % n
        if n <= 3:
            return "late" if seats_from_btn == 0 else "early"
        if seats_from_btn == 0 or seats_from_btn >= n - 1:
            return "late"
        if seats_from_btn <= max(2, n // 3):
            return "early"
        return "middle"

    # ── Action-log helpers ────────────────────────────────────────────────────

    def _was_aggressor(self, action_log: list, hero_seat, street: str) -> bool:
        """True if hero was the last raiser/bettor on the given street."""
        if hero_seat is None:
            return False
        for event in reversed(action_log):
            if event.get("street") == street and event.get("action") in ("raise", "bet"):
                return event.get("seat") == hero_seat
        return False

    # ── Board texture ─────────────────────────────────────────────────────────

    def _board_wetness(self, board: list) -> float:
        """
        0.0 = dry rainbow/unconnected  →  1.0 = flushy + connected.
        Two components: flush + connectivity, capped at 1.0.
        """
        if not board:
            return 0.0

        rank_order = "23456789TJQKA"
        rank_map   = {r: i for i, r in enumerate(rank_order)}
        suits      = [c[1] for c in board]
        ranks      = sorted(rank_map[c[0]] for c in board)

        max_suit      = max(suits.count(s) for s in "shdc")
        flush_score   = {1: 0.0, 2: 0.3, 3: 0.6}.get(max_suit, 0.6)

        if len(ranks) >= 2:
            gaps          = [ranks[i + 1] - ranks[i] for i in range(len(ranks) - 1)]
            avg_gap       = sum(gaps) / len(gaps)
            connect_score = 0.4 if avg_gap <= 1 else (0.2 if avg_gap <= 2 else 0.0)
        else:
            connect_score = 0.0

        return min(1.0, flush_score + connect_score)

    def _board_texture(self, board: list) -> dict:
        """
        Detailed board analysis used by postflop for texture-aware decisions.

        Returns:
          has_ace        bool   — ace on board (our EP range advantage)
          is_paired      bool   — one pair on board (bluffing more credible)
          is_trips       bool   — trips on board
          is_monotone    bool   — all same suit (danger zone)
          is_two_tone    bool   — two suits (flush draw possible)
          high_card      int    — highest board card rank (0-12 scale)
          wetness        float  — same as board_wetness
        """
        if not board:
            return {
                "has_ace": False, "is_paired": False, "is_trips": False,
                "is_monotone": False, "is_two_tone": False,
                "high_card": 0, "wetness": 0.0,
            }

        rank_order  = "23456789TJQKA"
        rank_map    = {r: i for i, r in enumerate(rank_order)}

        ranks  = [rank_map[c[0]] for c in board]
        suits  = [c[1] for c in board]

        from collections import Counter
        rank_counts = Counter(ranks)
        max_count   = max(rank_counts.values())
        n_suits     = len(set(suits))

        return {
            "has_ace":    12 in ranks,
            "is_paired":  max_count >= 2,
            "is_trips":   max_count >= 3,
            "is_monotone": n_suits == 1,
            "is_two_tone": n_suits == 2,
            "high_card":  max(ranks),
            "wetness":    self._board_wetness(board),
        }
