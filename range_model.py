"""
range_model.py  —  Layer 1b
Range advantage, blocker scoring, and opponent range tightness.

Provides three methods used by preprocessor and postflop:
  range_advantage()         — which player has a stronger range on this board
  blocker_score()           — how much do our hole cards block strong combos
  opponent_range_tightness() — infer how narrow opponent's range is from actions
"""

from collections import Counter


class RangeModel:

    # ── Range advantage ───────────────────────────────────────────────────────

    def range_advantage(self, board: list, our_position: str,
                        our_preflop_raiser: bool,
                        opp_preflop_raised: bool) -> float:
        """
        Returns -0.5 to +0.5 (positive = we have range advantage on this board).

        Parameters
        ----------
        board              list[str]   community cards dealt so far
        our_position       str         "early" | "middle" | "late"
        our_preflop_raiser bool        True if we raised preflop
        opp_preflop_raised bool        True if opponent raised preflop
        """
        if not board:
            return 0.0

        rank_order = "23456789TJQKA"
        rank_map   = {r: i for i, r in enumerate(rank_order)}

        ranks  = sorted((rank_map[c[0]] for c in board), reverse=True)
        suits  = [c[1] for c in board]
        high   = ranks[0] if ranks else 0

        # Count suit frequencies for monotone detection
        suit_counts = Counter(suits)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0

        is_ep_mp = our_position in ("early", "middle")
        is_late  = our_position == "late"

        score = 0.0

        # Ace-high board
        if high == 12:  # Ace
            if is_ep_mp or our_preflop_raiser:
                score += 0.25
            else:
                score -= 0.10

        # King-high board (no ace)
        elif high == 11:  # King
            if is_ep_mp:
                score += 0.12
            else:
                score += 0.02

        # Connected low/mid board: avg gap ≤ 2 and high card ≤ 8 (rank index 6 = '8')
        if len(ranks) >= 2:
            sorted_asc = sorted(ranks)
            gaps = [sorted_asc[i + 1] - sorted_asc[i]
                    for i in range(len(sorted_asc) - 1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0

            if avg_gap <= 2 and high <= 6:   # high_card ≤ 8 means rank index ≤ 6
                if is_late:
                    score += 0.12
                elif is_ep_mp:
                    score -= 0.08

        # Paired board
        rank_counts = Counter(ranks)
        if max(rank_counts.values()) >= 2:
            if our_preflop_raiser:
                score += 0.10

        # Monotone board
        if max_suit_count == len(board) and len(board) >= 2:
            score -= 0.08

        # Opponent 3-bet preflop and board is low (high_card rank index < 8 = 'T')
        if opp_preflop_raised and high < 8:   # rank index 8 = 'T'
            score += 0.05

        # Clamp
        return max(-0.5, min(0.5, score))

    # ── Blocker score ─────────────────────────────────────────────────────────

    def blocker_score(self, hole: list, board: list) -> float:
        """
        Returns 0.0–1.0. How much do our hole cards block opponent's strong combos?

        Parameters
        ----------
        hole   list[str]   our two hole cards
        board  list[str]   community cards dealt so far
        """
        if not hole:
            return 0.0

        rank_order = "23456789TJQKA"
        rank_map   = {r: i for i, r in enumerate(rank_order)}

        score = 0.0

        # ── Flush blockers ────────────────────────────────────────────────────
        board_suits = [c[1] for c in board]
        suit_counts = Counter(board_suits)

        # Find dominant suit (appears 2+ times on board)
        dominant_suit = None
        dominant_count = 0
        for suit, cnt in suit_counts.items():
            if cnt >= 2 and cnt > dominant_count:
                dominant_suit = suit
                dominant_count = cnt

        if dominant_suit is not None:
            for card in hole:
                card_suit = card[1]
                card_rank = card[0]
                if card_suit == dominant_suit:
                    if card_rank == 'A':
                        score += 0.30   # nut flush blocker
                    else:
                        score += 0.08   # any card of dominant suit

        # ── Straight blockers ─────────────────────────────────────────────────
        # Check if each hole card completes a straight with 4 board cards
        board_ranks = set(rank_map[c[0]] for c in board)
        # Add ace-low
        if 12 in board_ranks:
            board_ranks = board_ranks | {-1}   # -1 as low ace placeholder

        for hole_card in hole:
            hole_rank = rank_map[hole_card[0]]
            # Check all possible 5-card straights that include this hole rank
            # A straight spans 5 consecutive ranks: [r, r+1, r+2, r+3, r+4]
            # We need hole_rank in the window and 4 board cards to fill the rest
            for start in range(hole_rank - 4, hole_rank + 1):
                window = set(range(start, start + 5))
                # Adjust for valid rank range
                if min(window) < 0 or max(window) > 12:
                    continue
                needed_from_board = window - {hole_rank}
                if needed_from_board.issubset(board_ranks):
                    score += 0.12   # hole card completes a straight with 4 board cards

        # ── Board rank blockers (blocks sets/boats) ───────────────────────────
        board_rank_list = [rank_map[c[0]] for c in board]
        board_rank_counter = Counter(board_rank_list)
        is_board_paired = any(v >= 2 for v in board_rank_counter.values())

        for hole_card in hole:
            hole_rank = rank_map[hole_card[0]]
            if hole_rank in board_rank_counter:
                block = 0.10
                if is_board_paired:
                    block *= 2
                score += block

        # ── Top board card held ───────────────────────────────────────────────
        if board_rank_list:
            top_board_rank = max(board_rank_list)
            for hole_card in hole:
                if rank_map[hole_card[0]] == top_board_rank:
                    score += 0.12

        # Clamp
        return max(0.0, min(1.0, score))

    # ── Opponent range tightness ──────────────────────────────────────────────

    def opponent_range_tightness(self, action_log: list, opp_seat: int,
                                  current_street: str) -> str:
        """
        Returns "very_tight", "tight", "medium", "wide" based on opponent actions.

        Parameters
        ----------
        action_log      list[dict]  full action log for the hand
        opp_seat        int         opponent's seat number
        current_street  str         current street (used to filter relevant actions)
        """
        if opp_seat is None:
            return "medium"

        raised_preflop  = False
        raised_postflop = False
        calls_count     = 0

        for event in action_log:
            if event.get("seat") != opp_seat:
                continue
            act    = event.get("action", "")
            street = event.get("street", "preflop")

            if street == "preflop":
                if act == "raise":
                    raised_preflop = True
            else:
                if act in ("raise", "bet"):
                    raised_postflop = True
                elif act == "call":
                    calls_count += 1

        if raised_preflop and raised_postflop:
            return "very_tight"
        if raised_preflop or raised_postflop:
            return "tight"
        if calls_count >= 2:
            return "medium"
        return "wide"
