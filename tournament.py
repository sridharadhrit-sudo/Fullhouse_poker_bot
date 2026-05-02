"""
tournament.py — Tournament mode management.

Controls ICM pressure, stack-depth adjustments, and tournament-stage logic.
The bot runs in "cash" mode by default; call configure() once at the start
of a tournament session to switch modes.

Stages:
    early        — plenty of chips, play normally
    middle       — tighten slightly, preserve stack
    bubble       — maximum fold equity, avoid marginal spots
    final_table  — ICM awareness, stack-dependent aggression
"""


class TournamentManager:

    STAGE_FACTORS = {
        "cash":        1.00,
        "early":       1.00,
        "middle":      1.10,
        "bubble":      1.75,
        "final_table": 1.40,
    }

    def __init__(self, mode: str = "cash"):
        assert mode in ("cash", "tournament"), f"Unknown mode: {mode}"
        self.mode              = mode
        self.total_players     = None
        self.paid_places       = None
        self.players_remaining = None
        self.your_stack        = None
        self.avg_stack         = None
        self.big_blind         = 10      # updated each hand

    # ── Setup ─────────────────────────────────────────────────────────────────

    def configure(self, total_players: int, paid_places: int):
        """Call once at tournament start."""
        self.total_players = total_players
        self.paid_places   = paid_places
        self.mode          = "tournament"

    def update(self, players_remaining: int, your_stack: int,
               avg_stack: int, big_blind: int):
        """Call each hand with current tournament state."""
        self.players_remaining = players_remaining
        self.your_stack        = your_stack
        self.avg_stack         = avg_stack
        self.big_blind         = max(1, big_blind)

    # ── Stage detection ───────────────────────────────────────────────────────

    def stage(self) -> str:
        if self.mode == "cash" or self.players_remaining is None:
            return "cash"

        n     = self.players_remaining
        total = self.total_players or n
        paid  = self.paid_places   or max(1, total // 10)

        if n > total * 0.65:
            return "early"
        if n > paid * 1.5:
            return "middle"
        if n <= paid * 1.15:
            return "bubble"
        return "final_table"

    # ── ICM pressure ──────────────────────────────────────────────────────────

    def bubble_factor(self) -> float:
        """
        Multiplier on equity requirements before committing chips.
        1.0 = normal, 1.75 = bubble (very tight on marginal spots).
        """
        return self.STAGE_FACTORS[self.stage()]

    def equity_premium(self) -> float:
        """
        Extra equity needed on top of pot odds before calling/raising.
        bubble_factor of 1.75 → need 7.5% more equity than normal.
        """
        return (self.bubble_factor() - 1.0) * 0.10

    # ── Stack-depth classification ────────────────────────────────────────────

    def stack_depth(self) -> str:
        """Classify current stack relative to blinds."""
        if self.your_stack is None:
            return "deep"
        bbs = self.your_stack / self.big_blind
        if bbs <= 10:
            return "short"      # push-fold only
        if bbs <= 25:
            return "medium"     # cautious, avoid big pots without strong hands
        return "deep"           # normal play

    def push_fold_bbs(self) -> float:
        """Stack depth (in BBs) below which we switch to push-or-fold."""
        stage = self.stage()
        # Bubble: push-fold kicks in earlier (preserve equity)
        thresholds = {
            "cash":        10,
            "early":       10,
            "middle":      12,
            "bubble":      15,
            "final_table": 13,
        }
        return thresholds.get(stage, 10)

    # ── Summary ───────────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "mode":             self.mode,
            "stage":            self.stage(),
            "bubble_factor":    self.bubble_factor(),
            "equity_premium":   round(self.equity_premium(), 3),
            "stack_depth":      self.stack_depth(),
            "push_fold_bbs":    self.push_fold_bbs(),
        }
