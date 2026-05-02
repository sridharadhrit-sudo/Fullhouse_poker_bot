"""
opponent_model.py  —  Layer 4
Tracks opponent behaviour across the action_log and classifies each
player as one of 4 archetypes that post-flop logic can exploit.

Stats tracked:
  VPIP   — Voluntarily Put money In Pot (preflop). High = loose.
  PFR    — Pre-Flop Raise %. High = aggressive preflop.
  AF     — Aggression Factor postflop = (bets+raises) / calls. High = aggressive.
  WTSD%  — Went To ShowDown %. High = station who won't fold to river bets.

Archetypes:
  tight_passive    (nit)              VPIP<20, AF<1
  tight_aggressive (TAG)              VPIP<30, AF>1  ← most dangerous
  loose_passive    (calling station)  VPIP>35, AF<1.5
  loose_aggressive (LAG / maniac)     VPIP>35, AF>=1.5

Exploit profile:
  Each seat gets an exploit_profile dict with scalar adjustments that
  downstream modules can apply directly to their thresholds / sizings.

Fix applied:
  - update() previously replayed the entire action_log on every call,
    causing every stat to be inflated by 4x by the river.
    Now uses _last_log_len to process only new entries each call.
  - VPIP and PFR are counted at most once per player per hand via
    _vpip_counted / _pfr_counted sets that reset on each new hand.
"""

from collections import defaultdict, Counter


# Minimum hands before we trust stats enough to exploit
_MIN_HANDS_CLASSIFY = 5
_MIN_HANDS_EXPLOIT  = 8


class OpponentModel:

    def __init__(self):
        self._hands_played   = defaultdict(int)
        self._vpip           = defaultdict(int)
        self._pfr            = defaultdict(int)
        self._postflop_aggr  = defaultdict(int)
        self._postflop_call  = defaultdict(int)

        # WTSD tracking — incremented when a showdown is detected
        self._wtsd_reached   = defaultdict(int)   # hands that reached showdown
        self._wtsd_counted   = set()              # prevent double-count per hand

        # Duplicate-prevention state
        self._last_log_len   = 0
        self._vpip_counted   = set()   # seats already counted for VPIP this hand
        self._pfr_counted    = set()   # seats already counted for PFR this hand

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, action_log: list, players: list):
        """
        Parse the action_log and update per-seat stats.
        Called once per decide() — processes only new log entries.
        """
        # Detect new hand: log shrinks (engine resets it between hands)
        if len(action_log) < self._last_log_len:
            self._last_log_len = 0
            self._vpip_counted.clear()
            self._pfr_counted.clear()
            self._wtsd_counted.clear()
            for p in players:
                seat = p.get("seat")
                if seat is not None:
                    self._hands_played[seat] += 1

        new_entries        = action_log[self._last_log_len:]
        self._last_log_len = len(action_log)

        for event in new_entries:
            seat   = event.get("seat")
            act    = event.get("action", "")
            street = event.get("street", "preflop")

            if seat is None:
                continue

            if street == "preflop":
                if act in ("call", "raise") and seat not in self._vpip_counted:
                    self._vpip[seat] += 1
                    self._vpip_counted.add(seat)
                if act == "raise" and seat not in self._pfr_counted:
                    self._pfr[seat] += 1
                    self._pfr_counted.add(seat)
            else:
                if act in ("bet", "raise"):
                    self._postflop_aggr[seat] += 1
                elif act == "call":
                    self._postflop_call[seat] += 1

                # WTSD: any action on the river means they stayed in to showdown
                if street == "river" and seat not in self._wtsd_counted:
                    self._wtsd_reached[seat] += 1
                    self._wtsd_counted.add(seat)

    def classify(self, seat: int) -> str:
        """Return archetype string for a given seat (min 10 hands required)."""
        n = self._hands_played[seat]
        if n < _MIN_HANDS_CLASSIFY:
            return "unknown"

        vpip_rate   = self._vpip[seat] / n
        calls       = self._postflop_call[seat]
        aggr_factor = self._postflop_aggr[seat] / calls if calls > 0 else 2.0

        if vpip_rate < 0.20:
            return "tight_passive" if aggr_factor < 1 else "tight_aggressive"
        elif vpip_rate < 0.35:
            return "tight_aggressive" if aggr_factor >= 1 else "tight_passive"
        else:
            return "loose_aggressive" if aggr_factor >= 1.5 else "calling_station"

    def classify_table(self, players: list) -> str:
        """Dominant archetype at the table (majority vote among classified seats)."""
        archetypes = [
            self.classify(p["seat"])
            for p in players
            if p.get("seat") is not None
        ]
        if not archetypes:
            return "unknown"
        counts = Counter(a for a in archetypes if a != "unknown")
        return counts.most_common(1)[0][0] if counts else "unknown"

    def wtsd(self, seat: int) -> float:
        """
        Went-To-ShowDown rate for a seat (0.0–1.0).
        Returns 0.3 (neutral) if insufficient data.
        """
        n = self._hands_played[seat]
        if n < _MIN_HANDS_CLASSIFY:
            return 0.30   # neutral default
        return self._wtsd_reached[seat] / n

    def vpip_rate(self, seat: int) -> float:
        """VPIP rate for a seat. Returns 0.28 (neutral) if insufficient data."""
        n = self._hands_played[seat]
        if n < _MIN_HANDS_CLASSIFY:
            return 0.28
        return self._vpip[seat] / n

    def exploit_profile(self, seat: int) -> dict:
        """
        Returns a dict of scalar adjustments for downstream modules.
        All values are additive deltas unless noted.

        Keys:
          value_threshold_adj   float  — add to STRONG_HAND / MEDIUM_HAND thresholds
                                         negative = bet thinner, positive = be tighter
          bluff_threshold_adj   float  — add to BLUFF_EQUITY threshold
                                         positive = bluff less, negative = bluff more
          bet_size_multiplier   float  — multiply base bet fraction
          raise_threshold_adj   float  — add to EV RAISE_THRESHOLD
                                         positive = raise less, negative = raise more
          steal_freq_adj        float  — add to open-raise frequency in late pos
                                         positive = steal more, negative = steal less
          confidence            float  — 0.0 (no data) → 1.0 (full exploit)
        """
        n    = self._hands_played[seat]
        arch = self.classify(seat)
        wtsd = self.wtsd(seat)

        # Confidence: ramp from 0 at _MIN_HANDS_EXPLOIT to 1.0 at 23 hands
        confidence = min(1.0, max(0.0, (n - _MIN_HANDS_EXPLOIT) / 15.0))

        if arch == "unknown" or confidence == 0.0:
            return _neutral_profile()

        # Base adjustments per archetype
        if arch == "calling_station":
            base = {
                "value_threshold_adj":  -0.07,   # bet thinner — they'll call
                "bluff_threshold_adj":  +0.15,   # stop bluffing — they call everything
                "bet_size_multiplier":   1.20,   # size up for value
                "raise_threshold_adj":  -0.05,   # raise more for value
                "steal_freq_adj":       -0.05,   # don't try to steal — they call
            }
        elif arch == "tight_passive":
            base = {
                "value_threshold_adj":  +0.05,   # need a stronger hand vs tight range
                "bluff_threshold_adj":  -0.10,   # bluff more — they fold a lot
                "bet_size_multiplier":   0.85,   # smaller sizes (they fold big bets)
                "raise_threshold_adj":  +0.05,   # raise less (they only continue strong)
                "steal_freq_adj":       +0.12,   # steal aggressively
            }
        elif arch == "tight_aggressive":
            base = {
                "value_threshold_adj":  +0.05,   # respect their calling range
                "bluff_threshold_adj":  +0.05,   # reduce bluffs (they raise bluffs)
                "bet_size_multiplier":   1.00,   # standard sizing
                "raise_threshold_adj":  +0.03,   # slightly tighter raises
                "steal_freq_adj":       +0.05,   # steal a bit more vs tight range
            }
        elif arch == "loose_aggressive":
            base = {
                "value_threshold_adj":  +0.03,   # don't stack off light
                "bluff_threshold_adj":  +0.10,   # don't bluff — they raise back
                "bet_size_multiplier":   0.95,   # slightly smaller (induce raises)
                "raise_threshold_adj":  +0.05,   # be selective raising
                "steal_freq_adj":       -0.08,   # don't steal — they 3-bet light
            }
        else:
            base = _neutral_profile()

        # WTSD overlay: very high WTSD → stop bluffing even more; low WTSD → bluff more
        wtsd_bluff_adj = 0.0
        if wtsd > 0.38:
            wtsd_bluff_adj = +min(0.12, (wtsd - 0.38) * 1.5)   # call-down machine
        elif wtsd < 0.22:
            wtsd_bluff_adj = -min(0.08, (0.22 - wtsd) * 1.0)   # very foldable

        # Scale all adjustments by confidence (Bayesian drift: GTO → exploitative)
        profile = {}
        neutral = _neutral_profile()
        for key in neutral:
            if key == "confidence":
                continue
            delta = base[key] - neutral[key]
            profile[key] = neutral[key] + delta * confidence

        profile["bluff_threshold_adj"] += wtsd_bluff_adj * confidence
        profile["confidence"]           = confidence

        # Blend in early reads if we don't have full confidence yet
        if confidence < 1.0:
            early = self.early_reads(seat)
            for key in early:
                if key in profile and key != "confidence":
                    blend = early[key] * (1 - confidence)
                    profile[key] = profile[key] + blend

        return profile

    def early_reads(self, seat: int) -> dict:
        """
        Returns weak exploit signals when we have 2–4 hands of data.
        Uses raw action counts before full classification is possible.
        Acts as a Bayesian prior: small adjustments, not full exploitation.
        """
        n = self._hands_played[seat]
        if n < 2 or n >= _MIN_HANDS_CLASSIFY:
            return {}   # too little data or full model takes over

        vpip_rate   = self._vpip[seat] / n
        aggr_count  = self._postflop_aggr[seat]
        conf        = (n - 1) / (_MIN_HANDS_CLASSIFY - 1) * 0.4  # max 40% confidence

        reads = {}
        # Very high VPIP in tiny sample → likely loose, value bet more
        if vpip_rate > 0.70:
            reads["value_threshold_adj"]  = -0.04 * conf
            reads["bet_size_multiplier"]  =  1.10 * conf + (1 - conf)
            reads["bluff_threshold_adj"]  =  0.06 * conf
        # Very low VPIP → likely tight, steal more
        elif vpip_rate < 0.20:
            reads["steal_freq_adj"]       =  0.06 * conf
            reads["bluff_threshold_adj"]  = -0.06 * conf
        # Postflop aggression visible early
        if aggr_count >= 2:
            reads["bluff_threshold_adj"]  = reads.get("bluff_threshold_adj", 0) + 0.05 * conf

        return reads

    def stats_summary(self, seat: int) -> str:
        """Human-readable one-liner for logging / debug."""
        n    = self._hands_played[seat]
        arch = self.classify(seat)
        w    = self.wtsd(seat)
        v    = self.vpip_rate(seat)
        return (f"seat={seat} n={n} arch={arch} "
                f"vpip={v:.0%} wtsd={w:.0%}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _neutral_profile() -> dict:
    return {
        "value_threshold_adj":  0.0,
        "bluff_threshold_adj":  0.0,
        "bet_size_multiplier":  1.0,
        "raise_threshold_adj":  0.0,
        "steal_freq_adj":       0.0,
        "confidence":           0.0,
    }
