"""
hand_logger.py — Records every hand to logs/ for later analysis.

Each session creates one JSON file:  logs/session_YYYYMMDD_HHMMSS.json
The file is a JSON array of hand objects, written after every hand.

Hand object shape:
{
  "hand_num":     1,
  "timestamp":    "2026-04-28T15:30:00",
  "dealer":       0,
  "blinds":       [5, 10],
  "stacks_start": {"0": 1000, "1": 1000},
  "hole_cards":   {"0": ["Ah", "Kd"], "1": ["7c", "2s"]},
  "streets": [
    {
      "street":    "preflop",
      "board":     [],
      "actions": [
        {"seat": 0, "action": "raise", "amount": 25, "pot_before": 15}
      ]
    },
    ...
  ],
  "result": {
    "method":      "fold" | "showdown",
    "winner":      0,
    "pot":         50,
    "hand_ranks":  {"0": "Pair", "1": "High Card"},   (showdown only)
    "stacks_end":  {"0": 1050, "1": 950}
  }
}
"""

import json
import os
from datetime import datetime
from pathlib import Path


class HandLogger:

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        stamp             = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path        = self.log_dir / f"session_{stamp}.json"
        self._hands: list = []
        self._current     = None

    # ── Hand lifecycle ────────────────────────────────────────────────────────

    def start_hand(self, hand_num: int, stacks: dict, dealer: int, blinds: tuple):
        self._current = {
            "hand_num":     hand_num,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "dealer":       dealer,
            "blinds":       list(blinds),
            "stacks_start": {str(k): v for k, v in stacks.items()},
            "hole_cards":   {},
            "streets":      [],
            "result":       None,
            "_current_street": None,
        }

    def record_hole_cards(self, seat: int, cards: list):
        if self._current:
            self._current["hole_cards"][str(seat)] = cards

    def start_street(self, street: str, board: list):
        if not self._current:
            return
        self._current["_current_street"] = {
            "street":  street,
            "board":   list(board),
            "actions": [],
        }

    def record_action(self, seat: int, action: str, amount: int, pot_before: int):
        if not self._current or not self._current["_current_street"]:
            return
        self._current["_current_street"]["actions"].append({
            "seat":       seat,
            "action":     action,
            "amount":     amount,
            "pot_before": pot_before,
        })

    def end_street(self):
        if self._current and self._current["_current_street"]:
            s = self._current.pop("_current_street")
            self._current["streets"].append(s)

    def end_hand(self, winner: int, pot: int, stacks: dict,
                 method: str, hand_ranks: dict = None):
        """
        method: "fold" | "showdown"
        hand_ranks: {seat: rank_string}  (showdown only)
        """
        if not self._current:
            return

        # Close any open street
        if self._current.get("_current_street"):
            self.end_street()

        self._current["result"] = {
            "method":    method,
            "winner":    winner,
            "pot":       pot,
            "stacks_end": {str(k): v for k, v in stacks.items()},
        }
        if hand_ranks:
            self._current["result"]["hand_ranks"] = {
                str(k): v for k, v in hand_ranks.items()
            }

        self._hands.append(self._current)
        self._current = None
        self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._hands, f, indent=2)

    @property
    def session_file(self) -> str:
        return str(self._path)
