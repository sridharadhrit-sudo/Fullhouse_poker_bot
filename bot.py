"""
bot.py  —  Fullhouse Hackathon entry point
The engine calls decide(game_state) once per action.
Return one action dict within 2 seconds or auto-fold.

Optional: call configure_tournament() once at session start to enable
tournament mode (ICM, bubble factor, stack-depth adjustments).
"""

from preprocessor    import Preprocessor
from preflop_ranges  import PreflopRanges
from ev_calculator   import EVCalculator
from postflop        import PostflopLogic
from opponent_model  import OpponentModel
from tournament      import TournamentManager
from rl_policy       import RLPolicy

# ── Module singletons ─────────────────────────────────────────────────────────
_pre   = Preprocessor()
_pfr   = PreflopRanges()
_ev    = EVCalculator()
_post  = PostflopLogic()
_opp   = OpponentModel()
_tourn = TournamentManager(mode="cash")   # default: cash game

# ── RL policy (greedy inference — loads rl_model.pt if it exists) ─────────────
_rl = RLPolicy()
_rl_active = _rl.load()   # True if a trained model was found


# ── Optional tournament setup ─────────────────────────────────────────────────

def configure_tournament(total_players: int, paid_places: int):
    """
    Call once at the start of a tournament to enable ICM-aware decisions.
    Example:
        configure_tournament(total_players=100, paid_places=15)
    """
    _tourn.configure(total_players, paid_places)


def update_tournament_state(players_remaining: int, your_stack: int,
                             avg_stack: int, big_blind: int):
    """
    Call each hand with current tournament chip counts.
    Example (called before each hand from your harness):
        update_tournament_state(players_remaining=42, your_stack=8500,
                                avg_stack=11900, big_blind=400)
    """
    _tourn.update(players_remaining, your_stack, avg_stack, big_blind)


# ── Main entry point ──────────────────────────────────────────────────────────

def decide(game_state: dict) -> dict:
    """
    Called by the Fullhouse engine once per action.

    game_state keys (minimum required):
        your_cards       list[str]   e.g. ["As", "Kh"]
        community_cards  list[str]   e.g. ["7d", "Tc", "2s"]
        street           str         "preflop"|"flop"|"turn"|"river"
        pot              int
        your_stack       int
        amount_owed      int         0 = free check
        can_check        bool
        current_bet      int
        min_raise_to     int
        players          list
        action_log       list
    """

    # 1. Enrich game state with derived context
    ctx = _pre.process(game_state, tournament=_tourn)

    # 2. Update opponent models (new log entries only)
    _opp.update(game_state["action_log"], game_state["players"])

    # 3. Apply tournament equity premium (tighten thresholds near bubble)
    equity_premium = ctx["tournament_info"].get("equity_premium", 0.0)

    # 4. Preflop: always rule-based (GTO ranges are well-tuned;
    #    RL doesn't yet have enough training to handle preflop hand selection)
    if ctx["street"] == "preflop":
        return _validate(_pfr.decide(ctx, opp_model=_opp), game_state)

    # 5. Postflop: RL override when a sufficiently-trained model is loaded.
    #    Threshold raised to 2,000 episodes so early-training aggression
    #    bias doesn't bleed into live play.
    if _rl_active and _rl.episodes_trained >= 2000:
        rl_action = _rl.select_action_greedy(ctx, game_state)
        if rl_action is not None:
            return _validate(rl_action, game_state)

    # 6. Rule-based postflop fallback
    ev_action   = _ev.decide(ctx, opp_model=_opp)
    post_action = _post.decide(ctx, _opp)
    action      = _merge(ev_action, post_action, ctx, equity_premium)

    # 6. Safety check — never return an illegal action
    return _validate(action, game_state)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge(ev_action: dict, post_action: dict, ctx: dict,
           equity_premium: float = 0.0) -> dict:
    """
    Combine EV signal and postflop signal.

    Logic:
      - Strong hand (≥0.65): trust postflop (opponent-aware sizing)
      - River:               trust postflop (no draws left to consider)
      - Near bubble / ICM:   prefer the more conservative of the two
      - Medium equity:       flop → build pot; turn → play safe
      - Weak:                trust EV (fold/check lean)
    """
    hs     = ctx["hand_strength"]
    street = ctx["street"]

    # ICM pressure: always take the more conservative action near bubble
    if equity_premium > 0.05:
        _pri = {"fold": 0, "check": 1, "call": 2, "raise": 3, "all_in": 4}
        ev_rank   = _pri.get(ev_action.get("action",   "fold"), 0)
        post_rank = _pri.get(post_action.get("action", "fold"), 0)
        return ev_action if ev_rank <= post_rank else post_action

    if hs >= 0.65 or street == "river":
        return post_action

    if hs < 0.45:
        return ev_action

    _priority = {"fold": 0, "check": 1, "call": 2, "raise": 3, "all_in": 4}
    ev_rank   = _priority.get(ev_action.get("action",   "fold"), 0)
    post_rank = _priority.get(post_action.get("action", "fold"), 0)

    if street == "flop":
        return post_action if post_rank >= ev_rank else ev_action

    # Turn: conservative
    return ev_action if ev_rank <= post_rank else post_action


def _validate(action: dict, game_state: dict) -> dict:
    """Ensure the action is legal; default to safe fallback if not."""
    valid = {"fold", "check", "call", "raise", "all_in"}
    if action.get("action") not in valid:
        return {"action": "fold"}

    if action["action"] == "check" and not game_state["can_check"]:
        return {"action": "call"} if game_state.get("amount_owed", 0) > 0 \
               else {"action": "fold"}

    if action["action"] == "raise":
        amount = action.get("amount", 0)
        if amount < game_state["min_raise_to"]:
            action["amount"] = game_state["min_raise_to"]
        if action["amount"] >= game_state["your_stack"]:
            return {"action": "all_in"}

    return action
