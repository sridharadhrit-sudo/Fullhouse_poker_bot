"""
train.py — Self-play RL training for the poker bot.

Runs the RL policy (seat 0) against rule-based opponents in a headless loop.
Supports configurable number of players (default 4).

After each hand the REINFORCE algorithm updates the policy network based on
the chip outcome.  Progress is printed every 100 hands.

Usage:
    python3 train.py                        # 5 000 hands, 4 players
    python3 train.py --episodes 50000       # longer training run
    python3 train.py --players 6            # 6-handed game
    python3 train.py --self-play            # RL vs RL instead of vs rule-based
    python3 train.py --eval 500             # evaluate existing model (no training)
    python3 train.py --resume               # continue from saved checkpoint
    python3 train.py --stack 1000 --bb 10   # custom stack / blinds

Model is saved to rl_model.pt in the same directory as train.py.
The main bot (bot.py) loads this automatically on startup.
"""

import sys
import os
import argparse
import random
from collections import deque

import eval7

sys.path.insert(0, os.path.dirname(__file__))

from preprocessor import Preprocessor
from bot          import decide as rule_decide
from rl_policy    import RLPolicy

# ── Constants ─────────────────────────────────────────────────────────────────

RL_SEAT = 0   # RL agent always sits at seat 0

_pre = Preprocessor()


# ── Position helper ───────────────────────────────────────────────────────────

def _position(seat, dealer, n_players):
    """
    Return preflop position string ('early'/'middle'/'late') for a seat.

    Relative seat positions (0 = dealer/BTN):
      0          → late   (BTN — best postflop position)
      n-1        → late   (CO  — second-best, 4+ players)
      1          → middle (SB)
      2, 3, ...  → early  (BB, UTG, UTG+1 …)
    """
    if n_players <= 2:
        return "late"
    rel = (seat - dealer) % n_players
    if rel == 0:
        return "late"                          # BTN
    if n_players >= 4 and rel == n_players - 1:
        return "late"                          # CO
    if rel == 1:
        return "middle"                        # SB
    return "early"                             # BB, UTG, HJ, …


# ── Game-state builder (works for any seat) ───────────────────────────────────

def _make_gs(seat, hands, community, street, pot, stacks, owed,
             current_bet, min_raise, can_check, dealer, action_log, n_players):
    """Build a game_state dict from the perspective of `seat`."""
    players = [
        {
            "seat":      s,
            "stack":     stacks[s],
            "is_active": stacks[s] > 0,
            "is_me":     s == seat,
            "is_hero":   s == seat,
            "is_dealer": s == dealer,
        }
        for s in range(n_players)
    ]
    return {
        "your_cards":      hands[seat],
        "community_cards": community,
        "street":          street,
        "pot":             pot,
        "your_stack":      stacks[seat],
        "amount_owed":     owed,
        "can_check":       can_check,
        "current_bet":     current_bet,
        "min_raise_to":    min_raise,
        "position":        _position(seat, dealer, n_players),
        "num_opponents":   n_players - 1,
        "players":         players,
        "action_log":      action_log,
    }


# ── Headless betting round ────────────────────────────────────────────────────

def _betting_round(street, pot, stacks, community, dealer,
                   hands, action_log, current_bet, street_bets,
                   rl_policy, self_play, bb, n_players, active):
    """
    Run one betting round for up to n_players.

    Returns (pot, stacks, action_log, active_set).
    active_set has 1 element if everyone else folded.
    """
    bets       = dict(street_bets)
    active     = set(active)          # copy so caller's set isn't mutated
    last_raise = bb

    # ── Action order ──────────────────────────────────────────────────────────
    # Preflop: UTG first (dealer+3 for 4+ players; dealer for heads-up)
    # Postflop: SB first (dealer+1)
    if street == "preflop":
        first = dealer if n_players == 2 else (dealer + 3) % n_players
    else:
        first = (dealer + 1) % n_players

    # Build initial to-act queue in seat order starting from `first`
    to_act = []
    s = first
    for _ in range(n_players):
        if s in active:
            to_act.append(s)
        s = (s + 1) % n_players

    while to_act:
        seat = to_act.pop(0)
        if seat not in active:
            continue
        if stacks[seat] == 0:          # all-in, can't act
            continue

        owed      = max(0, current_bet - bets.get(seat, 0))
        owed      = min(owed, stacks[seat])
        can_check = owed == 0
        min_raise = min(current_bet + last_raise,
                        stacks[seat] + bets.get(seat, 0))

        # ── Get action ────────────────────────────────────────────────────────
        gs = _make_gs(seat, hands, community, street, pot, stacks,
                      owed, current_bet, min_raise, can_check,
                      dealer, action_log[:], n_players)

        if seat == RL_SEAT:
            if street == "preflop":
                # Preflop always rule-based; RL only trains postflop
                action = rule_decide(gs)
            else:
                ctx    = _pre.process(gs)
                action = rl_policy.select_action(ctx, gs)
                if action is None:
                    action = {"action": "fold"}
        else:
            if self_play:
                ctx    = _pre.process(gs)
                action = rl_policy.select_action_greedy(ctx, gs) or {"action": "fold"}
            else:
                action = rule_decide(gs)

        # ── Apply action ──────────────────────────────────────────────────────
        act = action.get("action", "fold")

        if act == "fold":
            active.discard(seat)
            action_log.append({"seat": seat, "action": "fold", "street": street})
            if len(active) == 1:
                break

        elif act == "check":
            action_log.append({"seat": seat, "action": "check", "street": street})

        elif act == "call":
            put_in        = owed
            stacks[seat] -= put_in
            bets[seat]    = bets.get(seat, 0) + put_in
            pot          += put_in
            action_log.append({"seat": seat, "action": "call",
                                "amount": put_in, "street": street})

        elif act in ("raise", "all_in"):
            total_to  = (stacks[seat] + bets.get(seat, 0)) if act == "all_in" \
                        else max(action.get("amount", min_raise), min_raise)
            total_to      = min(total_to, stacks[seat] + bets.get(seat, 0))
            put_in        = total_to - bets.get(seat, 0)
            put_in        = min(put_in, stacks[seat])
            last_raise    = max(total_to - current_bet, bb)
            stacks[seat] -= put_in
            bets[seat]    = bets.get(seat, 0) + put_in
            pot          += put_in
            current_bet   = bets[seat]
            action_log.append({"seat": seat,
                                "action": "raise" if act == "raise" else "all_in",
                                "amount": current_bet, "street": street})
            # Re-open action for all active players not already queued
            for other in range(n_players):
                if other in active and other != seat and other not in to_act:
                    to_act.append(other)

    return pot, stacks, action_log, active


# ── Single hand ───────────────────────────────────────────────────────────────

def run_hand(stacks, dealer, rl_policy, self_play, sb_size, bb_size, n_players):
    """
    Simulate one complete hand with n_players.
    Returns (new_stacks, chip_delta_for_rl).
    """
    deck = eval7.Deck()
    deck.shuffle()
    all_cards = [str(c) for c in deck.cards]

    # Deal 2 hole cards per player, then 5 board cards
    hands      = {s: all_cards[s * 2: s * 2 + 2] for s in range(n_players)}
    full_board = all_cards[n_players * 2: n_players * 2 + 5]

    community    = []
    action_log   = []
    pot          = 0
    start_stacks = dict(stacks)
    active       = set(range(n_players))

    # ── Blinds ────────────────────────────────────────────────────────────────
    sb_seat = (dealer + 1) % n_players
    bb_seat = (dealer + 2) % n_players
    sb_post = min(sb_size, stacks[sb_seat])
    bb_post = min(bb_size, stacks[bb_seat])
    stacks[sb_seat] -= sb_post
    stacks[bb_seat] -= bb_post
    pot += sb_post + bb_post

    blind_bets           = {s: 0 for s in range(n_players)}
    blind_bets[sb_seat]  = sb_post
    blind_bets[bb_seat]  = bb_post
    action_log += [
        {"seat": sb_seat, "action": "blind", "amount": sb_post, "street": "preflop"},
        {"seat": bb_seat, "action": "blind", "amount": bb_post, "street": "preflop"},
    ]

    # ── Street loop ───────────────────────────────────────────────────────────
    streets = [
        ("preflop", [],              bb_size, blind_bets),
        ("flop",    full_board[:3],  0,       None),
        ("turn",    full_board[3:4], 0,       None),
        ("river",   full_board[4:5], 0,       None),
    ]

    for street, new_cards, init_bet, init_bets in streets:
        community.extend(new_cards)

        if len(active) <= 1:
            break

        # If only one player has chips left, skip betting
        can_act = [s for s in active if stacks[s] > 0]
        if len(can_act) <= 1 and street != "preflop":
            continue

        pot, stacks, action_log, active = _betting_round(
            street, pot, stacks, community, dealer,
            hands, action_log,
            current_bet  = init_bet,
            street_bets  = init_bets or {s: 0 for s in range(n_players)},
            rl_policy    = rl_policy,
            self_play    = self_play,
            bb           = bb_size,
            n_players    = n_players,
            active       = active,
        )

        if len(active) == 1:
            winner = next(iter(active))
            stacks[winner] += pot
            return stacks, stacks[RL_SEAT] - start_stacks[RL_SEAT]

    # ── Showdown ──────────────────────────────────────────────────────────────
    # Simplified: award pot to best hand among active players.
    # Side-pot edge cases (all-ins of different sizes) are approximated —
    # accurate enough for training signal purposes.
    best_score  = -1
    best_seats  = []
    for s in active:
        cards = [eval7.Card(c) for c in hands[s] + community]
        score = eval7.evaluate(cards)
        if score > best_score:
            best_score = score
            best_seats = [s]
        elif score == best_score:
            best_seats.append(s)

    share = pot // len(best_seats)
    for s in best_seats:
        stacks[s] += share
    # Any rounding remainder goes to the first winner
    stacks[best_seats[0]] += pot - share * len(best_seats)

    return stacks, stacks[RL_SEAT] - start_stacks[RL_SEAT]


# ── Evaluation (no gradient updates) ─────────────────────────────────────────

def evaluate(rl_policy, n_hands, stack, sb, bb, self_play, n_players):
    """Run n_hands without training. Returns RL win rate and avg BB/hand."""
    stacks  = {s: stack for s in range(n_players)}
    dealer  = RL_SEAT
    wins    = 0
    total_delta = 0

    for _ in range(n_hands):
        if any(stacks[s] <= 0 for s in range(n_players)):
            stacks = {s: stack for s in range(n_players)}

        new_stacks, delta = run_hand(stacks, dealer, rl_policy,
                                     self_play=self_play,
                                     sb_size=sb, bb_size=bb,
                                     n_players=n_players)
        stacks       = new_stacks
        dealer       = (dealer + 1) % n_players
        total_delta += delta
        if delta > 0:
            wins += 1

    win_rate = wins / n_hands * 100
    bb_per_h = total_delta / n_hands / bb
    return win_rate, bb_per_h


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    sb        = args.sb
    bb        = args.bb
    stack     = args.stack
    n_players = args.players

    temp_start = 1.5
    temp_end   = 0.5

    policy = RLPolicy(lr=args.lr, temperature=temp_start)

    if not policy.is_ready:
        print("PyTorch is required for RL training.  Exiting.")
        return

    if args.resume:
        policy.load()

    print(f"\n{'═'*60}")
    print(f"  RL Training — {'Self-play' if args.self_play else 'vs Rule-based bot'}")
    print(f"  Players:   {n_players}  |  RL seat: {RL_SEAT}")
    print(f"  Episodes:  {args.episodes:,}")
    print(f"  Stack:     {stack}  |  Blinds: {sb}/{bb}")
    print(f"  LR:        {args.lr}  |  Temp: {temp_start} → {temp_end}")
    print(f"{'═'*60}\n")

    stacks        = {s: stack for s in range(n_players)}
    dealer        = RL_SEAT
    recent_deltas = deque(maxlen=200)
    wins_200      = 0
    save_every    = max(100, args.episodes // 50)

    for ep in range(1, args.episodes + 1):

        # Restock if any player busted
        if any(stacks[s] <= 0 for s in range(n_players)):
            stacks = {s: stack for s in range(n_players)}

        # DQN uses epsilon-greedy internally; temperature schedule kept for
        # API compatibility but has no effect on DQN training
        policy.temperature = (temp_start
                              + (temp_end - temp_start) * (ep / args.episodes))

        # Run one hand
        policy.start_episode()
        new_stacks, delta = run_hand(stacks, dealer, policy,
                                     self_play=args.self_play,
                                     sb_size=sb, bb_size=bb,
                                     n_players=n_players)

        reward = delta / bb
        policy.finish_episode(reward)

        stacks = new_stacks
        dealer = (dealer + 1) % n_players

        # Track recent performance
        if len(recent_deltas) == 200 and recent_deltas[0] > 0:
            wins_200 -= 1
        recent_deltas.append(delta)
        if delta > 0:
            wins_200 += 1

        # ── Progress report every 100 hands ───────────────────────────────────
        if ep % 100 == 0:
            win_pct  = wins_200 / min(200, ep) * 100
            avg_bb   = sum(recent_deltas) / len(recent_deltas) / bb
            print(f"  ep {ep:>6,} | win% {win_pct:5.1f} | "
                  f"avg {avg_bb:+.2f} BB/h | "
                  f"ε={policy.epsilon:.3f} | "
                  f"grad_eps {policy.episodes_trained:,}")

        # ── Save checkpoint ───────────────────────────────────────────────────
        if ep % save_every == 0:
            policy.save()
            print(f"  [saved checkpoint at episode {ep:,}]")

    # ── Final save + evaluation ───────────────────────────────────────────────
    policy.save()
    print(f"\n  Training complete — {args.episodes:,} hands.")
    print(f"  Gradient updates (postflop decisions): {policy.episodes_trained:,}")
    print(f"  Running final evaluation ({args.eval_hands} hands)...")

    eval_policy = RLPolicy(temperature=0.1)
    eval_policy.load()
    wr, bb_h = evaluate(eval_policy, args.eval_hands, stack, sb, bb,
                        args.self_play, n_players)
    print(f"\n  Eval result: {wr:.1f}% win rate  |  {bb_h:+.2f} BB/hand  "
          f"(vs {'self' if args.self_play else 'rule-based bot'})")
    print(f"  Model saved → rl_model.pt\n")


# ── Evaluation-only mode ──────────────────────────────────────────────────────

def eval_only(args):
    policy = RLPolicy(temperature=0.1)
    if not policy.load():
        print("No trained model found (rl_model.pt).  Train first.")
        return

    print(f"\n  Evaluating over {args.eval} hands ({args.players} players)...")
    wr, bb_h = evaluate(policy, args.eval, args.stack,
                        args.sb, args.bb, args.self_play, args.players)
    print(f"  Win rate : {wr:.1f}%")
    print(f"  BB/hand  : {bb_h:+.2f}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the RL poker policy")
    parser.add_argument("--episodes",   type=int,   default=5_000,
                        help="Number of training hands (default 5000)")
    parser.add_argument("--players",    type=int,   default=4,
                        help="Number of players at the table (default 4)")
    parser.add_argument("--stack",      type=int,   default=1_000,
                        help="Starting stack size (default 1000)")
    parser.add_argument("--bb",         type=int,   default=10,
                        help="Big blind size (default 10)")
    parser.add_argument("--sb",         type=int,   default=5,
                        help="Small blind size (default 5)")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate (default 3e-4)")
    parser.add_argument("--self-play",  action="store_true",
                        help="RL vs RL self-play instead of vs rule-based bot")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--eval",       type=int,   default=0,
                        help="Evaluate existing model over N hands (no training)")
    parser.add_argument("--eval-hands", type=int,   default=500,
                        help="Hands used for final evaluation (default 500)")
    args = parser.parse_args()

    if args.eval > 0:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
