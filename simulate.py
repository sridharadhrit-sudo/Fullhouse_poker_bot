"""
simulate.py — Play heads-up No-Limit Texas Hold'em against the bot.

Usage:
    python simulate.py                          # cash game, play until bust
    python simulate.py --hands 20               # stop after 20 hands
    python simulate.py --stack 2000             # custom starting stack
    python simulate.py --tournament             # enable tournament mode
    python simulate.py --tournament --players 50 --paid 8

During play:
    check | call | fold | raise <amount> | all-in
"""

import argparse

import eval7

from bot        import decide, configure_tournament, update_tournament_state
from hand_logger import HandLogger

SB          = 5
BB          = 10
START_STACK = 1000
HUMAN       = 0
BOT         = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Card helpers
# ═══════════════════════════════════════════════════════════════════════════════

def deal_cards():
    deck = eval7.Deck()
    deck.shuffle()
    c = [str(x) for x in deck.cards]
    return c[:2], c[2:4], c[4:9]


def hand_rank(hole, board):
    cards = [eval7.Card(c) for c in hole + board]
    score = eval7.evaluate(cards)
    return score, eval7.handtype(score)


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

W = 58

def hr(char="─"):
    print(char * W)

def show_table(community, human_hand, stacks, pot, street,
               reveal_bot=False, bot_hand=None):
    hr()
    board = " ".join(community) if community else "(waiting)"
    print(f"  {street.upper():<10}  Pot: {pot:<6}  Board: {board}")
    hr("─")
    print(f"  You  [{' '.join(human_hand)}]   Stack: {stacks[HUMAN]}")
    if reveal_bot and bot_hand:
        print(f"  Bot  [{' '.join(bot_hand)}]   Stack: {stacks[BOT]}")
    else:
        print(f"  Bot  [?? ??]          Stack: {stacks[BOT]}")
    hr()

def announce_bot(action, stacks):
    act = action["action"]
    if act == "raise":
        print(f"    Bot  ▶  RAISES to {action['amount']}")
    elif act == "all_in":
        print(f"    Bot  ▶  ALL-IN  ({stacks[BOT]} chips)")
    else:
        print(f"    Bot  ▶  {act.upper()}")


# ═══════════════════════════════════════════════════════════════════════════════
# Human input
# ═══════════════════════════════════════════════════════════════════════════════

def prompt_human(pot, stacks, owed, can_check, min_raise,
                 community, human_hand, street):
    show_table(community, human_hand, stacks, pot, street)
    opts = []
    if can_check:
        opts.append("check")
    if owed > 0:
        opts.append(f"call {owed}")
    if stacks[HUMAN] > owed:
        opts.append(f"raise <n>  (min {min_raise})")
    opts.append(f"all-in ({stacks[HUMAN]})")
    opts.append("fold")
    print("  " + "  |  ".join(opts))

    while True:
        raw    = input("  Your move: ").strip().lower()
        tokens = raw.split()
        cmd    = tokens[0] if tokens else ""

        if cmd == "check" and can_check:
            return {"action": "check"}
        if cmd in ("call", "c") and owed > 0:
            return {"action": "call"}
        if cmd == "fold":
            return {"action": "fold"}
        if cmd in ("all-in", "allin", "shove", "a"):
            return {"action": "all_in"}
        if cmd in ("raise", "r", "bet", "b"):
            if len(tokens) == 2:
                try:
                    amt = int(tokens[1])
                    if amt < min_raise:
                        print(f"  Minimum raise is {min_raise}.")
                    elif amt > stacks[HUMAN]:
                        print(f"  You only have {stacks[HUMAN]} chips.")
                    else:
                        return {"action": "raise", "amount": amt}
                    continue
                except ValueError:
                    pass
            print(f"  Usage: raise <amount>   (min {min_raise})")
            continue
        print("  Unrecognised. Try: check / call / raise <n> / all-in / fold")


# ═══════════════════════════════════════════════════════════════════════════════
# Bot game-state builder
# ═══════════════════════════════════════════════════════════════════════════════

def make_bot_state(bot_hand, community, street, pot, stacks,
                   owed, current_bet, min_raise, can_check,
                   dealer, action_log):
    return {
        "your_cards":      bot_hand,
        "community_cards": community,
        "street":          street,
        "pot":             pot,
        "your_stack":      stacks[BOT],
        "amount_owed":     owed,
        "can_check":       can_check,
        "current_bet":     current_bet,
        "min_raise_to":    min_raise,
        "players": [
            {"seat": HUMAN, "stack": stacks[HUMAN], "is_active": True,
             "is_me": False, "is_dealer": dealer == HUMAN},
            {"seat": BOT,   "stack": stacks[BOT],   "is_active": True,
             "is_me": True, "is_hero": True, "is_dealer": dealer == BOT},
        ],
        "action_log": action_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Betting round
# ═══════════════════════════════════════════════════════════════════════════════

def betting_round(street, pot, stacks, community, dealer,
                  human_hand, bot_hand, action_log,
                  current_bet=0, street_bets=None, logger=None):
    bets      = dict(street_bets) if street_bets else {HUMAN: 0, BOT: 0}
    active    = {HUMAN, BOT}
    order     = [dealer, 1-dealer] if street == "preflop" else [1-dealer, dealer]
    to_act    = list(order)
    last_raise = BB

    if logger:
        logger.start_street(street, community)

    while to_act:
        seat = to_act.pop(0)
        if seat not in active or stacks[seat] == 0:
            continue

        owed      = max(0, current_bet - bets[seat])
        owed      = min(owed, stacks[seat])
        can_check = owed == 0
        min_raise = min(current_bet + last_raise, stacks[seat] + bets[seat])

        if seat == HUMAN:
            action = prompt_human(pot, stacks, owed, can_check, min_raise,
                                  community, human_hand, street)
        else:
            gs = make_bot_state(bot_hand, community, street, pot, stacks,
                                owed, current_bet, min_raise, can_check,
                                dealer, action_log[:])
            action = decide(gs)
            announce_bot(action, stacks)

        act = action["action"]

        if logger:
            logger.record_action(seat, act,
                                 action.get("amount", owed if act=="call" else 0),
                                 pot)

        if act == "fold":
            active.discard(seat)
            action_log.append({"seat": seat, "action": "fold", "street": street})
            break

        elif act == "check":
            action_log.append({"seat": seat, "action": "check", "street": street})

        elif act == "call":
            put_in       = owed
            stacks[seat] -= put_in
            bets[seat]   += put_in
            pot          += put_in
            action_log.append({"seat": seat, "action": "call",
                                "amount": put_in, "street": street})

        elif act in ("raise", "all_in"):
            total_to = (stacks[seat] + bets[seat]) if act == "all_in" \
                       else max(action.get("amount", min_raise), min_raise)
            total_to     = min(total_to, stacks[seat] + bets[seat])
            put_in       = total_to - bets[seat]
            put_in       = min(put_in, stacks[seat])
            last_raise   = max(total_to - current_bet, BB)
            stacks[seat] -= put_in
            bets[seat]   += put_in
            pot          += put_in
            current_bet  = bets[seat]
            action_log.append({"seat": seat,
                                "action": "raise" if act == "raise" else "all_in",
                                "amount": current_bet, "street": street})
            for other in [HUMAN, BOT]:
                if other in active and other != seat and other not in to_act:
                    to_act.append(other)

    if logger:
        logger.end_street()

    folded = HUMAN if HUMAN not in active else (BOT if BOT not in active else None)
    return pot, stacks, action_log, folded


# ═══════════════════════════════════════════════════════════════════════════════
# Single hand
# ═══════════════════════════════════════════════════════════════════════════════

def play_hand(stacks, dealer, hand_num, logger=None,
              tournament_mode=False, players_remaining=2):
    print(f"\n{'═' * W}")
    mode_tag = f"  [Tournament — {players_remaining} left]" if tournament_mode else ""
    print(f"  Hand #{hand_num}   Dealer: {'You' if dealer == HUMAN else 'Bot'}"
          f"   Blinds: {SB}/{BB}{mode_tag}")
    print(f"  Your stack: {stacks[HUMAN]}   Bot stack: {stacks[BOT]}")
    print(f"{'═' * W}")

    human_hand, bot_hand, full_board = deal_cards()
    start_stacks = dict(stacks)
    action_log   = []
    community    = []
    pot          = 0

    if logger:
        logger.start_hand(hand_num, stacks, dealer, (SB, BB))
        logger.record_hole_cards(HUMAN, human_hand)
        logger.record_hole_cards(BOT,   bot_hand)

    # ── Blinds ───────────────────────────────────────────────────────────────
    sb, bb  = dealer, 1 - dealer
    sb_post = min(SB, stacks[sb])
    bb_post = min(BB, stacks[bb])
    stacks[sb] -= sb_post
    stacks[bb] -= bb_post
    pot          += sb_post + bb_post
    blind_bets   = {sb: sb_post, bb: bb_post}
    action_log  += [
        {"seat": sb, "action": "blind", "amount": sb_post, "street": "preflop"},
        {"seat": bb, "action": "blind", "amount": bb_post, "street": "preflop"},
    ]
    print(f"  Blinds: SB {sb_post}  BB {bb_post}")
    print(f"  Your cards: {' '.join(human_hand)}")

    # ── Update tournament state ───────────────────────────────────────────────
    if tournament_mode:
        avg = (stacks[HUMAN] + stacks[BOT]) // 2
        update_tournament_state(
            players_remaining, stacks[BOT], avg, BB
        )

    # ── Streets ───────────────────────────────────────────────────────────────
    streets = [
        ("preflop", [],             BB,  blind_bets),
        ("flop",    full_board[:3], 0,   None),
        ("turn",    full_board[3:4], 0,  None),
        ("river",   full_board[4:5], 0,  None),
    ]

    for street, new_cards, init_bet, init_bets in streets:
        if new_cards:
            community.extend(new_cards)
            print(f"\n  ── {street.upper()} ──  Board: {' '.join(community)}")

        both_can_bet = stacks[HUMAN] > 0 and stacks[BOT] > 0
        if not both_can_bet and street != "preflop":
            continue

        pot, stacks, action_log, folded = betting_round(
            street, pot, stacks, community, dealer,
            human_hand, bot_hand, action_log,
            current_bet=init_bet, street_bets=init_bets,
            logger=logger,
        )

        if folded is not None:
            winner = 1 - folded
            label  = "You win!" if winner == HUMAN else "Bot wins."
            stacks[winner] += pot
            name   = "Bot" if folded == BOT else "You"
            print(f"\n  {name} fold.  {label}  (+{pot})")
            hr("─")
            print(f"  Your cards : {' '.join(human_hand)}")
            print(f"  Bot cards  : {' '.join(bot_hand)}")
            if community:
                print(f"  Board      : {' '.join(community)}")
            runout = full_board[len(community):]
            if runout:
                runout_labels = ["turn", "turn", "river", "river", "river"]
                # label each remaining card by its street
                if len(runout) == 1:
                    print(f"  Run-out    : [{runout[0]}]  ← river")
                elif len(runout) == 2:
                    print(f"  Run-out    : [{runout[0]}]  ← turn   [{runout[1]}]  ← river")
                else:
                    flop_r  = " ".join(runout[:3])
                    rest    = runout[3:]
                    line    = f"  Run-out    : [{flop_r}]  ← flop"
                    if len(rest) >= 1:
                        line += f"   [{rest[0]}]  ← turn"
                    if len(rest) >= 2:
                        line += f"   [{rest[1]}]  ← river"
                    print(line)
            hr("─")

            if logger:
                logger.end_hand(winner, pot, stacks, "fold")
            return stacks, 1 - dealer

    # ── Showdown ──────────────────────────────────────────────────────────────
    h_score, h_rank = hand_rank(human_hand, community)
    b_score, b_rank = hand_rank(bot_hand,   community)

    hr("─")
    print(f"  ── SHOWDOWN ──  Board: {' '.join(community)}")
    hr("─")
    print(f"  Your cards : {' '.join(human_hand)}   ({h_rank})")
    print(f"  Bot cards  : {' '.join(bot_hand)}   ({b_rank})")
    hr("─")

    # Side-pot resolution
    human_in = start_stacks[HUMAN] - stacks[HUMAN]
    bot_in   = start_stacks[BOT]   - stacks[BOT]
    excess   = abs(human_in - bot_in)
    if excess:
        overpayer = HUMAN if human_in > bot_in else BOT
        stacks[overpayer] += excess
        pot -= excess

    if h_score > b_score:
        stacks[HUMAN] += pot
        winner = HUMAN
        print(f"  ✓ You win {pot} chips!")
    elif b_score > h_score:
        stacks[BOT] += pot
        winner = BOT
        print(f"  Bot wins {pot} chips.")
    else:
        half = pot // 2
        stacks[HUMAN] += half
        stacks[BOT]   += pot - half
        winner = -1
        print(f"  Split pot ({half} each).")

    if logger:
        logger.end_hand(winner, pot, stacks, "showdown",
                        hand_ranks={HUMAN: h_rank, BOT: b_rank})
    return stacks, 1 - dealer


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Play heads-up NLHE against the poker bot."
    )
    parser.add_argument("--hands",      type=int, default=0)
    parser.add_argument("--stack",      type=int, default=START_STACK)
    parser.add_argument("--tournament", action="store_true",
                        help="Enable tournament mode (ICM awareness)")
    parser.add_argument("--players",    type=int, default=50,
                        help="Total players at tournament start (default 50)")
    parser.add_argument("--paid",       type=int, default=8,
                        help="Paid places in tournament (default 8)")
    parser.add_argument("--no-log",     action="store_true",
                        help="Disable hand history logging")
    args = parser.parse_args()

    # ── Tournament setup ──────────────────────────────────────────────────────
    if args.tournament:
        configure_tournament(args.players, args.paid)
        print(f"\n  Tournament mode: {args.players} players, top {args.paid} paid")

    # ── Logger setup ─────────────────────────────────────────────────────────
    logger = None if args.no_log else HandLogger()
    if logger:
        print(f"  Logging to: {logger.session_file}")

    stacks            = {HUMAN: args.stack, BOT: args.stack}
    dealer            = HUMAN
    hand_num          = 1
    players_remaining = args.players if args.tournament else 2

    print("\n" + "═" * W)
    print("  Heads-Up Texas No-Limit Hold'em  —  You vs the Bot")
    print(f"  Starting stacks: {args.stack}  |  Blinds: {SB}/{BB}")
    print(f"  Commands: check  call  fold  raise <n>  all-in")
    print("═" * W)

    while True:
        if stacks[HUMAN] <= 0 or stacks[BOT] <= 0:
            break
        if args.hands and hand_num > args.hands:
            break

        stacks, dealer = play_hand(
            stacks, dealer, hand_num, logger=logger,
            tournament_mode=args.tournament,
            players_remaining=players_remaining,
        )
        hand_num += 1

        if stacks[HUMAN] <= 0:
            print("\n  You're out of chips — bot wins.")
            break
        if stacks[BOT] <= 0:
            print("\n  Bot is out of chips — you win!")
            break

        again = input("\n  Next hand? [Y/n]: ").strip().lower()
        if again in ("n", "no"):
            break

    hr("═")
    diff = stacks[HUMAN] - stacks[BOT]
    print(f"  Final — You: {stacks[HUMAN]}   Bot: {stacks[BOT]}")
    print(f"  {'Up' if diff>=0 else 'Down'} {abs(diff)} chips overall.")
    if logger:
        print(f"  Hand history saved → {logger.session_file}")
        print(f"  Review with:  python analyse.py {logger.session_file}")
    hr("═")


if __name__ == "__main__":
    main()
