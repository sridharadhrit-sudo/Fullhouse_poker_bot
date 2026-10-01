# Fullhouse Poker Bot

A Python no-limit Texas Hold'em bot built for the [Fullhouse Hackathon](https://fullhousehackathon.com) — the UK's first quantitative poker bot competition (1–5 June 2026, London, sponsored by Quadrature Capital).

The bot plays a hybrid strategy: GTO-frequency preflop ranges, EV/heuristic postflop logic, a trained DQN (deep Q-network) policy that can override postflop play once sufficiently trained, live opponent modelling, and tournament-aware (ICM) adjustments.

## How it plugs into the engine

The [Fullhouse engine](https://github.com/uzlez/fullhouse-engine) calls one function once per action:

```python
def decide(game_state: dict) -> dict:
    ...
    return {"action": "raise", "amount": 120}
```

Everything below is what happens inside that call.

## Architecture

```
game_state
    │
    ▼
Preprocessor ──► hand strength (eval7 Monte Carlo), pot odds, SPR,
 (+ RangeModel)    position, board texture, tournament context
    │ 
    ▼
┌─────────────────────────────┬──────────────────────────────┐
│  Preflop (preflop-only)     │  Postflop (flop/turn/river)  │
│  GTO ranges from            │  DQN policy (if trained &    │
│  gto_ranges.json, mixed     │  confident) OR EV calculator │
│  strategy sampling          │  + rule-based postflop logic,│
│                              │  merged by street/equity     │
└─────────────────────────────┴──────────────────────────────┘
    │
    ▼
Opponent model (VPIP/PFR/AF/WTSD → archetype) shifts thresholds
and GTO frequencies toward an exploitative strategy over time
    │
    ▼
Validator ─► guarantees a legal action (fold/check/call/raise/all-in)
```

### Modules

| File | Role |
|---|---|
| `bot.py` | Entry point — `decide()`, tournament config hooks, signal merging, legality validation |
| `preprocessor.py` | Layer 1 — hand strength, pot odds, SPR, position, board texture, tournament context |
| `range_model.py` | Range advantage, blocker scoring, opponent range-tightness inference |
| `preflop_ranges.py` | GTO-frequency preflop decisions (169 hand types × position × scenario) from `gto_ranges.json` |
| `ev_calculator.py` | Expected-value postflop decisions, multi-way adjusted, exploit-aware raise sizing |
| `postflop.py` | Rule-based postflop logic — c-betting, value/pot-control/semi-bluff/give-up tree |
| `opponent_model.py` | Tracks VPIP, PFR, AF, WTSD per seat; classifies opponents into 4 archetypes and produces an exploit profile |
| `rl_policy.py` | DQN policy: 22-feature state → 9 discrete actions (fold through all-in sizings), with replay buffer, target network and epsilon-greedy exploration |
| `train.py` | Self-play / vs-rule-based training loop that produces `rl_model.pt` |
| `tournament.py` | ICM-aware mode — stage detection (early/middle/bubble/final table) and stack-depth adjustments |
| `hand_logger.py` | Writes every hand to `logs/session_*.json` for later analysis |
| `simulate.py` | Interactive CLI — play heads-up against the bot yourself |
| `gto_ranges.json` | Editable preflop frequency tables (no code changes needed to retune ranges) |

### Decision flow (`bot.py`)

1. **Preflop** — always rule-based GTO ranges (well-tuned; RL doesn't yet have enough training signal for preflop hand selection).
2. **Postflop** — the DQN policy takes over once trained past a confidence threshold (2,000+ episodes), otherwise falls back to a merge of the EV calculator and rule-based postflop logic, weighted by hand strength, street, and ICM pressure near the bubble.
3. Every action passes through a validator that guarantees legality (correct min-raise, all-in conversion, etc.) before being returned to the engine.

## Getting started

```bash
# engine dependencies
pip3 install "Cython<3"
pip3 install --no-build-isolation eval7==0.1.7
pip3 install numpy scipy treys

# optional: RL policy
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

**Play against the bot:**
```bash
python3 simulate.py                 # cash game, play until bust
python3 simulate.py --hands 20      # stop after 20 hands
python3 simulate.py --tournament --players 50 --paid 8
```

**Train the RL policy:**
```bash
python3 train.py                        # 5,000 hands, 4 players
python3 train.py --episodes 50000       # longer run
python3 train.py --resume               # continue from checkpoint
python3 train.py --eval 500             # evaluate only, no training
```

Trained weights are saved to `rl_model.pt`, which `bot.py` loads automatically on startup.

## Competition constraints

- 2-second decision limit inside `decide()`
- No network calls or file I/O during play
- Allowed libraries: `eval7`, `numpy`, `scipy`, `treys`

## License

MIT
