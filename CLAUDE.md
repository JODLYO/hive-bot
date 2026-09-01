# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AlphaZero-style bot for the board game Hive: a fast game engine, MCTS
guided by a PyTorch policy/value network, and a self-play training loop run
from a Google Colab notebook. Standalone from the `ttbg-web-app` Django repo
that hosts the actual playable game (a sibling directory, not a dependency).

## Commands

```
make setup       # uv sync --group dev
make lint         # ruff check + ruff format --check
make format       # ruff format + ruff check --fix
make typecheck    # mypy src
make test         # fast suite (~seconds) -- what to run routinely
make test-full    # exhaustive engine-vs-reference-oracle parity gate (~10 min)
                   # -- run before trusting ANY change to src/hive_bot/engine
make selfplay     # tiny network, few games -- smoke test the self-play path
make train        # tiny network, few iterations -- smoke test the train loop
```

Single test: `uv run pytest tests/test_engine_rules.py::test_move_cap_forces_a_draw`.
Slow/property tests are marked `@pytest.mark.slow` and excluded by default
(`addopts = "-m 'not slow'"` in pyproject.toml); `make test-full` runs just
those.

**Known environment flakiness**: the editable install's `.pth` mechanism has
been observed to silently stop resolving `hive_bot` (Python 3.14 + uv +
hatchling), independent of anything in this project's code. `conftest.py`
(sys.path insert) and the Makefile's `export PYTHONPATH := src` are
belt-and-suspenders fixes for this. If `ModuleNotFoundError: No module named
'hive_bot'` shows up somewhere new, don't chase it as a real bug -- add the
same `src` path fix, or just re-run `uv sync --group dev` (sometimes
`rm -rf .venv && uv sync --group dev`).

## Architecture

### Two parallel rule engines, on purpose

`tests/reference/` vendors an unmodified copy of the Django app's
`game_state.py`/`helpers.py` (pydantic-based, deep-copies the board per
candidate move) as a **correctness oracle**. `tests/reference/oracle.py` is
NOT a copy -- it's a from-scratch re-transcription of the move-orchestration
logic that lives on the Django `GameState` model in the source app (which
can't be vendored as-is since it's entangled with the ORM). It exists purely
so `test_engine_vs_reference.py` can enumerate "every legal move" from the
oracle's side to compare against the fast engine.

`src/hive_bot/engine/` is the real, fast engine used everywhere else (plain
dict/tuple/int state, no pydantic, make/unmake move application instead of
copying). It reimplements the same rules independently rather than reusing
the reference, because MCTS self-play needs to apply/revert thousands of
moves per second. Any engine change must pass `make test-full` before being
trusted -- that's the whole reason the oracle comparison exists.

Two draw rules (`MAX_PLIES_BEFORE_DRAW`, `REPETITION_LIMIT` in
`constants.py`) exist only in this engine, not official Hive or the
reference app -- needed because an untrained/near-random self-play policy
can otherwise shuffle pieces indefinitely. Bug-for-bug notes worth knowing
if oracle-parity tests ever fail mysteriously: the pillbug freeze check in
`GameState.is_frozen` deliberately uses `ply - 1` (not the more obvious
`ply`) to match the reference's exact comparison; `_move_keeps_hive_connected`
was deliberately made *more correct* than the reference's connectivity
check (which has a real, reachable bug for a beetle standing on a
structurally load-bearing stack) -- both are called out in code comments
where they diverge.

### Board sizing is fixed for the full expansion set, even though v1 only trains base pieces

`BOARD_RADIUS = 27` / `BOARD_DIM = 55` are sized for all 28 pieces (14/side,
full expansion set) ever being on the board, even though `BASE_PIECE_TYPES`
(queen/ant/spider/beetle/grasshopper) is what training currently uses. This
means tensors/checkpoints never need to change shape when
mosquito/ladybug/pillbug get enabled later. First-placed piece is always at
cube coordinate `(0, 0, 0)`, never recentered.

### Perspective-relative encoding

`encode_state` (engine/encode.py) always encodes "current player as me,
other player as opponent" rather than raw owner 0/1 -- so the network only
ever learns one perspective, and its value output is always "how good is
this position for whoever is about to move." MCTS's backprop (model/mcts.py)
relies on this: each level up the tree negates the value, since parent and
child alternate whose turn it is. Get this backwards and training will
silently learn to lose.

### Policy is a dual-embedding bilinear head, not a dense action tensor

Hive's action space doesn't fit a fixed per-square move-plane design
(chess/shogi-style) because slides are pathfinding-based, not fixed-direction,
and the naive dense (from-cell x to-cell) tensor would be ~9M entries. Instead
`HiveNet` (model/network.py) produces per-cell `from`/`to` embeddings plus
context-dependent hand-slot embeddings for placements; a move's score is a
dot product. `engine/actions.py` maps each `Move` to a `(kind, from_index,
to_index)` key; `score_actions` gathers scores only for a position's actual
legal keys -- there's never a materialized dense action tensor anywhere.
MOVE and THROW can share the same `from` cell (a pillbug can throw a piece,
or that piece can move itself) so `kind_bias` nudges them apart before the
dot product.

### MCTS shares one mutable GameState

`model/mcts.py` doesn't clone `GameState` per node. It descends by calling
`apply_move`, backs out with `undo_move`, and reads the engine's own
generated moves at each expansion -- the same make/unmake pattern the engine
uses internally, for the same throughput reason.

### Training loop resumability

`training/train.py`'s `train()` takes `resume_from` (a checkpoint path) and
derives `start_iteration` from the checkpoint's own stored iteration number,
so checkpoint numbering and Adam's momentum state stay continuous across
separate calls (i.e. separate Colab sessions), rather than resetting to 0
and overwriting old checkpoints. Checkpoints are `{"model": state_dict,
"optimizer": state_dict, "iteration": int}`; `HiveBot.from_checkpoint`
(analysis/bot.py) accepts either that format or a bare model state_dict.

### Action-space policy targets, not fixed-size vectors

Because the legal-move set differs per position, `training.Sample` stores
`action_keys` (parallel list) + `target_policy` (aligned probabilities) per
sample rather than a fixed-length policy vector. `compute_loss` batches the
network's conv forward pass across a batch, but the policy loss itself loops
per-sample (varying action-key lists can't trivially vectorize) using
`score_actions` + cross-entropy against the MCTS visit-count distribution.
