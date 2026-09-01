# hive-bot

An AlphaZero-style bot for the board game Hive: a fast, dependency-light game
engine, MCTS guided by a PyTorch policy/value network, and a self-play
training loop runnable from a Google Colab notebook.

Standalone from the `ttbg-web-app` Django app that hosts the playable game --
see `tests/reference/` for how correctness is validated against that app's
rules.

## Setup

```
make setup      # uv sync --group dev
make lint
make typecheck
make test        # fast suite (~seconds)
make test-full   # exhaustive engine-vs-reference-oracle parity gate (~10 min)
```

## Try it locally

```
make selfplay   # tiny network, a few quick self-play games
make train      # tiny network, a couple of quick self-play/train iterations
```

Both use a small network and a low ply cap purely as a smoke test. Real
training happens in `notebooks/train_colab.ipynb`.

## Layout

- `src/hive_bot/engine/` -- board state, move generation, apply/undo, tensor encoding, and the move<->policy-action-index mapping.
- `src/hive_bot/model/` -- the PyTorch network (`HiveNet`, dual-embedding bilinear policy head) and PUCT MCTS.
- `src/hive_bot/training/` -- self-play game generation, the replay buffer, and the self-play/train iterate loop (resumable across sessions via checkpointed model + optimizer state).
- `src/hive_bot/analysis/` -- `HiveBot`: load a checkpoint and get a best-move + win-probability readout for a position -- the "what's the best move here" API.
- `notebooks/train_colab.ipynb` -- the Colab training entrypoint: mounts Drive, resumes from the latest checkpoint automatically, runs the training loop, and demonstrates `HiveBot.analyze`.
- `tests/reference/` -- a vendored, unmodified copy of the Django app's
  `game_state.py`/`helpers.py` (correctness oracle only) plus `oracle.py`, a
  from-scratch re-transcription of the app's move-orchestration logic needed
  to enumerate legal moves for comparison (see its docstring for the two real
  bugs this found in the shipped app).

## Rules notes

- v1 only enables the base 5 piece types (queen/ant/spider/beetle/grasshopper); the board/tensors are already sized for the full expansion set (mosquito/ladybug/pillbug), whose move logic is implemented and tested but not yet wired into training.
- Two rules exist only in this engine, not official Hive or the reference app: a move-count cap (`MAX_PLIES_BEFORE_DRAW`) and threefold-repetition (`REPETITION_LIMIT`), both forcing a draw. Needed because an untrained/near-random self-play policy can otherwise shuffle pieces indefinitely -- see `constants.py`.

See the design doc for the full architecture and rationale (board sizing,
tensor encoding, policy head design, etc.).
