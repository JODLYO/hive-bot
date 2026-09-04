"""Dev-only tool: validate scraped hivegame.com games (see
scrape_hivegame_archive.py) against this engine's rules and split them into
train/val JSONL files for `training.pretrain` -- see the plan doc
("Bootstrap training from real hivegame.com games").

This does NOT pre-encode games into `training.selfplay.Sample`s or pickle
them: a `Sample`'s board tensor alone is ~213KB (55x55x18 float32, see
engine/encode.py), and the full archive is on the order of a million-plus
positions -- materializing that as one in-memory (or pickled) collection
is on the order of hundreds of GB, not something any machine here can hold.
`training.pretrain`'s training loop instead replays a bounded number of
games at a time each epoch, encoding samples on the fly and discarding
them once trained on -- so this script's only job is to produce small
(history + result only, no tensors) JSONL files for it to stream from.

Validation is pure-Python engine replay (no I/O, no shared state) per
game, so it's split across a process pool -- at real archive scale
(tens of thousands of games) a single process showed zero progress output
for many hours with no way to tell how far along it was, which is its own
problem independent of raw speed: this reports progress via tqdm as
games complete, not just once at the very end.

A real spider-move engine bug (see `data/hivegame_archive.py` and
engine/moves.py's `_slide_reachable_exact`) used to make a small but
consistent fraction of real games fail to replay; with that fixed, a
160-game sample from the archive replays 160/160. Games that fail are
still logged and skipped rather than silently corrupting the dataset
(never assume 100% going forward -- report the rate every run), since a
sudden drop would be worth investigating as a real regression rather than
silently eating data.

The train/val split happens on *games*, not the resulting samples --
splitting samples would leak positions across the split (many plies
within one game, especially the highly repetitive opening ones, are
near-duplicates of each other), which would make validation loss an
overly optimistic estimate of how well the network generalizes to
genuinely unseen games.

Usage: uv run python scripts/build_pretrain_dataset.py \\
    --games data/hivegame_archive/games.jsonl \\
    --train-out data/hivegame_samples/train_games.jsonl \\
    --val-out data/hivegame_samples/val_games.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from hive_bot.data.hivegame_archive import (
    UhpReplayError,
    load_base_games_jsonl,
    replay_uhp_game,
    winner_from_game_status,
)
from hive_bot.engine.constants import BASE_PIECE_TYPES


def _validate_one(game: dict[str, Any]) -> str | None:
    """Runs in a worker process -- returns None if `game` replays cleanly,
    or a short failure-reason string otherwise. Only that verdict crosses
    the process boundary, never the replayed samples themselves (keeping
    inter-process communication cheap regardless of how big a game's
    encoded tensors would be)."""
    history = [tuple(m) for m in game["history"]]
    winner = winner_from_game_status(game.get("game_status"))
    try:
        replay_uhp_game(history, winner, BASE_PIECE_TYPES)
    except UhpReplayError as exc:
        # Group by the trailing "no legal ..." reason, not the full
        # message (which includes game-specific coordinates), so the
        # summary shows failure *patterns*, not one unique line per game.
        return str(exc).split(": ", 2)[-1]
    return None


def _validate_games(
    games: list[dict[str, Any]],
    failure_reasons: Counter[str],
    workers: int,
    desc: str,
) -> list[dict[str, Any]]:
    """Games that don't replay cleanly are dropped here rather than at
    training time, so a training run never has to handle/skip a bad game
    mid-epoch."""
    valid: list[dict[str, Any]] = []
    if not games:
        return valid
    # chunksize batches work per task handed to a worker, cutting IPC
    # overhead when there are tens of thousands of (individually fast)
    # games -- without it, every single game is its own round trip.
    chunksize = max(1, len(games) // (workers * 20))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_validate_one, games, chunksize=chunksize)
        for game, reason in zip(
            games, tqdm(results, total=len(games), desc=desc), strict=True
        ):
            if reason is None:
                valid.append(game)
            else:
                failure_reasons[reason] += 1
    return valid


def build_dataset(
    games_path: Path,
    train_out: Path,
    val_out: Path,
    val_fraction: float,
    seed: int,
    workers: int,
) -> None:
    games = load_base_games_jsonl(games_path)

    # Split by game, before validating -- see the module docstring for why.
    random.Random(seed).shuffle(games)
    num_val_games = round(len(games) * val_fraction)
    val_games = games[:num_val_games]
    train_games = games[num_val_games:]

    failure_reasons: Counter[str] = Counter()
    valid_train = _validate_games(train_games, failure_reasons, workers, "validating train")
    valid_val = _validate_games(val_games, failure_reasons, workers, "validating val")

    print(
        f"validated {len(valid_train) + len(valid_val)}/{len(games)} games "
        f"(train: {len(valid_train)}/{len(train_games)}; "
        f"val: {len(valid_val)}/{len(val_games)})"
    )
    if failure_reasons:
        print("failure reasons (top 10):")
        for reason, count in failure_reasons.most_common(10):
            print(f"  {count:5d}  {reason}")

    for path, valid_games_out in ((train_out, valid_train), (val_out, valid_val)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for game in valid_games_out:
                f.write(json.dumps(game, default=str) + "\n")
        print(f"wrote {len(valid_games_out)} games to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path, default=Path("data/hivegame_archive/games.jsonl")
    )
    parser.add_argument(
        "--train-out", type=Path, default=Path("data/hivegame_samples/train_games.jsonl")
    )
    parser.add_argument(
        "--val-out", type=Path, default=Path("data/hivegame_samples/val_games.jsonl")
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of games (not samples) held out for validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Shuffle seed for the train/val game split."
    )
    parser.add_argument(
        "--workers",
        type=int,
        # Half the machine's cores by default rather than all of them --
        # leaves headroom for whatever else is running instead of adding
        # to contention on an already-busy machine.
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Worker processes for validation (games are independent, so this parallelizes cleanly).",
    )
    args = parser.parse_args()
    build_dataset(
        args.games, args.train_out, args.val_out, args.val_fraction, args.seed, args.workers
    )


if __name__ == "__main__":
    main()
