"""Supervised "behavior cloning" pretraining on real human games (see
scripts/build_pretrain_dataset.py and the plan doc, "Bootstrap training
from real hivegame.com games"). Unlike `train.py`'s self-play loop, this
doesn't generate any of its own data -- it just does ordinary epoch-based
supervised training, reusing `compute_loss` unchanged (it's agnostic to
whether a sample's `target_policy` came from MCTS visit counts or a
one-hot behavior-cloning target).

Games are replayed into `training.selfplay.Sample`s in bounded-size chunks
rather than all up front: a `Sample`'s board tensor alone is ~213KB
(55x55x18 float32), and the full hivegame.com archive is on the order of a
million-plus positions -- holding that as one in-memory collection is on
the order of hundreds of GB. `_iter_sample_batches` instead shuffles the
*games*, replays/encodes `games_per_chunk` of them at a time (shuffling
the resulting samples within that chunk for batch diversity), yields
batches, and discards the chunk before moving to the next one -- so peak
memory is bounded by chunk size, not dataset size, at the cost of
re-running engine replay every epoch. That replay is pure-Python engine
move generation (not the GPU-bound part of training at all), so it's the
main thing worth parallelizing: `workers > 1` replays a chunk's games
across a process pool, same pattern as
scripts/build_pretrain_dataset.py's validation pass, cutting the
CPU-bound wall-clock cost roughly by the worker count.

The whole point of this module is the checkpoint it produces: saved in the
exact `{"model": ..., "optimizer": ..., "iteration": 0}` shape `train()`'s
`resume_from` already expects, so self-play can continue straight from a
pretrained network with zero changes to the self-play loop itself --
`iteration` is always 0 here (this isn't self-play-iteration numbering),
so `train(resume_from=...)` naturally starts self-play at iteration 1.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm, trange

from ..data.hivegame_archive import (
    UhpReplayError,
    load_base_games_jsonl,
    replay_uhp_game,
    winner_from_game_status,
)
from ..engine.constants import BASE_PIECE_TYPES, PieceType
from ..model.network import HiveNet
from .selfplay import Sample
from .train import compute_loss

DEFAULT_GAMES_PER_CHUNK = 200  # ~200 games * ~38 plies/game ~= 1.6GB of encoded samples


def _replay_game_samples(
    game: dict[str, Any], enabled_types: frozenset[PieceType]
) -> list[Sample]:
    """Module-level (so it's picklable for a process pool) single-game
    replay -- returns [] for a game that doesn't replay cleanly rather
    than raising, since games passed in should already have been
    validated by build_pretrain_dataset.py; this is just defensive."""
    history = [tuple(m) for m in game["history"]]
    winner = winner_from_game_status(game.get("game_status"))
    try:
        return replay_uhp_game(history, winner, enabled_types).samples
    except UhpReplayError:
        return []


def _iter_sample_batches(
    games: list[dict[str, Any]],
    enabled_types: frozenset[PieceType],
    batch_size: int,
    games_per_chunk: int,
    rng: random.Random,
    workers: int = 1,
) -> Iterator[list[Sample]]:
    """Replaying/encoding a chunk is pure-Python engine work -- CPU-bound,
    not GPU-bound -- so `workers > 1` spreads a chunk's games across a
    process pool instead of replaying them one at a time on the caller's
    process. This doesn't change *which* samples end up in which batch
    (same chunk membership, same post-chunk shuffle), only how fast the
    chunk gets produced."""
    shuffled = list(games)
    rng.shuffle(shuffled)

    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for chunk_start in range(0, len(shuffled), games_per_chunk):
            chunk_games = shuffled[chunk_start : chunk_start + games_per_chunk]
            chunk_samples: list[Sample] = []
            if pool is not None:
                for samples in pool.map(
                    _replay_game_samples,
                    chunk_games,
                    (enabled_types for _ in chunk_games),
                    chunksize=max(1, len(chunk_games) // (workers * 4)),
                ):
                    chunk_samples.extend(samples)
            else:
                for game in chunk_games:
                    chunk_samples.extend(_replay_game_samples(game, enabled_types))
            rng.shuffle(chunk_samples)
            for start in range(0, len(chunk_samples), batch_size):
                yield chunk_samples[start : start + batch_size]
    finally:
        if pool is not None:
            pool.shutdown()


def _mean_val_loss(
    model: HiveNet,
    val_games: list[dict[str, Any]],
    enabled_types: frozenset[PieceType],
    batch_size: int,
    games_per_chunk: int,
    workers: int,
) -> float:
    """No shuffling matters here, no grad, no optimizer step -- this never
    influences the weights, it only measures how the current weights
    generalize to games the model hasn't trained on (see
    build_pretrain_dataset.py's docstring for why the split happens on
    whole games, not samples)."""
    model.eval()
    loss_total = 0.0
    num_batches = 0
    # seed=0 here is arbitrary and irrelevant to correctness -- shuffling
    # only affects chunk *order*/composition for memory chunking, not
    # which games get evaluated or the resulting mean.
    with torch.no_grad():
        for batch in _iter_sample_batches(
            val_games, enabled_types, batch_size, games_per_chunk, random.Random(0), workers
        ):
            total_loss, _, _ = compute_loss(model, batch)
            loss_total += total_loss.item()
            num_batches += 1
    return loss_total / max(num_batches, 1)


# Half the machine's cores by default rather than all of them -- leaves
# headroom for whatever else is running (e.g. the notebook kernel itself,
# or other work sharing the machine) instead of adding to contention.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)


def pretrain(
    model: HiveNet,
    games: list[dict[str, Any]],
    epochs: int,
    *,
    enabled_types: frozenset[PieceType] = BASE_PIECE_TYPES,
    batch_size: int = 32,
    games_per_chunk: int = DEFAULT_GAMES_PER_CHUNK,
    workers: int = DEFAULT_WORKERS,
    lr: float = 1e-3,
    val_games: list[dict[str, Any]] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_batches: int | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> HiveNet:
    """`games`/`val_games` are raw scraped-game dicts (each needs a
    `history` field and a `game_status` field) -- typically loaded via
    `data.hivegame_archive.load_base_games_jsonl` from the JSONL files
    scripts/build_pretrain_dataset.py produces, which have already been
    validated to replay cleanly against `enabled_types`.

    `device` defaults to CUDA if available -- nothing else in this
    codebase moves the model there automatically, so without this (or the
    caller doing it themselves) training silently runs on CPU even with a
    GPU attached, which is most of why a "30s/batch"-scale run turns out
    to be CPU-bound: a real-sized ResNet-ish net at BOARD_DIM=55 resolution
    is not fast on CPU alone."""
    if not games:
        raise ValueError("pretrain() needs at least one game")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    global_batch = 0

    for epoch in trange(epochs, desc="pretrain"):
        model.train()
        last_losses = (float("nan"), float("nan"), float("nan"))
        num_batches = 0
        loss_total = 0.0

        batches = _iter_sample_batches(
            games, enabled_types, batch_size, games_per_chunk, rng, workers
        )
        for batch in tqdm(batches, desc=f"epoch {epoch}", leave=False):
            total_loss, policy_loss, value_loss = compute_loss(model, batch)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            last_losses = (total_loss.item(), policy_loss.item(), value_loss.item())
            loss_total += last_losses[0]
            num_batches += 1
            global_batch += 1

            if (
                checkpoint_dir is not None
                and checkpoint_every_batches is not None
                and global_batch % checkpoint_every_batches == 0
            ):
                _save_checkpoint(
                    model, optimizer, checkpoint_dir / f"pretrain_batch_{global_batch}.pt"
                )

        val_msg = ""
        if val_games:
            val_loss = _mean_val_loss(
                model, val_games, enabled_types, batch_size, games_per_chunk, workers
            )
            val_msg = f" val_loss={val_loss:.4f}"

        tqdm.write(
            f"epoch {epoch}: mean_loss={loss_total / max(num_batches, 1):.4f} "
            f"last_loss={last_losses[0]:.4f} policy={last_losses[1]:.4f} "
            f"value={last_losses[2]:.4f}{val_msg}"
        )

        if checkpoint_dir is not None:
            _save_checkpoint(
                model, optimizer, checkpoint_dir / f"pretrain_epoch_{epoch}.pt"
            )

    return model


def _save_checkpoint(model: HiveNet, optimizer: torch.optim.Optimizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": 0,
        },
        path,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Supervised pretrain on real human games.")
    parser.add_argument(
        "--train-games", type=Path, default=Path("data/hivegame_samples/train_games.jsonl")
    )
    parser.add_argument(
        "--val-games",
        type=Path,
        default=Path("data/hivegame_samples/val_games.jsonl"),
        help="Pass a missing path to skip validation reporting.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--games-per-chunk",
        type=int,
        default=DEFAULT_GAMES_PER_CHUNK,
        help="How many games' worth of samples to hold in memory at once.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Worker processes replaying/encoding games (they're independent, so this parallelizes cleanly).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=None,
        help="Also save mid-epoch every N batches (in addition to once per epoch).",
    )
    parser.add_argument(
        "--tiny-net",
        action="store_true",
        help="Use a small network (fast on CPU) instead of the real training size.",
    )
    args = parser.parse_args()

    games = load_base_games_jsonl(args.train_games)
    val_games = load_base_games_jsonl(args.val_games) if args.val_games.exists() else None

    net_kwargs = (
        {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}
        if args.tiny_net
        else {}
    )
    model = HiveNet(**net_kwargs)

    pretrain(
        model,
        games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        games_per_chunk=args.games_per_chunk,
        workers=args.workers,
        lr=args.lr,
        val_games=val_games,
        seed=args.seed,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        checkpoint_every_batches=args.checkpoint_every_batches,
    )


if __name__ == "__main__":
    _main()
