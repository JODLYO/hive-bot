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
re-running engine replay each epoch (cheap; it's pure-Python move
generation, not the tensor encoding that dominates memory).

The whole point of this module is the checkpoint it produces: saved in the
exact `{"model": ..., "optimizer": ..., "iteration": 0}` shape `train()`'s
`resume_from` already expects, so self-play can continue straight from a
pretrained network with zero changes to the self-play loop itself --
`iteration` is always 0 here (this isn't self-play-iteration numbering),
so `train(resume_from=...)` naturally starts self-play at iteration 1.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterator
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


def _iter_sample_batches(
    games: list[dict[str, Any]],
    enabled_types: frozenset[PieceType],
    batch_size: int,
    games_per_chunk: int,
    rng: random.Random,
) -> Iterator[list[Sample]]:
    shuffled = list(games)
    rng.shuffle(shuffled)
    for chunk_start in range(0, len(shuffled), games_per_chunk):
        chunk_samples: list[Sample] = []
        for game in shuffled[chunk_start : chunk_start + games_per_chunk]:
            history = [tuple(m) for m in game["history"]]
            winner = winner_from_game_status(game.get("game_status"))
            try:
                result = replay_uhp_game(history, winner, enabled_types)
            except UhpReplayError:
                # Already filtered by build_pretrain_dataset.py -- this is
                # just defensive, in case games are ever passed in without
                # going through that validation step first.
                continue
            chunk_samples.extend(result.samples)
        rng.shuffle(chunk_samples)
        for start in range(0, len(chunk_samples), batch_size):
            yield chunk_samples[start : start + batch_size]


def _mean_val_loss(
    model: HiveNet,
    val_games: list[dict[str, Any]],
    enabled_types: frozenset[PieceType],
    batch_size: int,
    games_per_chunk: int,
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
            val_games, enabled_types, batch_size, games_per_chunk, random.Random(0)
        ):
            total_loss, _, _ = compute_loss(model, batch)
            loss_total += total_loss.item()
            num_batches += 1
    return loss_total / max(num_batches, 1)


def pretrain(
    model: HiveNet,
    games: list[dict[str, Any]],
    epochs: int,
    *,
    enabled_types: frozenset[PieceType] = BASE_PIECE_TYPES,
    batch_size: int = 32,
    games_per_chunk: int = DEFAULT_GAMES_PER_CHUNK,
    lr: float = 1e-3,
    val_games: list[dict[str, Any]] | None = None,
    checkpoint_dir: Path | None = None,
    seed: int | None = None,
) -> HiveNet:
    """`games`/`val_games` are raw scraped-game dicts (each needs a
    `history` field and a `game_status` field) -- typically loaded via
    `data.hivegame_archive.load_base_games_jsonl` from the JSONL files
    scripts/build_pretrain_dataset.py produces, which have already been
    validated to replay cleanly against `enabled_types`."""
    if not games:
        raise ValueError("pretrain() needs at least one game")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)

    for epoch in trange(epochs, desc="pretrain"):
        model.train()
        last_losses = (float("nan"), float("nan"), float("nan"))
        num_batches = 0
        loss_total = 0.0

        batches = _iter_sample_batches(
            games, enabled_types, batch_size, games_per_chunk, rng
        )
        for batch in tqdm(batches, desc=f"epoch {epoch}", leave=False):
            total_loss, policy_loss, value_loss = compute_loss(model, batch)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            last_losses = (total_loss.item(), policy_loss.item(), value_loss.item())
            loss_total += last_losses[0]
            num_batches += 1

        val_msg = ""
        if val_games:
            val_loss = _mean_val_loss(
                model, val_games, enabled_types, batch_size, games_per_chunk
            )
            val_msg = f" val_loss={val_loss:.4f}"

        tqdm.write(
            f"epoch {epoch}: mean_loss={loss_total / max(num_batches, 1):.4f} "
            f"last_loss={last_losses[0]:.4f} policy={last_losses[1]:.4f} "
            f"value={last_losses[2]:.4f}{val_msg}"
        )

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": 0,
                },
                checkpoint_dir / f"pretrain_epoch_{epoch}.pt",
            )

    return model


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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
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
        lr=args.lr,
        val_games=val_games,
        seed=args.seed,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )


if __name__ == "__main__":
    _main()
