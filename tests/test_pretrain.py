"""Smoke tests for the supervised pretrain loop -- tiny network, real (but
small) human-game data, just checking the plumbing runs end to end and
produces a checkpoint `train()` can resume from. Not a play-strength test.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from hive_bot.engine.constants import BASE_PIECE_TYPES
from hive_bot.model.network import HiveNet
from hive_bot.training import pretrain as pretrain_module
from hive_bot.training.pretrain import pretrain
from hive_bot.training.train import train

from .test_uhp_replay import (
    _GAME_NO_CLIMB,
    _GAME_NO_CLIMB_WINNER,
    _GAME_WITH_CLIMB,
    _GAME_WITH_CLIMB_WINNER,
)

TINY_NET_KWARGS = {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}


def _tiny_model() -> HiveNet:
    torch.manual_seed(0)
    return HiveNet(**TINY_NET_KWARGS)


def _game_dict(history: list[tuple[str, str]], winner: int | None) -> dict:
    """`pretrain()` takes raw scraped-game dicts (see
    `data.hivegame_archive.load_base_games_jsonl`), not pre-built Samples
    -- it replays/encodes them itself, in bounded chunks, rather than
    expecting the whole dataset already materialized as tensors (see
    pretrain.py's module docstring for why). This wraps the hand-written
    UHP fixtures from test_uhp_replay.py into that same shape."""
    winner_name = None if winner is None else ("White" if winner == 0 else "Black")
    game_status = (
        {"Finished": {"Draw": None}}
        if winner is None
        else {"Finished": {"Winner": winner_name}}
    )
    return {"history": history, "game_status": game_status}


def _game_dataset() -> list[dict]:
    return [
        _game_dict(_GAME_NO_CLIMB, _GAME_NO_CLIMB_WINNER),
        _game_dict(_GAME_WITH_CLIMB, _GAME_WITH_CLIMB_WINNER),
    ]


def test_pretrain_runs_and_updates_weights() -> None:
    model = _tiny_model()
    before = next(model.parameters()).clone()

    trained = pretrain(model, _game_dataset(), epochs=2, batch_size=8, seed=0)

    after = next(model.parameters())
    assert trained is model
    assert not torch.equal(before, after)


def test_pretrain_checkpoint_is_iteration_zero(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    pretrain(
        _tiny_model(),
        _game_dataset(),
        epochs=1,
        batch_size=8,
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )
    checkpoint = torch.load(checkpoint_dir / "pretrain_epoch_0.pt", map_location="cpu")
    assert checkpoint["iteration"] == 0
    assert "model" in checkpoint and "optimizer" in checkpoint


def test_val_games_are_reported_but_never_affect_training(capsys) -> None:
    """Validation must be pure evaluation -- same seed/data/hyperparameters
    with vs. without val_games must produce byte-identical trained
    weights, proving the val pass never contributes a gradient step."""
    without_val = pretrain(_tiny_model(), _game_dataset(), epochs=2, batch_size=8, seed=0)

    dataset = _game_dataset()
    with_val = pretrain(
        _tiny_model(),
        dataset,
        epochs=2,
        batch_size=8,
        seed=0,
        val_games=dataset[:1],
    )

    for p1, p2 in zip(without_val.parameters(), with_val.parameters(), strict=True):
        assert torch.equal(p1, p2)

    assert "val_loss=" in capsys.readouterr().out


def test_chunking_never_loses_or_duplicates_samples() -> None:
    """Chunking is purely a memory-bounding mechanism, not a subsampling
    one: every game's samples must appear in exactly one batch across a
    full pass, regardless of how many games fit in a chunk at a time.
    (It does change *batch composition/order*, which is expected to shift
    final trained weights under SGD -- that's not tested here, this only
    checks that no data is silently lost or duplicated.)"""
    games = _game_dataset()
    expected_total = len(_GAME_NO_CLIMB) + len(_GAME_WITH_CLIMB)

    for games_per_chunk in (1, 200):
        total = sum(
            len(batch)
            for batch in pretrain_module._iter_sample_batches(
                games,
                BASE_PIECE_TYPES,
                batch_size=8,
                games_per_chunk=games_per_chunk,
                rng=random.Random(0),
            )
        )
        assert total == expected_total, f"games_per_chunk={games_per_chunk}"


def test_pretrain_checkpoint_can_be_resumed_by_selfplay_train(tmp_path: Path) -> None:
    """The whole point of pretrain(): its checkpoint must be a drop-in
    `resume_from` for the existing self-play loop, continuing iteration
    numbering from 0 rather than resetting it."""
    checkpoint_dir = tmp_path / "checkpoints"
    pretrain(
        _tiny_model(),
        _game_dataset(),
        epochs=1,
        batch_size=8,
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )

    train(
        iterations=1,
        games_per_iter=1,
        simulations=4,
        batch_size=4,
        batches_per_iter=2,
        seed=0,
        model=_tiny_model(),
        checkpoint_dir=checkpoint_dir,
        resume_from=checkpoint_dir / "pretrain_epoch_0.pt",
    )
    assert (checkpoint_dir / "checkpoint_1.pt").exists()
