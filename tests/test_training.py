"""Smoke tests for the self-play/replay-buffer/train loop -- tiny network,
few simulations/games, just checking the plumbing runs end to end and
produces sane shapes/values. Not a play-strength or convergence test."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from hive_bot.engine.constants import BASE_PIECE_TYPES
from hive_bot.model.network import HiveNet, score_actions
from hive_bot.training.replay_buffer import ReplayBuffer
from hive_bot.training.selfplay import play_game
from hive_bot.training.train import compute_loss, train

TINY_NET_KWARGS = {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}


def _tiny_model() -> HiveNet:
    torch.manual_seed(0)
    model = HiveNet(**TINY_NET_KWARGS)
    model.eval()
    return model


def test_play_game_produces_one_sample_per_recorded_ply() -> None:
    model = _tiny_model()
    samples = play_game(
        model,
        num_simulations=4,
        enabled_types=BASE_PIECE_TYPES,
        rng=random.Random(0),
        np_rng=np.random.default_rng(0),
        max_plies=8,
    )
    assert len(samples) == 8  # hits max_plies without a real game-over
    for sample in samples:
        assert sample.board.shape[0] > 0
        assert len(sample.action_keys) == sample.target_policy.shape[0]
        assert abs(sample.target_policy.sum().item() - 1.0) < 1e-5
        assert sample.value_target in (-1.0, 0.0, 1.0)


def test_replay_buffer_add_and_sample() -> None:
    model = _tiny_model()
    samples = play_game(
        model,
        num_simulations=4,
        rng=random.Random(1),
        np_rng=np.random.default_rng(1),
        max_plies=6,
    )
    buffer = ReplayBuffer(capacity=100)
    buffer.add_game(samples)
    assert len(buffer) == len(samples)

    batch = buffer.sample(4, random.Random(2))
    assert len(batch) == min(4, len(samples))


def test_compute_loss_is_finite_and_positive() -> None:
    model = _tiny_model()
    model.train()
    samples = play_game(
        model,
        num_simulations=4,
        rng=random.Random(3),
        np_rng=np.random.default_rng(3),
        max_plies=6,
    )
    total_loss, policy_loss, value_loss = compute_loss(model, samples)
    assert torch.isfinite(total_loss)
    assert policy_loss.item() >= 0
    assert value_loss.item() >= 0


def test_compute_loss_matches_unvectorized_per_sample_reference() -> None:
    """`compute_loss` scores every sample's legal actions in one batched
    gather + segmented softmax instead of a Python loop calling
    `score_actions` once per sample (see network.py's `score_actions_batch`
    docstring for why) -- this must produce the exact same loss as the
    naive per-sample version, computed independently here, or the
    refactor silently changed what the network is being trained on."""
    model = _tiny_model()
    model.train()
    samples = play_game(
        model,
        num_simulations=4,
        rng=random.Random(3),
        np_rng=np.random.default_rng(3),
        max_plies=6,
    )

    boards = torch.stack([s.board for s in samples])
    global_features = torch.stack([s.global_features for s in samples])
    with torch.no_grad():
        output = model(boards, global_features)
        reference_policy_losses = []
        for i, sample in enumerate(samples):
            scores = score_actions(output, sample.action_keys, batch_index=i)
            log_probs = torch.log_softmax(scores, dim=0)
            reference_policy_losses.append(-(sample.target_policy * log_probs).sum())
        reference_policy_loss = torch.stack(reference_policy_losses).mean()

        _, policy_loss, _ = compute_loss(model, samples)

    assert torch.allclose(policy_loss, reference_policy_loss, atol=1e-5)


def test_train_loop_runs_and_updates_weights() -> None:
    model = _tiny_model()
    before = next(model.parameters()).clone()

    trained = train(
        iterations=1,
        games_per_iter=1,
        simulations=4,
        batch_size=4,
        batches_per_iter=2,
        seed=0,
        model=model,
    )

    after = next(trained.parameters())
    assert trained is model
    assert not torch.equal(before, after)


def test_train_loop_checkpoint_resume_continues_iteration_count(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    train(
        iterations=2,
        games_per_iter=1,
        simulations=4,
        batch_size=4,
        batches_per_iter=2,
        seed=0,
        model=_tiny_model(),
        checkpoint_dir=checkpoint_dir,
    )
    assert {p.name for p in checkpoint_dir.glob("*.pt")} == {
        "checkpoint_0.pt",
        "checkpoint_1.pt",
    }

    # Resuming from iteration 1's checkpoint should pick up at iteration 2,
    # not reset to 0 and overwrite what's already there.
    train(
        iterations=1,
        games_per_iter=1,
        simulations=4,
        batch_size=4,
        batches_per_iter=2,
        seed=1,
        model=_tiny_model(),
        checkpoint_dir=checkpoint_dir,
        resume_from=checkpoint_dir / "checkpoint_1.pt",
    )
    assert {p.name for p in checkpoint_dir.glob("*.pt")} == {
        "checkpoint_0.pt",
        "checkpoint_1.pt",
        "checkpoint_2.pt",
    }
