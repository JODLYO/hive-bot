"""Tests for the best-move/eval analysis API in analysis/bot.py."""

from __future__ import annotations

from pathlib import Path

import torch

from hive_bot.analysis.bot import HiveBot
from hive_bot.engine.constants import BASE_PIECE_TYPES
from hive_bot.engine.moves import generate_legal_moves
from hive_bot.engine.state import GameState
from hive_bot.model.network import HiveNet

TINY_NET_KWARGS = {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}


def _tiny_model() -> HiveNet:
    torch.manual_seed(0)
    model = HiveNet(**TINY_NET_KWARGS)
    model.eval()
    return model


def test_analyze_returns_a_legal_best_move_and_full_evaluation_list() -> None:
    model = _tiny_model()
    bot = HiveBot(model, num_simulations=8)
    state = GameState.new_game(BASE_PIECE_TYPES)
    legal = generate_legal_moves(state)

    analysis = bot.analyze(state)

    assert analysis.best_move in legal
    assert 0.0 <= analysis.win_probability <= 1.0
    assert len(analysis.move_evaluations) == len(legal)
    assert {e.move for e in analysis.move_evaluations} == set(legal)
    assert abs(sum(e.visit_fraction for e in analysis.move_evaluations) - 1.0) < 1e-6
    # Sorted descending by visit count, and the best move is the top one.
    counts = [e.visit_count for e in analysis.move_evaluations]
    assert counts == sorted(counts, reverse=True)
    assert analysis.move_evaluations[0].move == analysis.best_move


def test_from_checkpoint_round_trips_new_style_checkpoint(tmp_path: Path) -> None:
    model = _tiny_model()
    checkpoint_path = tmp_path / "checkpoint_0.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": {}, "iteration": 0}, checkpoint_path
    )

    bot = HiveBot.from_checkpoint(checkpoint_path, num_simulations=4, **TINY_NET_KWARGS)
    state = GameState.new_game(BASE_PIECE_TYPES)
    analysis = bot.analyze(state)
    assert analysis.best_move in generate_legal_moves(state)


def test_from_checkpoint_round_trips_bare_state_dict(tmp_path: Path) -> None:
    model = _tiny_model()
    checkpoint_path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), checkpoint_path)

    bot = HiveBot.from_checkpoint(checkpoint_path, num_simulations=4, **TINY_NET_KWARGS)
    state = GameState.new_game(BASE_PIECE_TYPES)
    analysis = bot.analyze(state)
    assert analysis.best_move in generate_legal_moves(state)
