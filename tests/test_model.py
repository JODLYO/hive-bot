"""Sanity tests for the network and MCTS -- shapes, determinism, and that a
full search runs to completion on real positions without crashing. Not a
test of play strength (nothing here trains the network)."""

from __future__ import annotations

import random

import numpy as np
import torch

from hive_bot.engine.actions import legal_action_keys
from hive_bot.engine.apply import apply_move
from hive_bot.engine.constants import BASE_PIECE_TYPES
from hive_bot.engine.encode import encode_state
from hive_bot.engine.moves import generate_legal_moves
from hive_bot.engine.state import GameState
from hive_bot.model.mcts import MCTS, select_move, visit_counts
from hive_bot.model.network import HiveNet, score_actions

TINY_NET_KWARGS = {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}


def _tiny_model() -> HiveNet:
    torch.manual_seed(0)
    model = HiveNet(**TINY_NET_KWARGS)
    model.eval()
    return model


def test_network_forward_shapes() -> None:
    model = _tiny_model()
    state = GameState.new_game(BASE_PIECE_TYPES)
    encoded = encode_state(state)
    with torch.no_grad():
        output = model(encoded.board.unsqueeze(0), encoded.global_features.unsqueeze(0))

    embed_dim = TINY_NET_KWARGS["embed_dim"]
    from hive_bot.engine.constants import BOARD_DIM, NUM_PIECE_TYPES

    assert output.from_map.shape == (1, embed_dim, BOARD_DIM, BOARD_DIM)
    assert output.to_map.shape == (1, embed_dim, BOARD_DIM, BOARD_DIM)
    assert output.hand_embed.shape == (1, NUM_PIECE_TYPES, embed_dim)
    assert output.kind_bias.shape == (2, embed_dim)
    assert output.value.shape == (1,)
    assert -1.0 <= output.value.item() <= 1.0


def test_score_actions_matches_move_count() -> None:
    model = _tiny_model()
    state = GameState.new_game(BASE_PIECE_TYPES)
    moves = generate_legal_moves(state)
    keys = legal_action_keys(state, moves)
    encoded = encode_state(state)
    with torch.no_grad():
        output = model(encoded.board.unsqueeze(0), encoded.global_features.unsqueeze(0))
        scores = score_actions(output, keys)
    assert scores.shape == (len(moves),)
    assert torch.isfinite(scores).all()


def test_mcts_run_produces_visit_distribution_over_root_children() -> None:
    model = _tiny_model()
    mcts = MCTS(model, rng=np.random.default_rng(0))
    state = GameState.new_game(BASE_PIECE_TYPES)
    legal = generate_legal_moves(state)

    root = mcts.run(state, num_simulations=16, add_root_noise=True)

    assert len(root.children) == len(legal)
    total_child_visits = sum(edge.child.visit_count for edge in root.children.values())
    # Every simulation walks exactly one edge below the root before hitting
    # a not-yet-expanded (freshly created) node, so this should equal the
    # simulation count exactly.
    assert total_child_visits == 16
    assert root.visit_count == 16
    assert -1.0 <= root.value <= 1.0

    # The engine state must be back exactly where it started -- MCTS only
    # ever applies/undoes moves on the shared state during search.
    assert state.current_player == 0
    assert state.ply == 0
    assert state.board == {}


def test_mcts_select_move_is_legal_and_deterministic_at_zero_temperature() -> None:
    model = _tiny_model()
    mcts = MCTS(model, rng=np.random.default_rng(1))
    state = GameState.new_game(BASE_PIECE_TYPES)
    root = mcts.run(state, num_simulations=8)

    legal = generate_legal_moves(state)
    move = select_move(root, temperature=0.0, rng=random.Random(0))
    assert move in legal

    pairs = visit_counts(root)
    best = max(c for _, c in pairs)
    assert any(m == move and c == best for m, c in pairs)


def test_mcts_plays_a_few_plies_without_crashing() -> None:
    """A short game entirely through MCTS.run + select_move + apply_move,
    checking the search survives real mid-game positions (stacks, multiple
    piece types) without shape/index errors -- not a play-strength check."""
    model = _tiny_model()
    mcts = MCTS(model, rng=np.random.default_rng(2))
    rng = random.Random(2)
    state = GameState.new_game(BASE_PIECE_TYPES)

    for _ in range(10):
        if state.game_over:
            break
        root = mcts.run(state, num_simulations=6, add_root_noise=True)
        move = select_move(root, temperature=1.0, rng=rng)
        apply_move(state, move)
