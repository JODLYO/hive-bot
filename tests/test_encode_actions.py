"""Unit tests for encode.py's grid mapping/tensor shapes and actions.py's
move<->action-key mapping, independent of the reference oracle."""

from __future__ import annotations

import random

import pytest
import torch

from hive_bot.engine.actions import legal_action_keys, move_to_action_key
from hive_bot.engine.apply import apply_move
from hive_bot.engine.constants import BASE_PIECE_TYPES, BOARD_DIM, BOARD_RADIUS, PieceType
from hive_bot.engine.encode import (
    NUM_GLOBAL_FEATURES,
    NUM_SPATIAL_CHANNELS,
    cell_to_pos,
    encode_state,
    index_to_pos,
    pos_to_cell,
    pos_to_index,
)
from hive_bot.engine.moves import MoveKind, generate_legal_moves
from hive_bot.engine.state import GameState


@pytest.mark.parametrize(
    "pos",
    [
        (0, 0, 0),
        (1, -1, 0),
        (-1, 0, 1),
        (BOARD_RADIUS, -BOARD_RADIUS, 0),
        (-BOARD_RADIUS, 0, BOARD_RADIUS),
    ],
)
def test_pos_cell_roundtrip(pos: tuple[int, int, int]) -> None:
    row, col = pos_to_cell(pos)
    assert 0 <= row < BOARD_DIM
    assert 0 <= col < BOARD_DIM
    assert cell_to_pos(row, col) == pos


@pytest.mark.parametrize(
    "pos",
    [
        (0, 0, 0),
        (1, -1, 0),
        (BOARD_RADIUS, -BOARD_RADIUS, 0),
        (-BOARD_RADIUS, BOARD_RADIUS, 0),
    ],
)
def test_pos_index_roundtrip(pos: tuple[int, int, int]) -> None:
    index = pos_to_index(pos)
    assert 0 <= index < BOARD_DIM * BOARD_DIM
    assert index_to_pos(index) == pos


def test_encode_state_shapes_and_ranges() -> None:
    state = GameState.new_game(BASE_PIECE_TYPES)
    encoded = encode_state(state)
    assert encoded.board.shape == (NUM_SPATIAL_CHANNELS, BOARD_DIM, BOARD_DIM)
    assert encoded.global_features.shape == (NUM_GLOBAL_FEATURES,)
    assert encoded.board.dtype == torch.float32
    assert torch.all(encoded.board >= 0) and torch.all(encoded.board <= 1)
    assert torch.all(encoded.global_features >= 0) and torch.all(
        encoded.global_features <= 1
    )
    # Empty board: no piece occupies any cell yet.
    assert encoded.board.sum() == 0


def test_encode_state_reflects_placed_piece_from_movers_perspective() -> None:
    state = GameState.new_game(BASE_PIECE_TYPES)
    moves = generate_legal_moves(state)
    queen_move = next(
        m
        for m in moves
        if state.pieces[m.piece_id].piece_type == PieceType.QUEEN and m.to == (0, 0, 0)
    )
    apply_move(state, queen_move)  # player 0 places at origin; now player 1 to move

    encoded = encode_state(state)
    row, col = pos_to_cell((0, 0, 0))
    # Placed piece belongs to player 0, but player 1 is now "me" -- so it
    # must show up in the *opponent* one-hot block, not the "mine" one.
    from hive_bot.engine.constants import NUM_PIECE_TYPES

    assert encoded.board[PieceType.QUEEN.value, row, col] == 0.0
    assert encoded.board[NUM_PIECE_TYPES + PieceType.QUEEN.value, row, col] == 1.0
    # It was also the last piece moved.
    assert encoded.board[NUM_SPATIAL_CHANNELS - 1, row, col] == 1.0


def test_action_keys_are_unique_and_placements_share_kind_across_positions() -> None:
    """Every legal move at a position must get a distinct action key -- see
    actions.py's uniqueness argument (position uniqueness for MOVE/THROW,
    type-dedup for PLACE)."""
    rng = random.Random(7)
    state = GameState.new_game(BASE_PIECE_TYPES)
    for _ in range(30):
        if state.game_over:
            break
        moves = generate_legal_moves(state)
        keys = legal_action_keys(state, moves)
        assert len(keys) == len(set(keys)), f"duplicate action keys: {keys}"
        move = rng.choice(moves)
        apply_move(state, move)


def test_move_to_action_key_place_uses_type_not_instance() -> None:
    state = GameState.new_game(BASE_PIECE_TYPES)
    moves = generate_legal_moves(state)
    ant_moves = [m for m in moves if state.pieces[m.piece_id].piece_type == PieceType.ANT]
    assert len(ant_moves) == 1  # deduped to one representative per type/destination
    kind, from_index, to_index = move_to_action_key(state, ant_moves[0])
    assert kind == MoveKind.PLACE
    assert from_index == PieceType.ANT.value
    assert to_index == pos_to_index((0, 0, 0))
