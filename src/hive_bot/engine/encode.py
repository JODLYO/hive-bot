"""GameState -> tensor encoding, always from the current player's
perspective ("me" vs "opponent" rather than raw owner 0/1) -- the standard
choice for self-play networks, since it means one policy/value head serves
both sides instead of having to learn mirrored behavior per player.

Also owns the Pos <-> grid-cell mapping (`pos_to_index`/`index_to_pos`),
which `actions.py` builds its move<->action-index mapping on top of.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .constants import (
    BOARD_DIM,
    BOARD_RADIUS,
    MAX_STACK_HEIGHT,
    NUM_PIECE_TYPES,
    PIECE_COUNTS,
    PieceType,
)
from .state import GameState, Pos

# Per-cell (spatial) channels: top-piece one-hot (mine, then opponent's) +
# normalized stack height + "this cell's top piece moved last ply" (the
# pillbug freeze rule's visible signal).
NUM_SPATIAL_CHANNELS = 2 * NUM_PIECE_TYPES + 2
_STACK_HEIGHT_CHANNEL = 2 * NUM_PIECE_TYPES
_LAST_MOVED_CHANNEL = 2 * NUM_PIECE_TYPES + 1

# Global (non-spatial) features: hand counts per type (mine, then
# opponent's), each side's queen-placed flag, and a normalized turn number.
NUM_GLOBAL_FEATURES = 2 * NUM_PIECE_TYPES + 3
_TURN_NORMALIZATION = 50.0


def pos_to_cell(pos: Pos) -> tuple[int, int]:
    """Cube coordinate -> (row, col) in the BOARD_DIM x BOARD_DIM grid."""
    q, r, _s = pos
    return r + BOARD_RADIUS, q + BOARD_RADIUS


def cell_to_pos(row: int, col: int) -> Pos:
    q = col - BOARD_RADIUS
    r = row - BOARD_RADIUS
    return q, r, -q - r


def pos_to_index(pos: Pos) -> int:
    row, col = pos_to_cell(pos)
    return row * BOARD_DIM + col


def index_to_pos(index: int) -> Pos:
    row, col = divmod(index, BOARD_DIM)
    return cell_to_pos(row, col)


class EncodedState(NamedTuple):
    board: torch.Tensor  # (NUM_SPATIAL_CHANNELS, BOARD_DIM, BOARD_DIM) float32
    global_features: torch.Tensor  # (NUM_GLOBAL_FEATURES,) float32


def encode_state(state: GameState) -> EncodedState:
    me = state.current_player
    opponent = 1 - me

    board = torch.zeros((NUM_SPATIAL_CHANNELS, BOARD_DIM, BOARD_DIM), dtype=torch.float32)
    for pos, stack in state.board.items():
        row, col = pos_to_cell(pos)
        top_id = stack[-1]
        piece = state.pieces[top_id]
        owner_offset = 0 if piece.owner == me else NUM_PIECE_TYPES
        board[owner_offset + piece.piece_type.value, row, col] = 1.0
        board[_STACK_HEIGHT_CHANNEL, row, col] = len(stack) / MAX_STACK_HEIGHT
        if top_id == state.last_moved_piece_id and state.last_moved_ply == state.ply:
            board[_LAST_MOVED_CHANNEL, row, col] = 1.0

    global_features = torch.tensor(
        [
            *_hand_counts(state, me),
            *_hand_counts(state, opponent),
            float(state.queen_placed[me]),
            float(state.queen_placed[opponent]),
            min(state.turn_no, _TURN_NORMALIZATION) / _TURN_NORMALIZATION,
        ],
        dtype=torch.float32,
    )
    return EncodedState(board=board, global_features=global_features)


def _hand_counts(state: GameState, owner: int) -> list[float]:
    counts = [0] * NUM_PIECE_TYPES
    for pid in state.hand[owner]:
        counts[state.pieces[pid].piece_type.value] += 1
    return [count / PIECE_COUNTS[PieceType(i)] for i, count in enumerate(counts)]
