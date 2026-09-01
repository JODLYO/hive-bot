"""Move <-> policy-action-index mapping for the network's policy head.

Hive's action space (which piece moves where) doesn't fit a fixed-size,
dense policy vector the way chess/shogi's per-square move planes do, since
slides are pathfinding-based rather than fixed-direction -- see the plan
doc. The chosen design is a dual-embedding bilinear head: the network
produces a `from`-embedding and a `to`-embedding per grid cell (plus 8
extra learned embeddings for "place this piece type from hand"), and a
move's score is the dot product of its from/to embeddings. That only needs
this module to map each `Move` to a small `(kind, from_index, to_index)`
key -- there's no dense action vector to build or index into here; MCTS and
training gather scores for exactly the legal keys of a position rather than
masking a huge (from x to) matrix.

PLACE actions are keyed by piece *type* (a fixed hand slot, 0..7), not a
specific piece instance -- see moves.py for why in-hand pieces of the same
type are interchangeable. MOVE and THROW are keyed by the acting piece's
current board position, which is always unique (at most one top-of-stack
piece per cell), so within one position's legal-move list no two different
moves can ever produce the same `(kind, from_index, to_index)` key -- see
test_actions.py.
"""

from __future__ import annotations

from .encode import pos_to_index
from .moves import Move, MoveKind
from .state import GameState

# (kind, from_index, to_index). PLACE's from_index is a hand slot (a
# PieceType's .value, 0..7); MOVE/THROW's is a board cell index.
ActionKey = tuple[MoveKind, int, int]


def move_to_action_key(state: GameState, move: Move) -> ActionKey:
    to_index = pos_to_index(move.to)
    if move.kind == MoveKind.PLACE:
        from_index = state.pieces[move.piece_id].piece_type.value
    elif move.kind == MoveKind.MOVE:
        start = state.position[move.piece_id]
        assert start is not None
        from_index = pos_to_index(start)
    else:
        assert move.thrown_piece_id is not None
        start = state.position[move.thrown_piece_id]
        assert start is not None
        from_index = pos_to_index(start)
    return move.kind, from_index, to_index


def legal_action_keys(state: GameState, moves: list[Move]) -> list[ActionKey]:
    """Action keys parallel to `moves` (same order, same length) -- the
    intended usage is zipping `moves[i]` with `keys[i]`, e.g. to gather
    network scores for exactly these actions and sample an index into
    `moves` directly, rather than decoding a key back into a move."""
    return [move_to_action_key(state, m) for m in moves]
