"""Make/unmake move application for the fast Hive engine.

`apply_move` mutates a `GameState` in place and returns an `UndoInfo`;
`undo_move` reverses it exactly. This avoids the reference oracle's
deep-copy-the-whole-board-per-candidate approach, which is the main reason
the fast engine exists (see the plan doc / README) -- MCTS self-play needs
to apply and revert many thousands of moves per second.

Also owns the turn-structure bookkeeping ported from the reference's
`GameState._update_state_after_move`/`_check_game_over`: forced queen
placement, the ply/last-moved bookkeeping the pillbug freeze rule reads,
auto-passing a player with no legal move, and the double-surrounded-queen
draw.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import MAX_PLIES_BEFORE_DRAW, REPETITION_LIMIT, PieceType
from .moves import Move, MoveKind, generate_legal_moves
from .state import DRAW, GameState, Pos, PositionKey, neighbors, position_key


@dataclass(slots=True)
class UndoInfo:
    move: Move
    moved_piece_id: int  # the piece whose position/hand status changed
    prev_position: Pos | None  # None means it was in hand before this move
    prev_hand_index: int | None  # index to reinsert at, only set if prev_position is None
    prev_current_player: int
    prev_turn_no: int
    prev_ply: int
    prev_last_moved_piece_id: int | None
    prev_last_moved_ply: int | None
    prev_game_over: bool
    prev_winner: int | None
    prev_queen_placed_mover: bool
    position_key: PositionKey | None = None  # set once computed, for undo to decrement


def apply_move(state: GameState, move: Move) -> UndoInfo:
    assert not state.game_over
    mover = state.current_player
    acted_piece_id = move.thrown_piece_id if move.kind == MoveKind.THROW else move.piece_id
    assert acted_piece_id is not None
    acted_owner = state.pieces[acted_piece_id].owner

    prev_position = state.position[acted_piece_id]
    undo = UndoInfo(
        move=move,
        moved_piece_id=acted_piece_id,
        prev_position=prev_position,
        prev_hand_index=None,
        prev_current_player=state.current_player,
        prev_turn_no=state.turn_no,
        prev_ply=state.ply,
        prev_last_moved_piece_id=state.last_moved_piece_id,
        prev_last_moved_ply=state.last_moved_ply,
        prev_game_over=state.game_over,
        prev_winner=state.winner,
        prev_queen_placed_mover=state.queen_placed[acted_owner],
    )

    if prev_position is None:
        hand = state.hand[acted_owner]
        undo.prev_hand_index = hand.index(acted_piece_id)
        del hand[undo.prev_hand_index]
        if state.pieces[acted_piece_id].piece_type == PieceType.QUEEN:
            state.queen_placed[acted_owner] = True
    else:
        stack = state.board[prev_position]
        stack.pop()
        if not stack:
            del state.board[prev_position]

    dest = move.to
    if dest in state.board:
        state.board[dest].append(acted_piece_id)
    else:
        state.board[dest] = [acted_piece_id]
    state.position[acted_piece_id] = dest

    state.ply += 1
    state.last_moved_piece_id = acted_piece_id
    state.last_moved_ply = state.ply

    if mover == 1:
        state.turn_no += 1
    state.current_player = 1 - mover

    key = position_key(state)
    count = state.position_counts.get(key, 0) + 1
    state.position_counts[key] = count
    undo.position_key = key

    _check_game_over(state)
    if not state.game_over and (
        count >= REPETITION_LIMIT or state.ply >= MAX_PLIES_BEFORE_DRAW
    ):
        state.game_over = True
        state.winner = DRAW
    _resolve_auto_pass(state)

    return undo


def undo_move(state: GameState, undo: UndoInfo) -> None:
    piece_id = undo.moved_piece_id
    owner = state.pieces[piece_id].owner

    assert undo.position_key is not None
    count = state.position_counts[undo.position_key]
    if count <= 1:
        del state.position_counts[undo.position_key]
    else:
        state.position_counts[undo.position_key] = count - 1

    dest = state.position[piece_id]
    assert dest is not None
    stack = state.board[dest]
    stack.pop()
    if not stack:
        del state.board[dest]

    if undo.prev_position is None:
        assert undo.prev_hand_index is not None
        state.hand[owner].insert(undo.prev_hand_index, piece_id)
        state.position[piece_id] = None
    else:
        prev = undo.prev_position
        if prev in state.board:
            state.board[prev].append(piece_id)
        else:
            state.board[prev] = [piece_id]
        state.position[piece_id] = prev

    state.queen_placed[owner] = undo.prev_queen_placed_mover
    state.current_player = undo.prev_current_player
    state.turn_no = undo.prev_turn_no
    state.ply = undo.prev_ply
    state.last_moved_piece_id = undo.prev_last_moved_piece_id
    state.last_moved_ply = undo.prev_last_moved_ply
    state.game_over = undo.prev_game_over
    state.winner = undo.prev_winner


def _check_game_over(state: GameState) -> None:
    q0 = _queen_id(state, 0)
    q1 = _queen_id(state, 1)
    s0 = q0 is not None and _is_queen_surrounded(state, q0)
    s1 = q1 is not None and _is_queen_surrounded(state, q1)
    if not s0 and not s1:
        return
    state.game_over = True
    if s0 and s1:
        state.winner = DRAW
    else:
        state.winner = 1 if s0 else 0


def _queen_id(state: GameState, owner: int) -> int | None:
    for pid, piece in state.pieces.items():
        if (
            piece.owner == owner
            and piece.piece_type == PieceType.QUEEN
            and state.position[pid] is not None
        ):
            return pid
    return None


def _is_queen_surrounded(state: GameState, queen_id: int) -> bool:
    pos = state.position[queen_id]
    assert pos is not None
    occupied = state.occupied()
    return all(nb in occupied for nb in neighbors(pos))


def _resolve_auto_pass(state: GameState) -> None:
    """A player with no legal placement or move at all forfeits their turn.
    Two attempts cover both players once; if neither can act, it's a
    mutual deadlock (draw)."""
    for _ in range(2):
        if state.game_over:
            return
        if generate_legal_moves(state):
            return
        if state.current_player == 1:
            state.turn_no += 1
        state.current_player = 1 - state.current_player
    state.game_over = True
    state.winner = DRAW
