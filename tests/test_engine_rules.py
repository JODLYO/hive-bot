"""Direct unit tests for fast-engine behavior that isn't (or can't easily
be) covered by the randomized oracle-parity suite: the self-play-only
move-count cap and threefold-repetition draw rules (see constants.py), and
that apply_move/undo_move are exact inverses of each other."""

from __future__ import annotations

from hive_bot.engine.apply import apply_move, undo_move
from hive_bot.engine.constants import BASE_PIECE_TYPES, MAX_PLIES_BEFORE_DRAW, PieceType
from hive_bot.engine.moves import Move, MoveKind, generate_legal_moves
from hive_bot.engine.state import DRAW, GameState


def _find_move(state: GameState, piece_id: int, to: tuple[int, int, int]) -> Move:
    move = Move(MoveKind.MOVE, piece_id, to)
    assert move in generate_legal_moves(state), f"expected {move} to be legal"
    return move


def test_move_cap_forces_a_draw() -> None:
    state = GameState.new_game(BASE_PIECE_TYPES)
    state.ply = MAX_PLIES_BEFORE_DRAW - 1

    queen_move = next(
        m
        for m in generate_legal_moves(state)
        if state.pieces[m.piece_id].piece_type == PieceType.QUEEN
    )
    apply_move(state, queen_move)

    assert state.game_over
    assert state.winner == DRAW


def test_threefold_repetition_forces_a_draw() -> None:
    state = GameState.new_game(BASE_PIECE_TYPES)

    p0_queen = next(
        pid
        for pid, p in state.pieces.items()
        if p.owner == 0 and p.piece_type == PieceType.QUEEN
    )
    p1_queen = next(
        pid
        for pid, p in state.pieces.items()
        if p.owner == 1 and p.piece_type == PieceType.QUEEN
    )
    p0_ant = next(
        pid
        for pid, p in state.pieces.items()
        if p.owner == 0 and p.piece_type == PieceType.ANT
    )

    apply_move(state, Move(MoveKind.PLACE, p0_queen, (0, 0, 0)))
    apply_move(state, Move(MoveKind.PLACE, p1_queen, (1, -1, 0)))
    apply_move(state, Move(MoveKind.PLACE, p0_ant, (-1, 0, 1)))
    # P1's second real turn: oscillate their queen between two of P0's
    # queen's neighbors (both stay adjacent to the stationary P0 queen, so
    # the hive never disconnects regardless of where P0's ant is).
    apply_move(state, _find_move(state, p1_queen, (1, 0, -1)))

    # A repeating 4-ply cycle that returns the *entire* position (board +
    # hands + side to move) to exactly what it was after the moves above:
    # P0's ant shuttles between two of P0's queen's other neighbors, P1's
    # queen shuttles back and forth between the two spots picked above.
    for _ in range(3):
        if state.game_over:
            break
        apply_move(state, _find_move(state, p0_ant, (-1, 1, 0)))
        if state.game_over:
            break
        apply_move(state, _find_move(state, p1_queen, (1, -1, 0)))
        if state.game_over:
            break
        apply_move(state, _find_move(state, p0_ant, (-1, 0, 1)))
        if state.game_over:
            break
        apply_move(state, _find_move(state, p1_queen, (1, 0, -1)))

    assert state.game_over
    assert state.winner == DRAW


def test_apply_undo_is_an_exact_inverse_through_a_draw_by_repetition() -> None:
    """undo_move must perfectly reverse a move even when that move was the
    one that set game_over/winner via the new draw rules, since MCTS relies
    on this for its make/unmake tree traversal."""
    state = GameState.new_game(BASE_PIECE_TYPES)
    state.ply = MAX_PLIES_BEFORE_DRAW - 1
    before = (
        {pos: list(stack) for pos, stack in state.board.items()},
        [list(h) for h in state.hand],
        dict(state.position_counts),
        state.current_player,
        state.turn_no,
        state.ply,
        state.game_over,
        state.winner,
    )

    move = next(
        m
        for m in generate_legal_moves(state)
        if state.pieces[m.piece_id].piece_type == PieceType.QUEEN
    )
    undo = apply_move(state, move)
    assert state.game_over  # hit the move cap

    undo_move(state, undo)

    after = (
        {pos: list(stack) for pos, stack in state.board.items()},
        [list(h) for h in state.hand],
        dict(state.position_counts),
        state.current_player,
        state.turn_no,
        state.ply,
        state.game_over,
        state.winner,
    )
    assert before == after
