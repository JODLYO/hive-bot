"""Property-based parity tests: random games played through the fast engine
must agree with the vendored reference oracle on the exact set of legal
moves at every single ply. This is the correctness gate described in the
plan doc -- the fast engine reimplements the rules rather than reusing the
reference, so this is what earns the right to trust it for self-play.
"""

from __future__ import annotations

import random

import pytest

from hive_bot.engine.apply import apply_move
from hive_bot.engine.constants import BASE_PIECE_TYPES, EXPANSION_PIECE_TYPES, PieceType
from hive_bot.engine.moves import Move, MoveKind, generate_legal_moves
from hive_bot.engine.state import GameState, Pos

from .reference.game_state import HivePosition
from .reference.oracle import OracleMove, oracle_legal_moves, to_ref_type

FULL_PIECE_SET = BASE_PIECE_TYPES | EXPANSION_PIECE_TYPES

# The oracle side of this comparison is deliberately unoptimized (it
# re-validates every (piece, candidate) pair from scratch each ply, the same
# way the reference app does), so its cost grows sharply as more pieces hit
# the board. Capped low enough to keep the full parametrized suite in the
# tens-of-seconds range; this still exercises placement, every base piece's
# movement, stacking, and the midgame most heavily.
MAX_PLIES = 40


def _pos(p: Pos) -> HivePosition:
    q, r, s = p
    return HivePosition(q=q, r=r, s=s)


def _fast_move_to_oracle_key(state: GameState, move: Move) -> OracleMove:
    piece_type = to_ref_type(state.pieces[move.piece_id].piece_type)
    if move.kind == MoveKind.PLACE:
        return OracleMove("place", piece_type, None, _pos(move.to))
    if move.kind == MoveKind.MOVE:
        start = state.position[move.piece_id]
        assert start is not None
        return OracleMove("move", piece_type, _pos(start), _pos(move.to))
    assert move.thrown_piece_id is not None
    pillbug_pos = state.position[move.piece_id]
    thrown_pos = state.position[move.thrown_piece_id]
    assert pillbug_pos is not None and thrown_pos is not None
    return OracleMove(
        "throw",
        piece_type,
        _pos(pillbug_pos),
        _pos(move.to),
        thrown_type=to_ref_type(state.pieces[move.thrown_piece_id].piece_type),
        thrown_from=_pos(thrown_pos),
    )


def _assert_moves_match(state: GameState, ply: int, seed: int) -> list[Move]:
    fast_moves = generate_legal_moves(state)
    fast_keys = {_fast_move_to_oracle_key(state, m): m for m in fast_moves}
    oracle_keys = oracle_legal_moves(state)

    only_fast = set(fast_keys) - oracle_keys
    only_oracle = oracle_keys - set(fast_keys)
    assert not only_fast and not only_oracle, (
        f"seed={seed} ply={ply} current_player={state.current_player}\n"
        f"fast engine allows but oracle rejects: {only_fast}\n"
        f"oracle allows but fast engine rejects: {only_oracle}"
    )
    return fast_moves


def _play_random_game(
    seed: int, enabled_types: frozenset[PieceType], max_plies: int = MAX_PLIES
) -> None:
    rng = random.Random(seed)
    state = GameState.new_game(enabled_types)

    for ply in range(max_plies):
        if state.game_over:
            return
        fast_moves = _assert_moves_match(state, ply, seed)
        assert fast_moves, (
            "engines agree on legality but produced zero moves for a live game"
        )
        move = rng.choice(fast_moves)
        apply_move(state, move)


@pytest.mark.parametrize("seed", range(3))
def test_random_games_match_oracle_base_pieces_smoke(seed: int) -> None:
    """Fast subset that runs by default (`make test`) -- few seeds, capped
    early game, just enough to catch a regression quickly. The exhaustive
    version below is the real correctness gate; run it with `make
    test-full` (or `pytest -m slow`) before trusting engine changes."""
    _play_random_game(seed, BASE_PIECE_TYPES, max_plies=15)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(15))
def test_random_games_match_oracle_base_pieces(seed: int) -> None:
    _play_random_game(seed, BASE_PIECE_TYPES)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(6))
def test_random_games_match_oracle_full_expansion(seed: int) -> None:
    _play_random_game(seed, FULL_PIECE_SET)
