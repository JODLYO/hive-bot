"""Reference-oracle move enumeration, for test_engine_vs_reference.py.

game_state.py and helpers.py in this directory are a vendored, byte-for-byte
copy of the Django app's rules engine. This file is NOT a copy of anything
-- it's a from-scratch, Django-free re-transcription of the orchestration
that in the source app lives on the `GameState` Django model in models.py
(`_board_move_valid`, `_player_has_any_legal_move`, `_update_state_after_move`,
`_check_game_over`), which can't be vendored as-is because it's entangled
with `self.save()` / `self.state_obj` / the ORM. Every actual rules decision
it makes still goes through the vendored `helpers.py` functions -- this
module only reproduces the *dispatch* (which helper to call for which piece
type) and turn bookkeeping, transcribed to match models.py at source commit
43f452c62158015249a6cb1143a196aa741b61a9 as closely as possible, including
its known quirks (see `_board_move_keeps_hive_connected`).
"""

from __future__ import annotations

from dataclasses import dataclass

from hive_bot.engine.constants import PieceType
from hive_bot.engine.state import GameState as FastState
from hive_bot.engine.state import Pos, neighbors

from .game_state import (
    HiveBoardCell,
    HiveBoardState,
    HiveGameState,
    HivePieceState,
    HivePieceType,
    HivePlayerState,
    HivePosition,
)
from .helpers import (
    TURN_NUMBER_QUEEN_MUST_BE_PLACED,
    beetle_move_valid,
    board_move_candidate_positions,
    can_slide_path,
    candidate_placement_positions,
    check_valid_piece_from_hand_move,
    grasshopper_jump_valid,
    hive_is_connected,
    ladybug_move_valid,
    mosquito_move_valid,
    pillbug_throw_valid,
)

OWNER_NAMES = ("p0", "p1")


def to_ref_pos(pos: Pos) -> HivePosition:
    q, r, s = pos
    return HivePosition(q=q, r=r, s=s)


def to_ref_type(piece_type: PieceType) -> HivePieceType:
    return HivePieceType(piece_type.name.lower())


def to_reference_state(state: FastState) -> HiveGameState:
    """Translate the fast engine's GameState into an equivalent reference
    HiveGameState, fresh, for comparison at a single point in time (no
    incremental sync needed)."""
    piece_states: dict[int, HivePieceState] = {}
    for pid, piece in state.pieces.items():
        pos = state.position[pid]
        piece_states[pid] = HivePieceState(
            id=pid,
            piece_type=to_ref_type(piece.piece_type),
            owner=OWNER_NAMES[piece.owner],
            position=to_ref_pos(pos) if pos is not None else None,
            stack_height=state.stack_index_of(pid) if pos is not None else 0,
            placed=pos is not None,
        )

    cells: dict[HivePosition, HiveBoardCell] = {}
    for pos, stack in state.board.items():
        ref_pos = to_ref_pos(pos)
        cells[ref_pos] = HiveBoardCell(
            position=ref_pos, pieces=[piece_states[pid] for pid in stack]
        )

    players = []
    for owner in (0, 1):
        players.append(
            HivePlayerState(
                username=OWNER_NAMES[owner],
                has_placed_queen=state.queen_placed[owner],
                pieces_in_hand=[piece_states[pid] for pid in state.hand[owner]],
                pieces_on_board=[piece_states[pid] for pid in state.pieces_on_board(owner)],
            )
        )

    winner: str | None = None
    if state.winner is not None:
        winner = "draw" if state.winner == -1 else OWNER_NAMES[state.winner]

    return HiveGameState(
        player1_state=players[0],
        player2_state=players[1],
        player1_turn=state.current_player == 0,
        turn_no=state.turn_no,
        board_state=HiveBoardState(cells=cells),
        winner=winner,
        game_over=state.game_over,
        ply=state.ply,
        last_moved_piece_id=state.last_moved_piece_id,
        last_moved_ply=state.last_moved_ply,
    )


@dataclass(frozen=True, slots=True)
class OracleMove:
    kind: str  # "place" | "move" | "throw"
    piece_type: HivePieceType
    from_pos: HivePosition | None
    to_pos: HivePosition
    thrown_type: HivePieceType | None = None
    thrown_from: HivePosition | None = None


def _is_top_of_stack(ref_state: HiveGameState, piece: HivePieceState) -> bool:
    assert piece.position is not None
    cell = ref_state.board_state.cells[piece.position]
    return cell.pieces[-1].id == piece.id


def _board_move_keeps_hive_connected(
    ref_state: HiveGameState, piece: HivePieceState, new_pos: HivePosition
) -> bool:
    """Loosely transcribed from models.py `_board_move_valid`'s pre-check,
    but corrected: the original deletes the piece's *entire* cell (not just
    the top piece) before checking connectivity, even if something else
    remains stacked there. That's a real, reachable bug in the shipped app
    -- a beetle standing on a structurally load-bearing piece gets its
    otherwise-legal moves spuriously rejected -- found by
    test_engine_vs_reference.py while validating the fast engine, which
    does this correctly (see its `_resulting_occupied`). Fixed here too so
    the oracle reflects actual Hive rules rather than that bug; worth
    porting the fix upstream to the Django app separately."""
    assert piece.position is not None
    cell = ref_state.board_state.cells[piece.position]
    temp_cells = dict(ref_state.board_state.cells)
    if len(cell.pieces) == 1:
        del temp_cells[piece.position]
    else:
        temp_cells[piece.position] = HiveBoardCell(
            position=piece.position, pieces=[p for p in cell.pieces if p.id != piece.id]
        )
    return hive_is_connected(HiveBoardState(cells=temp_cells))


def _after_move_hive_connected(
    ref_state: HiveGameState, piece: HivePieceState, new_pos: HivePosition
) -> bool:
    """Transcribed from models.py `GameState.play_piece`'s *second*
    connectivity check -- run after the move actually applies (piece fully
    relocated to `new_pos`, stacking if something is already there), on top
    of (not instead of) `_board_move_valid`'s "during move" pre-check above.
    Missing this one initially produced false-positive oracle approvals for
    beetle moves that don't actually touch the rest of the hive."""
    assert piece.position is not None
    # hive_is_connected reads each piece's own `.position` field (not the
    # dict key it's stored under), so the moved piece needs a copy with
    # `.position` actually updated -- passing the stale `piece` object
    # silently made every destination look "connected" via its old spot.
    moved = piece.model_copy(update={"position": new_pos})
    cell = ref_state.board_state.cells[piece.position]
    temp_cells = dict(ref_state.board_state.cells)
    if len(cell.pieces) == 1:
        del temp_cells[piece.position]
    else:
        temp_cells[piece.position] = HiveBoardCell(
            position=piece.position, pieces=[p for p in cell.pieces if p.id != piece.id]
        )
    if new_pos in temp_cells:
        temp_cells[new_pos] = HiveBoardCell(
            position=new_pos, pieces=[*temp_cells[new_pos].pieces, moved]
        )
    else:
        temp_cells[new_pos] = HiveBoardCell(position=new_pos, pieces=[moved])
    return hive_is_connected(HiveBoardState(cells=temp_cells))


def _oracle_board_move_valid(
    ref_state: HiveGameState,
    player: HivePlayerState,
    piece: HivePieceState,
    new_pos: HivePosition,
) -> bool:
    if not player.has_placed_queen or piece.position is None:
        return False
    if not _board_move_keeps_hive_connected(ref_state, piece, new_pos):
        return False

    valid: bool
    if piece.piece_type == HivePieceType.ANT:
        valid, _ = can_slide_path(ref_state, piece.position, new_pos)
    elif piece.piece_type == HivePieceType.QUEEN:
        valid, _ = can_slide_path(ref_state, piece.position, new_pos, max_steps=1)
    elif piece.piece_type == HivePieceType.SPIDER:
        valid, _ = can_slide_path(
            ref_state, piece.position, new_pos, max_steps=3, require_exact_steps=3
        )
    elif piece.piece_type == HivePieceType.GRASSHOPPER:
        valid = grasshopper_jump_valid(ref_state, piece, new_pos)
    elif piece.piece_type == HivePieceType.BEETLE:
        valid = beetle_move_valid(ref_state, piece, new_pos)
    elif piece.piece_type == HivePieceType.PILLBUG:
        valid, _ = can_slide_path(ref_state, piece.position, new_pos, max_steps=1)
    elif piece.piece_type == HivePieceType.LADYBUG:
        valid = ladybug_move_valid(ref_state, piece, new_pos)
    elif piece.piece_type == HivePieceType.MOSQUITO:
        valid = mosquito_move_valid(ref_state, piece, new_pos)
    else:
        valid = False
    return valid and _after_move_hive_connected(ref_state, piece, new_pos)


def oracle_legal_moves(state: FastState) -> set[OracleMove]:
    """Enumerate every legal move for state.current_player using only the
    vendored reference helpers, for comparison against
    hive_bot.engine.moves.generate_legal_moves on the same position."""
    ref_state = to_reference_state(state)
    owner = state.current_player
    player = ref_state.player1_state if owner == 0 else ref_state.player2_state
    opponent = ref_state.player2_state if owner == 0 else ref_state.player1_state

    moves: set[OracleMove] = set()

    seen_types: set[HivePieceType] = set()
    must_place_queen = (
        not player.has_placed_queen and ref_state.turn_no == TURN_NUMBER_QUEEN_MUST_BE_PLACED
    )
    for piece in player.pieces_in_hand:
        if piece.piece_type in seen_types:
            continue
        if must_place_queen and piece.piece_type != HivePieceType.QUEEN:
            continue
        seen_types.add(piece.piece_type)
        for pos_t in candidate_placement_positions(ref_state.board_state):
            pos = HivePosition(q=pos_t[0], r=pos_t[1], s=pos_t[2])
            valid, _ = check_valid_piece_from_hand_move(ref_state, player, opponent, piece, pos)
            if valid:
                moves.add(OracleMove("place", piece.piece_type, None, pos))

    if player.has_placed_queen:
        all_candidates = board_move_candidate_positions(ref_state.board_state)
        for piece in player.pieces_on_board:
            assert piece.position is not None
            if not _is_top_of_stack(ref_state, piece):
                # The reference oracle's own `_board_move_valid` doesn't
                # check this (only `pillbug_throw_valid` checks it, and only
                # for the piece being thrown) -- a real gap in the shipped
                # app, since only the top of a stack should ever be able to
                # move. Filtered out here rather than reproduced, since the
                # fast engine correctly enforces it and shouldn't be
                # penalized in this comparison for being right.
                continue
            for pos_t in all_candidates - {(piece.position.q, piece.position.r, piece.position.s)}:
                pos = HivePosition(q=pos_t[0], r=pos_t[1], s=pos_t[2])
                if _oracle_board_move_valid(ref_state, player, piece, pos):
                    moves.add(OracleMove("move", piece.piece_type, piece.position, pos))

        for pillbug in player.pieces_on_board:
            if pillbug.piece_type != HivePieceType.PILLBUG:
                continue
            if not _is_top_of_stack(ref_state, pillbug):
                continue
            assert pillbug.position is not None
            all_pieces = player.pieces_on_board + opponent.pieces_on_board
            for target in all_pieces:
                if target.id == pillbug.id:
                    continue
                assert target.position is not None
                for dest in neighbors(
                    (pillbug.position.q, pillbug.position.r, pillbug.position.s)
                ):
                    dest_pos = HivePosition(q=dest[0], r=dest[1], s=dest[2])
                    valid, _ = pillbug_throw_valid(ref_state, pillbug, target, dest_pos)
                    if valid:
                        moves.add(
                            OracleMove(
                                "throw",
                                pillbug.piece_type,
                                pillbug.position,
                                dest_pos,
                                thrown_type=target.piece_type,
                                thrown_from=target.position,
                            )
                        )

    return moves
