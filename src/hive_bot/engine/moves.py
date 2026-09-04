"""Legal move generation for the fast Hive engine.

Ports the same rules as tests/reference/helpers.py (slide BFS with the
wedge/gate check, hive connectivity, grasshopper jump, beetle/mosquito
climbing, ladybug's fixed 3-step climb-climb-drop, pillbug throw, turn
structure rules), but computes each piece's full set of legal destinations
in one pass instead of validating one candidate destination at a time --
see `_slide_reachable`. Validated against the reference oracle in
tests/test_engine_vs_reference.py.

Placement actions are generated per piece *type*, not per specific piece
instance: two in-hand pieces of the same type are provably interchangeable
(same owner, same rules, no position yet), so instancing them as separate
actions would only bloat the action space without adding real choices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from .constants import HEX_DIRS, TURN_NUMBER_QUEEN_MUST_BE_PLACED, PieceType
from .state import GameState, Pos, neighbors, positions_connected, shared_neighbors


class MoveKind(IntEnum):
    PLACE = 0
    MOVE = 1
    THROW = 2


@dataclass(frozen=True, slots=True)
class Move:
    kind: MoveKind
    piece_id: int  # placed/moved piece, or the pillbug itself for THROW
    to: Pos
    thrown_piece_id: int | None = None  # THROW only: the piece being lifted


def generate_legal_moves(state: GameState) -> list[Move]:
    if state.game_over:
        return []
    moves = _placement_moves(state)
    if state.queen_placed[state.current_player]:
        moves += _board_moves(state)
        moves += _pillbug_throw_moves(state)
    return moves


# --- placement -----------------------------------------------------------


def _placement_moves(state: GameState) -> list[Move]:
    owner = state.current_player
    hand = state.hand[owner]
    if not hand:
        return []
    positions = _placement_positions(state, owner)
    if not positions:
        return []
    must_place_queen = (
        not state.queen_placed[owner] and state.turn_no == TURN_NUMBER_QUEEN_MUST_BE_PLACED
    )
    moves: list[Move] = []
    seen_types: set[PieceType] = set()
    for piece_id in hand:
        piece_type = state.pieces[piece_id].piece_type
        if piece_type in seen_types:
            continue
        if must_place_queen and piece_type != PieceType.QUEEN:
            continue
        seen_types.add(piece_type)
        moves.extend(Move(MoveKind.PLACE, piece_id, pos) for pos in positions)
    return moves


def _placement_positions(state: GameState, owner: int) -> set[Pos]:
    occupied = state.occupied()
    if not occupied:
        return {(0, 0, 0)}

    opponent = 1 - owner
    if state.turn_no == 1:
        # Each side's very first placement just needs to touch the hive so
        # far, which (on turn 1) can only be the opponent's lone piece.
        anchors = state.board_positions(opponent)
        return {nb for pos in anchors for nb in neighbors(pos) if nb not in occupied}

    own_positions = state.board_positions(owner)
    opp_positions = state.board_positions(opponent)
    candidates = {
        nb for pos in own_positions for nb in neighbors(pos) if nb not in occupied
    }
    return {c for c in candidates if all(nb not in opp_positions for nb in neighbors(c))}


# --- on-board moves --------------------------------------------------------


def _board_moves(state: GameState) -> list[Move]:
    owner = state.current_player
    moves: list[Move] = []
    for piece_id in state.pieces_on_board(owner):
        pos = state.position[piece_id]
        assert pos is not None
        if state.top_piece_at(pos) != piece_id:
            continue  # buried under a beetle/mosquito, can't move
        piece_type = state.pieces[piece_id].piece_type
        for dest in _destinations_for(state, piece_id, piece_type):
            if _move_keeps_hive_connected(state, piece_id, dest):
                moves.append(Move(MoveKind.MOVE, piece_id, dest))
    return moves


def _destinations_for(state: GameState, piece_id: int, piece_type: PieceType) -> set[Pos]:
    if piece_type in (PieceType.QUEEN, PieceType.PILLBUG):
        return _slide_destinations(state, piece_id, max_steps=1)
    if piece_type == PieceType.ANT:
        return _slide_destinations(state, piece_id, max_steps=None)
    if piece_type == PieceType.SPIDER:
        return _slide_destinations(state, piece_id, max_steps=3, exact_steps=3)
    if piece_type == PieceType.GRASSHOPPER:
        return _grasshopper_destinations(state, piece_id)
    if piece_type == PieceType.BEETLE:
        return _beetle_destinations(state, piece_id)
    if piece_type == PieceType.LADYBUG:
        return _ladybug_destinations(state, piece_id)
    if piece_type == PieceType.MOSQUITO:
        return _mosquito_destinations(state, piece_id)
    raise AssertionError(f"unhandled piece type {piece_type!r}")


def _resulting_occupied(state: GameState, piece_id: int, dest: Pos) -> frozenset[Pos]:
    """Occupied-position set after hypothetically moving `piece_id` to
    `dest`. Only vacates the start cell if this piece was its sole
    occupant -- a piece stacked under a beetle/mosquito keeps that position
    "occupied" for connectivity purposes when the top piece leaves.
    """
    start = state.position[piece_id]
    assert start is not None
    occ = set(state.occupied())
    if state.stack_height_at(start) == 1:
        occ.discard(start)
    occ.add(dest)
    return frozenset(occ)


def _move_keeps_hive_connected(state: GameState, piece_id: int, dest: Pos) -> bool:
    """The official "One Hive Rule" requires connectivity to hold both
    *during* a move (the instant the piece is lifted, before it lands --
    real Hive rules forbid a move that would momentarily split the hive,
    even if the piece's destination would reconnect it) and *after* it
    (the final board). Only matters "during" when the start cell truly
    empties (stack height 1); a beetle/mosquito climbing off a taller stack
    never affects connectivity by leaving, since something remains there.

    tests/reference/oracle.py's `_board_move_keeps_hive_connected` /
    `_after_move_hive_connected` mirror this same two-check structure
    (corrected from the reference oracle's original, which applied the
    "during" check unconditionally even to stacked-piece departures that
    don't actually vacate their cell -- a real, reachable bug in the
    shipped Django app; see test_engine_vs_reference.py)."""
    start = state.position[piece_id]
    assert start is not None
    if state.stack_height_at(start) == 1:
        during = set(state.occupied())
        during.discard(start)
        if not positions_connected(frozenset(during)):
            return False
    return positions_connected(_resulting_occupied(state, piece_id, dest))


def _slide_destinations(
    state: GameState,
    piece_id: int,
    max_steps: int | None,
    exact_steps: int | None = None,
) -> set[Pos]:
    start = state.position[piece_id]
    assert start is not None
    occupied_without_start = state.occupied() - {start}
    return _slide_reachable(occupied_without_start, start, max_steps, exact_steps)


def _slide_reachable(
    occupied: frozenset[Pos],
    start: Pos,
    max_steps: int | None,
    exact_steps: int | None,
) -> set[Pos]:
    """All destinations reachable from `start` by sliding across `occupied`
    (which must NOT include `start` itself -- the piece is lifted first).
    Exact-step sliding (spider) needs a different search than
    unbounded/max-step sliding (ant/queen) -- see `_slide_reachable_exact`
    for why -- so this just dispatches to whichever is correct.
    """
    if exact_steps is not None:
        return _slide_reachable_exact(occupied, start, exact_steps)

    # A single BFS pass computes every reachable destination instead of
    # validating one candidate at a time: since the wedge/gate check and
    # the "hive stays connected" check at each step only depend on the
    # current node and the single candidate next node (not on the path
    # taken to get there), and *any* non-self-crossing path to a node
    # within max_steps is equally good enough (there's no "must be exactly
    # this many steps" requirement here), a node's reachability doesn't
    # depend on which path first discovered it -- so a single shared
    # `visited` set across the whole search is exact, not an
    # approximation. (This reasoning does NOT extend to exact-step
    # sliding -- see `_slide_reachable_exact`.)
    visited = {start}
    reachable: set[Pos] = set()
    queue: deque[tuple[Pos, int]] = deque([(start, 0)])
    while queue:
        current, steps = queue.popleft()
        for nb in neighbors(current):
            if nb in visited:
                continue
            visited.add(nb)
            if nb in occupied:
                continue
            if not positions_connected(occupied | {nb}):
                continue
            n1, n2 = shared_neighbors(current, nb)
            if n1 in occupied and n2 in occupied:
                continue  # wedged: both flanking hexes occupied

            new_steps = steps + 1
            if max_steps is None or new_steps <= max_steps:
                reachable.add(nb)
            if max_steps is None or new_steps < max_steps:
                queue.append((nb, new_steps))
    return reachable


def _slide_reachable_exact(
    occupied: frozenset[Pos], start: Pos, exact_steps: int
) -> set[Pos]:
    """Destinations reachable from `start` in *exactly* `exact_steps`
    slides (spider), without ever crossing a hex already visited earlier
    in that same slide.

    Unlike max-step sliding, this can't reuse one shared `visited` set
    across the whole search: a hex first reached via a *shorter* path
    would then block rediscovering it via a *different*, longer,
    non-self-crossing path that legitimately reaches it in exactly
    `exact_steps` -- "no revisiting a hex" is a per-path rule, not a
    global one across every path from `start`. A real game found this
    concretely: a destination the shared-visited BFS marked unreachable
    (some other path had already visited it after 2 steps) but which a
    valid 3-step path -- through different intermediate hexes -- does
    reach. So this instead explores every non-self-crossing path
    explicitly; cheap in practice since `exact_steps` is always 3 (only
    spider uses this) with branching factor at most 6.

    This exact bug is present in the vendored reference's own
    `can_slide_path` (tests/reference/helpers.py -- an unmodified copy of
    the real ttbg-web-app's own code, not just this repo's re-transcribed
    oracle), so it isn't just a bug here -- it's a real, reachable bug in
    the shipped Django app itself. test_engine_vs_reference.py's oracle
    comparison inherits it via `tests/reference/oracle.py`, which calls
    the same `can_slide_path` -- see that test file for how spider moves
    are carved out of the direct oracle comparison as a result.
    """
    reachable: set[Pos] = set()

    def dfs(current: Pos, path: frozenset[Pos], steps: int) -> None:
        if steps == exact_steps:
            reachable.add(current)
            return
        for nb in neighbors(current):
            if nb in path or nb in occupied:
                continue
            if not positions_connected(occupied | {nb}):
                continue
            n1, n2 = shared_neighbors(current, nb)
            if n1 in occupied and n2 in occupied:
                continue  # wedged: both flanking hexes occupied
            dfs(nb, path | {nb}, steps + 1)

    dfs(start, frozenset({start}), 0)
    return reachable


def _grasshopper_destinations(state: GameState, piece_id: int) -> set[Pos]:
    start = state.position[piece_id]
    assert start is not None
    occupied = state.occupied()
    destinations: set[Pos] = set()
    for dq, dr, ds in HEX_DIRS:
        q, r, s = start
        hopped_over = 0
        while True:
            q, r, s = q + dq, r + dr, s + ds
            pos = (q, r, s)
            if pos not in occupied:
                if hopped_over > 0:
                    destinations.add(pos)
                break
            hopped_over += 1
    return destinations


def _beetle_destinations(state: GameState, piece_id: int) -> set[Pos]:
    start = state.position[piece_id]
    assert start is not None
    piece_level = state.stack_index_of(piece_id)
    destinations: set[Pos] = set()
    for end in neighbors(start):
        end_stack_len = state.stack_height_at(end)
        if piece_level != end_stack_len:
            # Climbing on/off a stack always changes level -- no gate check.
            destinations.add(end)
            continue
        n1, n2 = shared_neighbors(start, end)
        h1 = state.stack_height_at(n1) - 1  # top-of-stack index, -1 if empty
        h2 = state.stack_height_at(n2) - 1
        if h1 >= piece_level and h2 >= piece_level:
            continue  # wedged between two taller (or equal) neighbors
        destinations.add(end)
    return destinations


def _ladybug_destinations(state: GameState, piece_id: int) -> set[Pos]:
    start = state.position[piece_id]
    assert start is not None
    occupied = set(state.occupied())
    occupied.discard(start)
    destinations: set[Pos] = set()
    for a in neighbors(start):
        if a not in occupied:
            continue
        for b in neighbors(a):
            if b == start or b not in occupied:
                continue
            for c in neighbors(b):
                if c in (start, a) or c in occupied:
                    continue
                destinations.add(c)
    return destinations


def _mosquito_destinations(state: GameState, piece_id: int) -> set[Pos]:
    if state.stack_index_of(piece_id) > 0:
        # Having climbed up (by copying a beetle) it now only moves as one.
        return _beetle_destinations(state, piece_id)
    start = state.position[piece_id]
    assert start is not None
    adjacent_types = {
        state.pieces[top].piece_type
        for nb in neighbors(start)
        if (top := state.top_piece_at(nb)) is not None
    }
    adjacent_types.discard(PieceType.MOSQUITO)
    destinations: set[Pos] = set()
    for piece_type in adjacent_types:
        destinations |= _destinations_for(state, piece_id, piece_type)
    return destinations


# --- pillbug throw ---------------------------------------------------------


def _pillbug_throw_moves(state: GameState) -> list[Move]:
    owner = state.current_player
    moves: list[Move] = []
    for pillbug_id in state.pieces_on_board(owner):
        if state.pieces[pillbug_id].piece_type != PieceType.PILLBUG:
            continue
        pos = state.position[pillbug_id]
        assert pos is not None
        if state.top_piece_at(pos) != pillbug_id:
            continue
        occupied = state.occupied()
        empty_dests = [nb for nb in neighbors(pos) if nb not in occupied]
        if not empty_dests:
            continue
        for target_pos in neighbors(pos):
            target_id = state.top_piece_at(target_pos)
            if target_id is None or target_id == pillbug_id:
                continue
            if state.stack_height_at(target_pos) > 1:
                continue  # only a lone piece on the ground can be thrown
            if state.is_frozen(target_id):
                continue
            for dest in empty_dests:
                if _throw_keeps_hive_connected(state, target_id, dest):
                    moves.append(
                        Move(MoveKind.THROW, pillbug_id, dest, thrown_piece_id=target_id)
                    )
    return moves


def _throw_keeps_hive_connected(state: GameState, target_id: int, dest: Pos) -> bool:
    target_pos = state.position[target_id]
    assert target_pos is not None
    occ = set(state.occupied())
    occ.discard(target_pos)  # verified height 1 by the caller
    if not positions_connected(frozenset(occ)):
        return False
    occ.add(dest)
    return positions_connected(frozenset(occ))
