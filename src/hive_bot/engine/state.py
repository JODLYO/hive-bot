"""Plain-data board state for the fast Hive engine.

Deliberately dict/tuple/int based (no pydantic) -- see constants.py and the
plan doc for why. `apply.py` mutates a `GameState` in place (make/unmake)
rather than copying it, so this module also owns the small set of read-only
helpers (`neighbors`, `occupied`, `top_piece_at`, ...) that both move
generation and move application share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import BASE_PIECE_TYPES, HEX_DIRS, PIECE_COUNTS, PieceType

Pos = tuple[int, int, int]
PositionKey = tuple[
    tuple[tuple[Pos, tuple[int, ...]], ...], tuple[int, ...], tuple[int, ...], int
]

DRAW = -1  # sentinel for GameState.winner on a draw


def neighbors(pos: Pos) -> tuple[Pos, ...]:
    q, r, s = pos
    return tuple((q + dq, r + dr, s + ds) for dq, dr, ds in HEX_DIRS)


def shared_neighbors(a: Pos, b: Pos) -> tuple[Pos, Pos]:
    """The two hexes adjacent to both `a` and `b` (always exactly 2 for
    adjacent `a`/`b` on a hex grid) -- the pair a slide must not be
    wedged between."""
    na = set(neighbors(a))
    nb = set(neighbors(b))
    shared = na & nb
    it = iter(shared)
    return next(it), next(it)


def position_key(state: GameState) -> PositionKey:
    """A hashable snapshot of "the position" for repetition detection:
    board contents (including full stacks, since a beetle's height matters)
    plus each hand's contents plus whose turn it is. Doesn't account for the
    pillbug freeze state (which piece moved last) -- a pragmatic
    simplification, since this only needs to bound self-play game length,
    not be tournament-rules-perfect."""
    board_items = tuple(sorted((pos, tuple(stack)) for pos, stack in state.board.items()))
    hands = (tuple(sorted(state.hand[0])), tuple(sorted(state.hand[1])))
    return (board_items, *hands, state.current_player)


def positions_connected(occupied: frozenset[Pos]) -> bool:
    """Whether every position in `occupied` is reachable from every other
    via adjacency -- the "hive must stay one piece" rule. Pure graph
    connectivity over positions; piece identity doesn't matter."""
    if not occupied:
        return True
    start = next(iter(occupied))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in neighbors(cur):
            if nb in occupied and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(occupied)


@dataclass(frozen=True, slots=True)
class Piece:
    id: int
    piece_type: PieceType
    owner: int  # 0 or 1


def build_pieces(enabled_types: frozenset[PieceType]) -> list[Piece]:
    pieces = []
    pid = 0
    for owner in (0, 1):
        for piece_type in PieceType:
            if piece_type not in enabled_types:
                continue
            for _ in range(PIECE_COUNTS[piece_type]):
                pieces.append(Piece(id=pid, piece_type=piece_type, owner=owner))
                pid += 1
    return pieces


@dataclass(slots=True)
class GameState:
    pieces: dict[int, Piece]
    board: dict[Pos, list[int]]  # occupied cells only, bottom -> top piece ids
    position: dict[int, Pos | None]  # None means still in hand
    hand: list[list[int]]  # hand[owner] = piece ids not yet placed
    queen_placed: list[bool]  # queen_placed[owner]
    current_player: int
    turn_no: int
    ply: int
    last_moved_piece_id: int | None = None
    last_moved_ply: int | None = None
    game_over: bool = False
    winner: int | None = None  # 0, 1, DRAW, or None while ongoing

    # How many times each distinct position (board + hands + side to move)
    # has occurred, for threefold-repetition draw detection. Not an official
    # Hive rule -- see constants.py's REPETITION_LIMIT.
    position_counts: dict[PositionKey, int] = field(default_factory=dict)

    @classmethod
    def new_game(cls, enabled_types: frozenset[PieceType] = BASE_PIECE_TYPES) -> GameState:
        pieces_list = build_pieces(enabled_types)
        pieces = {p.id: p for p in pieces_list}
        hand: list[list[int]] = [[], []]
        for p in pieces_list:
            hand[p.owner].append(p.id)
        return cls(
            pieces=pieces,
            board={},
            position=dict.fromkeys(pieces),
            hand=hand,
            queen_placed=[False, False],
            current_player=0,
            turn_no=1,
            ply=0,
        )

    def opponent(self) -> int:
        return 1 - self.current_player

    def occupied(self) -> frozenset[Pos]:
        return frozenset(self.board.keys())

    def stack_height_at(self, pos: Pos) -> int:
        return len(self.board.get(pos, ()))

    def top_piece_at(self, pos: Pos) -> int | None:
        stack = self.board.get(pos)
        return stack[-1] if stack else None

    def stack_index_of(self, piece_id: int) -> int:
        """0-indexed height of `piece_id` within its stack (0 = ground)."""
        pos = self.position[piece_id]
        assert pos is not None
        return self.board[pos].index(piece_id)

    def board_positions(self, owner: int) -> set[Pos]:
        positions: set[Pos] = set()
        for pid in self.pieces_on_board(owner):
            pos = self.position[pid]
            assert pos is not None
            positions.add(pos)
        return positions

    def pieces_on_board(self, owner: int) -> list[int]:
        return [
            pid
            for pid, p in self.pieces.items()
            if p.owner == owner and self.position[pid] is not None
        ]

    def is_frozen(self, piece_id: int) -> bool:
        """Pillbug freeze rule: a piece that moved "last turn" can't be
        moved or thrown this ply. The `ply - 1` comparison (rather than the
        arguably more obvious `ply`) is copied verbatim from the reference
        engine's `pillbug_throw_valid` -- see tests/reference/helpers.py and
        its test_pillbug_throw_freeze_rule -- to stay byte-for-byte
        compatible with the oracle used in test_engine_vs_reference.py."""
        return self.last_moved_piece_id == piece_id and self.last_moved_ply == self.ply - 1
