"""Fixed sizing and rules constants for the fast Hive engine.

Board sizing is derived from the *full* expansion piece set (28 pieces total
on the board) so that base-game-only training never needs a board resize
later -- see the plan doc for the derivation. All coordinates use the same
cube hex system as the reference app (`q + r + s == 0`), with the
first-placed piece fixed at (0, 0, 0).
"""

from enum import IntEnum

HEX_DIRS: tuple[tuple[int, int, int], ...] = (
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),
)


class PieceType(IntEnum):
    QUEEN = 0
    ANT = 1
    SPIDER = 2
    BEETLE = 3
    GRASSHOPPER = 4
    MOSQUITO = 5
    LADYBUG = 6
    PILLBUG = 7


NUM_PIECE_TYPES = 8

# Per-side piece counts, full expansion set.
PIECE_COUNTS: dict[PieceType, int] = {
    PieceType.QUEEN: 1,
    PieceType.ANT: 3,
    PieceType.SPIDER: 2,
    PieceType.BEETLE: 2,
    PieceType.GRASSHOPPER: 3,
    PieceType.MOSQUITO: 1,
    PieceType.LADYBUG: 1,
    PieceType.PILLBUG: 1,
}

BASE_PIECE_TYPES: frozenset[PieceType] = frozenset(
    {
        PieceType.QUEEN,
        PieceType.ANT,
        PieceType.SPIDER,
        PieceType.BEETLE,
        PieceType.GRASSHOPPER,
    }
)
EXPANSION_PIECE_TYPES: frozenset[PieceType] = frozenset(
    {PieceType.MOSQUITO, PieceType.LADYBUG, PieceType.PILLBUG}
)

PIECES_PER_SIDE = sum(PIECE_COUNTS.values())  # 14
TOTAL_PIECES = PIECES_PER_SIDE * 2  # 28

# Worst-case straight chain of every piece on the board spans
# TOTAL_PIECES - 1 hexes from the origin in one direction. First-placed
# piece is fixed at (0, 0, 0), never recentered.
BOARD_RADIUS = TOTAL_PIECES - 1  # 27
BOARD_DIM = 2 * BOARD_RADIUS + 1  # 55, spatial tensor side length

# Only beetles and a mosquito copying a beetle can climb, so a stack is at
# most (2 beetles + 1 mosquito) per side, plus the base piece underneath.
MAX_STACK_HEIGHT = (
    2 * (PIECE_COUNTS[PieceType.BEETLE] + PIECE_COUNTS[PieceType.MOSQUITO]) + 1
)  # 7

TURN_NUMBER_QUEEN_MUST_BE_PLACED = 4

# Official Hive has neither rule, but self-play needs both to bound worst-case
# game length -- an untrained/near-random policy can otherwise shuffle pieces
# back and forth indefinitely. Deliberately generous thresholds: 40-ply
# random games used in the engine/oracle parity tests should essentially
# never hit either one (see tests/test_engine_vs_reference.py), so this
# stays a pure self-play safeguard rather than something that could ever
# make a normal game's outcome diverge from the reference oracle's rules.
MAX_PLIES_BEFORE_DRAW = 1000
REPETITION_LIMIT = 3
