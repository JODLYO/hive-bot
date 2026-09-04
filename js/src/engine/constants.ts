// Fixed sizing and rules constants for the in-browser Hive engine -- a
// direct port of the Python package's engine/constants.py. Keep these two
// files in sync; the values themselves (not just the shape) must match
// exactly, since the ONNX-exported network's input tensor shape depends on
// BOARD_DIM/NUM_SPATIAL_CHANNELS matching what it was trained with.
//
// Board sizing is derived from the *full* expansion piece set (28 pieces
// total on the board) so that base-game-only training never needs a board
// resize later. All coordinates use the same cube hex system as the
// reference Django app (`q + r + s === 0`), with the first-placed piece
// fixed at (0, 0, 0).

export const HEX_DIRS: readonly (readonly [number, number, number])[] = [
  [1, -1, 0],
  [1, 0, -1],
  [0, 1, -1],
  [-1, 1, 0],
  [-1, 0, 1],
  [0, -1, 1],
];

export enum PieceType {
  QUEEN = 0,
  ANT = 1,
  SPIDER = 2,
  BEETLE = 3,
  GRASSHOPPER = 4,
  MOSQUITO = 5,
  LADYBUG = 6,
  PILLBUG = 7,
}

export const NUM_PIECE_TYPES = 8;

export const ALL_PIECE_TYPES: readonly PieceType[] = [
  PieceType.QUEEN,
  PieceType.ANT,
  PieceType.SPIDER,
  PieceType.BEETLE,
  PieceType.GRASSHOPPER,
  PieceType.MOSQUITO,
  PieceType.LADYBUG,
  PieceType.PILLBUG,
];

// Per-side piece counts, full expansion set.
export const PIECE_COUNTS: Readonly<Record<PieceType, number>> = {
  [PieceType.QUEEN]: 1,
  [PieceType.ANT]: 3,
  [PieceType.SPIDER]: 2,
  [PieceType.BEETLE]: 2,
  [PieceType.GRASSHOPPER]: 3,
  [PieceType.MOSQUITO]: 1,
  [PieceType.LADYBUG]: 1,
  [PieceType.PILLBUG]: 1,
};

export const BASE_PIECE_TYPES: ReadonlySet<PieceType> = new Set([
  PieceType.QUEEN,
  PieceType.ANT,
  PieceType.SPIDER,
  PieceType.BEETLE,
  PieceType.GRASSHOPPER,
]);

export const EXPANSION_PIECE_TYPES: ReadonlySet<PieceType> = new Set([
  PieceType.MOSQUITO,
  PieceType.LADYBUG,
  PieceType.PILLBUG,
]);

export const PIECES_PER_SIDE = ALL_PIECE_TYPES.reduce(
  (sum, t) => sum + PIECE_COUNTS[t],
  0,
); // 14
export const TOTAL_PIECES = PIECES_PER_SIDE * 2; // 28

// Worst-case straight chain of every piece on the board spans
// TOTAL_PIECES - 1 hexes from the origin in one direction. First-placed
// piece is fixed at (0, 0, 0), never recentered.
export const BOARD_RADIUS = TOTAL_PIECES - 1; // 27
export const BOARD_DIM = 2 * BOARD_RADIUS + 1; // 55, spatial tensor side length

// Only beetles and a mosquito copying a beetle can climb, so a stack is at
// most (2 beetles + 1 mosquito) per side, plus the base piece underneath.
export const MAX_STACK_HEIGHT =
  2 * (PIECE_COUNTS[PieceType.BEETLE] + PIECE_COUNTS[PieceType.MOSQUITO]) + 1; // 7

export const TURN_NUMBER_QUEEN_MUST_BE_PLACED = 4;

// Official Hive has neither rule, but self-play needs both to bound
// worst-case game length. See constants.py for the full rationale.
export const MAX_PLIES_BEFORE_DRAW = 1000;
export const REPETITION_LIMIT = 3;
