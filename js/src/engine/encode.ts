// GameState -> tensor encoding, a direct port of the Python package's
// engine/encode.py. Always encodes from the current player's perspective
// ("me" vs "opponent" rather than raw owner 0/1) -- see that file's
// docstring for why, and model/mcts.ts's value backprop for how this
// convention gets used.
//
// `board`/`globalFeatures` are flat Float32Arrays in row-major ("C") order
// -- board is `[channel, row, col]` flattened, matching a PyTorch
// `(C, H, W)` tensor's memory layout exactly, so these map directly onto
// an onnxruntime-web `Tensor` without any reshaping (see model/network.ts).

import {
  BOARD_DIM,
  BOARD_RADIUS,
  MAX_STACK_HEIGHT,
  NUM_PIECE_TYPES,
  PIECE_COUNTS,
  PieceType,
} from "./constants.js";
import type { GameState, Owner, Pos } from "./state.js";
import { parsePosKey } from "./state.js";

// Per-cell (spatial) channels: top-piece one-hot (mine, then opponent's) +
// normalized stack height + "this cell's top piece moved last ply" (the
// pillbug freeze rule's visible signal).
export const NUM_SPATIAL_CHANNELS = 2 * NUM_PIECE_TYPES + 2;
const STACK_HEIGHT_CHANNEL = 2 * NUM_PIECE_TYPES;
const LAST_MOVED_CHANNEL = 2 * NUM_PIECE_TYPES + 1;

// Global (non-spatial) features: hand counts per type (mine, then
// opponent's), each side's queen-placed flag, and a normalized turn number.
export const NUM_GLOBAL_FEATURES = 2 * NUM_PIECE_TYPES + 3;
const TURN_NORMALIZATION = 50.0;

/** Cube coordinate -> [row, col] in the BOARD_DIM x BOARD_DIM grid. */
export function posToCell(pos: Pos): [number, number] {
  const [q, r] = pos;
  return [r + BOARD_RADIUS, q + BOARD_RADIUS];
}

export function cellToPos(row: number, col: number): Pos {
  const q = col - BOARD_RADIUS;
  const r = row - BOARD_RADIUS;
  return [q, r, -q - r];
}

export function posToIndex(pos: Pos): number {
  const [row, col] = posToCell(pos);
  return row * BOARD_DIM + col;
}

export function indexToPos(index: number): Pos {
  const row = Math.floor(index / BOARD_DIM);
  const col = index % BOARD_DIM;
  return cellToPos(row, col);
}

export interface EncodedState {
  board: Float32Array; // NUM_SPATIAL_CHANNELS * BOARD_DIM * BOARD_DIM
  globalFeatures: Float32Array; // NUM_GLOBAL_FEATURES
}

export function encodeState(state: GameState): EncodedState {
  const me = state.currentPlayer;
  const opponent: Owner = me === 0 ? 1 : 0;
  const planeSize = BOARD_DIM * BOARD_DIM;

  const board = new Float32Array(NUM_SPATIAL_CHANNELS * planeSize);
  for (const [key, stack] of state.board) {
    const [row, col] = posToCell(parsePosKey(key));
    const cellOffset = row * BOARD_DIM + col;
    const topId = stack[stack.length - 1];
    const piece = state.pieces.get(topId)!;
    const ownerOffset = piece.owner === me ? 0 : NUM_PIECE_TYPES;
    board[(ownerOffset + piece.pieceType) * planeSize + cellOffset] = 1.0;
    board[STACK_HEIGHT_CHANNEL * planeSize + cellOffset] = stack.length / MAX_STACK_HEIGHT;
    if (topId === state.lastMovedPieceId && state.lastMovedPly === state.ply) {
      board[LAST_MOVED_CHANNEL * planeSize + cellOffset] = 1.0;
    }
  }

  const globalFeatures = new Float32Array(NUM_GLOBAL_FEATURES);
  const meCounts = handCounts(state, me);
  const oppCounts = handCounts(state, opponent);
  globalFeatures.set(meCounts, 0);
  globalFeatures.set(oppCounts, NUM_PIECE_TYPES);
  globalFeatures[2 * NUM_PIECE_TYPES] = state.queenPlaced[me] ? 1 : 0;
  globalFeatures[2 * NUM_PIECE_TYPES + 1] = state.queenPlaced[opponent] ? 1 : 0;
  globalFeatures[2 * NUM_PIECE_TYPES + 2] =
    Math.min(state.turnNo, TURN_NORMALIZATION) / TURN_NORMALIZATION;

  return { board, globalFeatures };
}

function handCounts(state: GameState, owner: Owner): Float32Array {
  const counts = new Float32Array(NUM_PIECE_TYPES);
  for (const pid of state.hand[owner]) {
    counts[state.pieces.get(pid)!.pieceType] += 1;
  }
  for (let i = 0; i < NUM_PIECE_TYPES; i++) counts[i] /= PIECE_COUNTS[i as PieceType];
  return counts;
}
