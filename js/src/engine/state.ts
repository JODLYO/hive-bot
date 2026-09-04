// Plain-data board state for the in-browser Hive engine -- a direct port
// of the Python package's engine/state.py. See that file's docstring for
// why the design is dict/array based rather than richer objects: `apply.ts`
// mutates a GameState in place (make/unmake) rather than cloning it.
//
// Positions are `[q, r, s]` tuples, but JS can't use array/tuple values as
// Map keys by value (only by reference) -- so everywhere Python uses a
// `Pos` as a dict/set key, this port uses `posKey(pos)` (a `"q,r,s"`
// string, matching the same convention the Django app's own JSON
// serialization and the React frontend's `types.ts` already use) instead.

import {
  BASE_PIECE_TYPES,
  ALL_PIECE_TYPES,
  HEX_DIRS,
  PIECE_COUNTS,
  PieceType,
} from "./constants.js";

export type Pos = readonly [number, number, number];
export type Owner = 0 | 1;

export const DRAW = -1; // sentinel for GameState.winner on a draw

export function posKey(pos: Pos): string {
  return `${pos[0]},${pos[1]},${pos[2]}`;
}

export function parsePosKey(key: string): Pos {
  const parts = key.split(",").map(Number);
  return [parts[0], parts[1], parts[2]];
}

export function neighbors(pos: Pos): Pos[] {
  const [q, r, s] = pos;
  return HEX_DIRS.map(([dq, dr, ds]) => [q + dq, r + dr, s + ds] as Pos);
}

/** The two hexes adjacent to both `a` and `b` (always exactly 2 for
 * adjacent `a`/`b` on a hex grid) -- the pair a slide must not be wedged
 * between. */
export function sharedNeighbors(a: Pos, b: Pos): [Pos, Pos] {
  const na = neighbors(a);
  const bKeys = new Set(neighbors(b).map(posKey));
  const shared = na.filter((p) => bKeys.has(posKey(p)));
  return [shared[0], shared[1]];
}

/** Whether every position in `occupied` is reachable from every other via
 * adjacency -- the "hive must stay one piece" rule. Pure graph
 * connectivity over positions; piece identity doesn't matter. */
export function positionsConnected(occupied: ReadonlySet<string>): boolean {
  if (occupied.size === 0) return true;
  const start = occupied.values().next().value as string;
  const seen = new Set<string>([start]);
  const stack: Pos[] = [parsePosKey(start)];
  while (stack.length > 0) {
    const cur = stack.pop() as Pos;
    for (const nb of neighbors(cur)) {
      const key = posKey(nb);
      if (occupied.has(key) && !seen.has(key)) {
        seen.add(key);
        stack.push(nb);
      }
    }
  }
  return seen.size === occupied.size;
}

export interface Piece {
  readonly id: number;
  readonly pieceType: PieceType;
  readonly owner: Owner;
}

export function buildPieces(enabledTypes: ReadonlySet<PieceType>): Piece[] {
  const pieces: Piece[] = [];
  let pid = 0;
  for (const owner of [0, 1] as const) {
    for (const pieceType of ALL_PIECE_TYPES) {
      if (!enabledTypes.has(pieceType)) continue;
      for (let i = 0; i < PIECE_COUNTS[pieceType]; i++) {
        pieces.push({ id: pid, pieceType, owner });
        pid++;
      }
    }
  }
  return pieces;
}

export class GameState {
  pieces: Map<number, Piece>;
  board: Map<string, number[]>; // posKey -> stack of piece ids, bottom -> top
  position: Map<number, Pos | null>; // null means still in hand
  hand: [number[], number[]]; // hand[owner] = piece ids not yet placed
  queenPlaced: [boolean, boolean];
  currentPlayer: Owner;
  turnNo: number;
  ply: number;
  lastMovedPieceId: number | null = null;
  lastMovedPly: number | null = null;
  gameOver = false;
  winner: number | null = null; // 0, 1, DRAW, or null while ongoing

  // How many times each distinct position (board + hands + side to move)
  // has occurred, for threefold-repetition draw detection. Not an official
  // Hive rule -- see constants.ts's REPETITION_LIMIT.
  positionCounts: Map<string, number> = new Map();

  constructor(
    pieces: Map<number, Piece>,
    board: Map<string, number[]>,
    position: Map<number, Pos | null>,
    hand: [number[], number[]],
    queenPlaced: [boolean, boolean],
    currentPlayer: Owner,
    turnNo: number,
    ply: number,
  ) {
    this.pieces = pieces;
    this.board = board;
    this.position = position;
    this.hand = hand;
    this.queenPlaced = queenPlaced;
    this.currentPlayer = currentPlayer;
    this.turnNo = turnNo;
    this.ply = ply;
  }

  static newGame(enabledTypes: ReadonlySet<PieceType> = BASE_PIECE_TYPES): GameState {
    const piecesList = buildPieces(enabledTypes);
    const pieces = new Map(piecesList.map((p) => [p.id, p]));
    const hand: [number[], number[]] = [[], []];
    for (const p of piecesList) hand[p.owner].push(p.id);
    const position = new Map<number, Pos | null>(piecesList.map((p) => [p.id, null]));
    return new GameState(pieces, new Map(), position, hand, [false, false], 0, 1, 0);
  }

  opponent(): Owner {
    return this.currentPlayer === 0 ? 1 : 0;
  }

  occupied(): ReadonlySet<string> {
    return new Set(this.board.keys());
  }

  stackHeightAt(pos: Pos): number {
    return this.board.get(posKey(pos))?.length ?? 0;
  }

  topPieceAt(pos: Pos): number | null {
    const stack = this.board.get(posKey(pos));
    return stack && stack.length > 0 ? stack[stack.length - 1] : null;
  }

  /** 0-indexed height of `pieceId` within its stack (0 = ground). */
  stackIndexOf(pieceId: number): number {
    const pos = this.position.get(pieceId);
    if (pos == null) throw new Error(`piece ${pieceId} is not on the board`);
    const stack = this.board.get(posKey(pos));
    if (!stack) throw new Error(`no stack at ${posKey(pos)}`);
    return stack.indexOf(pieceId);
  }

  /** Positions where `owner` currently has the top piece of the stack.
   * Deliberately *not* "every position any piece of theirs has ever
   * occupied" -- a piece buried under an opponent's beetle/mosquito no
   * longer controls that hex for placement-adjacency purposes (the
   * "don't place touching an opponent" rule), only whichever piece is
   * currently on top does. Confirmed against the Python engine's oracle:
   * using every piece's own position here (buried or not) wrongly
   * forbids/allows placements near a mixed-color stack. */
  boardPositions(owner: Owner): Set<string> {
    const positions = new Set<string>();
    for (const [key, stack] of this.board) {
      const topId = stack[stack.length - 1];
      if (this.pieces.get(topId)!.owner === owner) positions.add(key);
    }
    return positions;
  }

  piecesOnBoard(owner: Owner): number[] {
    const result: number[] = [];
    for (const [pid, piece] of this.pieces) {
      if (piece.owner === owner && this.position.get(pid) != null) result.push(pid);
    }
    return result;
  }

  /** Pillbug freeze rule: a piece that moved "last turn" can't be moved or
   * thrown this ply. The `ply - 1` comparison (rather than the arguably
   * more obvious `ply`) is copied verbatim from the Python engine's
   * `is_frozen`, which itself copies the reference Django app's exact
   * comparison for oracle-parity reasons -- see engine/state.py. */
  isFrozen(pieceId: number): boolean {
    return this.lastMovedPieceId === pieceId && this.lastMovedPly === this.ply - 1;
  }
}

/** A snapshot of "the position" for repetition detection: board contents
 * (including full stacks, since a beetle's height matters) plus each
 * hand's contents plus whose turn it is. Doesn't account for the pillbug
 * freeze state (which piece moved last) -- a pragmatic simplification,
 * since this only needs to bound self-play game length, not be
 * tournament-rules-perfect. Python's version returns a hashable tuple;
 * this returns the equivalent as a single string, since that's what's
 * usable as a JS Map key. */
export function positionKey(state: GameState): string {
  const boardEntries = Array.from(state.board.entries())
    .map(([key, stack]) => `${key}|${stack.join(",")}`)
    .sort();
  const hand0 = [...state.hand[0]].sort((a, b) => a - b).join(",");
  const hand1 = [...state.hand[1]].sort((a, b) => a - b).join(",");
  return `${boardEntries.join(";")}::${hand0}::${hand1}::${state.currentPlayer}`;
}
