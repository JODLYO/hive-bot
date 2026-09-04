// Make/unmake move application for the in-browser Hive engine -- a direct
// port of the Python package's engine/apply.py. `applyMove` mutates a
// GameState in place and returns an UndoInfo; `undoMove` reverses it
// exactly. See that file's docstring for why (MCTS needs to apply/revert
// many moves per second) and for the turn-structure bookkeeping this also
// owns (forced queen placement, ply/last-moved bookkeeping, auto-pass,
// the double-surrounded-queen draw, and the engine-only move-cap/
// repetition draw rules).

import { MAX_PLIES_BEFORE_DRAW, PieceType, REPETITION_LIMIT } from "./constants.js";
import { generateLegalMoves, type Move, MoveKind } from "./moves.js";
import { DRAW, type GameState, neighbors, type Owner, type Pos, posKey, positionKey } from "./state.js";

export interface UndoInfo {
  move: Move;
  movedPieceId: number; // the piece whose position/hand status changed
  prevPosition: Pos | null; // null means it was in hand before this move
  prevHandIndex: number | null; // index to reinsert at, only set if prevPosition is null
  prevCurrentPlayer: Owner;
  prevTurnNo: number;
  prevPly: number;
  prevLastMovedPieceId: number | null;
  prevLastMovedPly: number | null;
  prevGameOver: boolean;
  prevWinner: number | null;
  prevQueenPlacedMover: boolean;
  positionKey: string | null; // set once computed, for undo to decrement
}

export function applyMove(state: GameState, move: Move): UndoInfo {
  if (state.gameOver) throw new Error("cannot apply a move to a finished game");
  const mover = state.currentPlayer;
  const actedPieceId = move.kind === MoveKind.THROW ? move.thrownPieceId! : move.pieceId;
  const actedOwner = state.pieces.get(actedPieceId)!.owner;

  const prevPosition = state.position.get(actedPieceId) ?? null;
  const undo: UndoInfo = {
    move,
    movedPieceId: actedPieceId,
    prevPosition,
    prevHandIndex: null,
    prevCurrentPlayer: state.currentPlayer,
    prevTurnNo: state.turnNo,
    prevPly: state.ply,
    prevLastMovedPieceId: state.lastMovedPieceId,
    prevLastMovedPly: state.lastMovedPly,
    prevGameOver: state.gameOver,
    prevWinner: state.winner,
    prevQueenPlacedMover: state.queenPlaced[actedOwner],
    positionKey: null,
  };

  if (prevPosition === null) {
    const hand = state.hand[actedOwner];
    undo.prevHandIndex = hand.indexOf(actedPieceId);
    hand.splice(undo.prevHandIndex, 1);
    if (state.pieces.get(actedPieceId)!.pieceType === PieceType.QUEEN) {
      state.queenPlaced[actedOwner] = true;
    }
  } else {
    const stack = state.board.get(posKey(prevPosition))!;
    stack.pop();
    if (stack.length === 0) state.board.delete(posKey(prevPosition));
  }

  const dest = move.to;
  const destKey = posKey(dest);
  const destStack = state.board.get(destKey);
  if (destStack) destStack.push(actedPieceId);
  else state.board.set(destKey, [actedPieceId]);
  state.position.set(actedPieceId, dest);

  state.ply += 1;
  state.lastMovedPieceId = actedPieceId;
  state.lastMovedPly = state.ply;

  if (mover === 1) state.turnNo += 1;
  state.currentPlayer = mover === 0 ? 1 : 0;

  const key = positionKey(state);
  const count = (state.positionCounts.get(key) ?? 0) + 1;
  state.positionCounts.set(key, count);
  undo.positionKey = key;

  checkGameOver(state);
  if (!state.gameOver && (count >= REPETITION_LIMIT || state.ply >= MAX_PLIES_BEFORE_DRAW)) {
    state.gameOver = true;
    state.winner = DRAW;
  }
  resolveAutoPass(state);

  return undo;
}

export function undoMove(state: GameState, undo: UndoInfo): void {
  const pieceId = undo.movedPieceId;
  const owner = state.pieces.get(pieceId)!.owner;

  if (undo.positionKey === null) throw new Error("UndoInfo.positionKey was never set");
  const count = state.positionCounts.get(undo.positionKey)!;
  if (count <= 1) state.positionCounts.delete(undo.positionKey);
  else state.positionCounts.set(undo.positionKey, count - 1);

  const dest = state.position.get(pieceId)!;
  const destKey = posKey(dest);
  const stack = state.board.get(destKey)!;
  stack.pop();
  if (stack.length === 0) state.board.delete(destKey);

  if (undo.prevPosition === null) {
    if (undo.prevHandIndex === null) throw new Error("UndoInfo.prevHandIndex was never set");
    state.hand[owner].splice(undo.prevHandIndex, 0, pieceId);
    state.position.set(pieceId, null);
  } else {
    const prev = undo.prevPosition;
    const prevKey = posKey(prev);
    const prevStack = state.board.get(prevKey);
    if (prevStack) prevStack.push(pieceId);
    else state.board.set(prevKey, [pieceId]);
    state.position.set(pieceId, prev);
  }

  state.queenPlaced[owner] = undo.prevQueenPlacedMover;
  state.currentPlayer = undo.prevCurrentPlayer;
  state.turnNo = undo.prevTurnNo;
  state.ply = undo.prevPly;
  state.lastMovedPieceId = undo.prevLastMovedPieceId;
  state.lastMovedPly = undo.prevLastMovedPly;
  state.gameOver = undo.prevGameOver;
  state.winner = undo.prevWinner;
}

function checkGameOver(state: GameState): void {
  const q0 = queenId(state, 0);
  const q1 = queenId(state, 1);
  const s0 = q0 !== null && isQueenSurrounded(state, q0);
  const s1 = q1 !== null && isQueenSurrounded(state, q1);
  if (!s0 && !s1) return;
  state.gameOver = true;
  if (s0 && s1) state.winner = DRAW;
  else state.winner = s0 ? 1 : 0;
}

function queenId(state: GameState, owner: Owner): number | null {
  for (const [pid, piece] of state.pieces) {
    if (piece.owner === owner && piece.pieceType === PieceType.QUEEN && state.position.get(pid) != null) {
      return pid;
    }
  }
  return null;
}

function isQueenSurrounded(state: GameState, pieceId: number): boolean {
  const pos = state.position.get(pieceId)!;
  const occupied = state.occupied();
  return neighbors(pos).every((nb) => occupied.has(posKey(nb)));
}

/** A player with no legal placement or move at all forfeits their turn.
 * Two attempts cover both players once; if neither can act, it's a mutual
 * deadlock (draw). */
function resolveAutoPass(state: GameState): void {
  for (let i = 0; i < 2; i++) {
    if (state.gameOver) return;
    if (generateLegalMoves(state).length > 0) return;
    if (state.currentPlayer === 1) state.turnNo += 1;
    state.currentPlayer = state.currentPlayer === 0 ? 1 : 0;
  }
  state.gameOver = true;
  state.winner = DRAW;
}
