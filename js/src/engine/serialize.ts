// GameState <-> a plain, structured-clone-safe shape.
//
// A `GameState` is a class instance -- its `Map` fields survive the
// structured clone algorithm `postMessage` uses (e.g. across a Web
// Worker boundary), but the class prototype does not, so the receiving
// side gets a plain object missing methods like `occupied()`/`topPieceAt()`
// that the engine relies on everywhere. `serializeGameState` produces a
// shape safe to pass across such a boundary (or through `JSON.stringify`,
// or into `localStorage`); `deserializeGameState` reconstructs a real
// `GameState` instance from it.

import type { Owner, Piece, Pos } from "./state.js";
import { GameState } from "./state.js";

export interface SerializedGameState {
  pieces: [number, Piece][];
  board: [string, number[]][];
  position: [number, Pos | null][];
  hand: [number[], number[]];
  queenPlaced: [boolean, boolean];
  currentPlayer: Owner;
  turnNo: number;
  ply: number;
  lastMovedPieceId: number | null;
  lastMovedPly: number | null;
  gameOver: boolean;
  winner: number | null;
  positionCounts: [string, number][];
}

export function serializeGameState(state: GameState): SerializedGameState {
  return {
    pieces: Array.from(state.pieces.entries()),
    board: Array.from(state.board.entries()),
    position: Array.from(state.position.entries()),
    hand: [
      [...state.hand[0]],
      [...state.hand[1]],
    ],
    queenPlaced: [...state.queenPlaced],
    currentPlayer: state.currentPlayer,
    turnNo: state.turnNo,
    ply: state.ply,
    lastMovedPieceId: state.lastMovedPieceId,
    lastMovedPly: state.lastMovedPly,
    gameOver: state.gameOver,
    winner: state.winner,
    positionCounts: Array.from(state.positionCounts.entries()),
  };
}

export function deserializeGameState(s: SerializedGameState): GameState {
  const state = new GameState(
    new Map(s.pieces),
    new Map(s.board),
    new Map(s.position),
    [[...s.hand[0]], [...s.hand[1]]],
    [...s.queenPlaced] as [boolean, boolean],
    s.currentPlayer,
    s.turnNo,
    s.ply,
  );
  state.lastMovedPieceId = s.lastMovedPieceId;
  state.lastMovedPly = s.lastMovedPly;
  state.gameOver = s.gameOver;
  state.winner = s.winner;
  state.positionCounts = new Map(s.positionCounts);
  return state;
}
