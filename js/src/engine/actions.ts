// Move <-> policy-action-index mapping for the network's policy head -- a
// direct port of the Python package's engine/actions.py. See that file's
// docstring for the dual-embedding bilinear design this feeds into
// (model/network.ts): there's no dense action tensor built here, just a
// small `(kind, fromIndex, toIndex)` key per move.
//
// PLACE actions are keyed by piece *type* (a fixed hand slot, 0..7), not a
// specific piece instance -- see moves.ts for why in-hand pieces of the
// same type are interchangeable. MOVE and THROW are keyed by the acting
// piece's current board position, which is always unique (at most one
// top-of-stack piece per cell), so within one position's legal-move list
// no two different moves can ever produce the same action key.

import { posToIndex } from "./encode.js";
import type { Move } from "./moves.js";
import { MoveKind } from "./moves.js";
import type { GameState } from "./state.js";

// [kind, fromIndex, toIndex]. PLACE's fromIndex is a hand slot (a
// PieceType value, 0..7); MOVE/THROW's is a board cell index.
export type ActionKey = readonly [MoveKind, number, number];

export function moveToActionKey(state: GameState, move: Move): ActionKey {
  const toIndex = posToIndex(move.to);
  let fromIndex: number;
  if (move.kind === MoveKind.PLACE) {
    fromIndex = state.pieces.get(move.pieceId)!.pieceType;
  } else if (move.kind === MoveKind.MOVE) {
    const start = state.position.get(move.pieceId)!;
    fromIndex = posToIndex(start);
  } else {
    const start = state.position.get(move.thrownPieceId!)!;
    fromIndex = posToIndex(start);
  }
  return [move.kind, fromIndex, toIndex];
}

/** Action keys parallel to `moves` (same order, same length) -- the
 * intended usage is zipping `moves[i]` with `keys[i]`, e.g. to gather
 * network scores for exactly these actions and sample an index into
 * `moves` directly, rather than decoding a key back into a move. */
export function legalActionKeys(state: GameState, moves: readonly Move[]): ActionKey[] {
  return moves.map((m) => moveToActionKey(state, m));
}

/** String form of an ActionKey, for use as a Map/object key -- JS can't
 * key a Map by array/tuple value equality the way Python keys a dict by
 * tuple (see engine/state.ts's posKey for the same pattern). */
export function actionKeyToString(key: ActionKey): string {
  return `${key[0]},${key[1]},${key[2]}`;
}
