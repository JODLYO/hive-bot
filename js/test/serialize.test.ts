import { describe, expect, it } from "vitest";
import { applyMove } from "../src/engine/apply.js";
import { BASE_PIECE_TYPES } from "../src/engine/constants.js";
import { deserializeGameState, serializeGameState } from "../src/engine/serialize.js";
import { generateLegalMoves } from "../src/engine/moves.js";
import { GameState } from "../src/engine/state.js";

describe("GameState serialize/deserialize round-trip", () => {
  it("preserves state through several moves, including via structuredClone", () => {
    const state = GameState.newGame(BASE_PIECE_TYPES);
    for (let i = 0; i < 8; i++) {
      const legal = generateLegalMoves(state);
      applyMove(state, legal[0]);
    }

    const serialized = serializeGameState(state);
    // structuredClone is what postMessage uses under the hood -- prove this
    // survives the actual mechanism, not just plain object copying.
    const cloned = structuredClone(serialized);
    const restored = deserializeGameState(cloned);

    expect(restored.currentPlayer).toBe(state.currentPlayer);
    expect(restored.turnNo).toBe(state.turnNo);
    expect(restored.ply).toBe(state.ply);
    expect(Array.from(restored.board.entries())).toEqual(Array.from(state.board.entries()));
    expect(Array.from(restored.hand[0])).toEqual(Array.from(state.hand[0]));
    expect(Array.from(restored.hand[1])).toEqual(Array.from(state.hand[1]));

    // And the restored instance must be a real GameState -- its methods
    // (not just its data) need to survive, since the engine calls them.
    const restoredMoves = generateLegalMoves(restored);
    const originalMoves = generateLegalMoves(state);
    expect(restoredMoves.length).toBe(originalMoves.length);
  });
});
