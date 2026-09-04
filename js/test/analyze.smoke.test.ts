// End-to-end smoke test: real ONNX Runtime Web inference against a real
// (if tiny/untrained) exported model, running the full self-play stack
// (encode -> network -> MCTS -> analysis) exactly as a browser would.
// Model is js/test/fixtures/tiny_model.onnx, produced by
// scripts/export_test_model.py -- regenerate with
// `uv run python scripts/export_test_model.py` if it ever needs updating.
//
// This is the piece none of the parity tests exercise: they check the
// engine/encode/actions math is correct, but not that onnxruntime-web
// actually loads a model and runs inference correctly end to end.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { HiveBot } from "../src/analysis/bot.js";
import { generateLegalMoves } from "../src/engine/moves.js";
import { GameState } from "../src/engine/state.js";
import { BASE_PIECE_TYPES } from "../src/engine/constants.js";
import { applyMove } from "../src/engine/apply.js";
import { MCTS, selectMove } from "../src/model/mcts.js";
import { HiveNetSession } from "../src/model/network.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = join(__dirname, "fixtures", "tiny_model.onnx");

function moveEqual(a: { kind: number; pieceId: number; to: readonly number[] }, b: typeof a) {
  return a.kind === b.kind && a.pieceId === b.pieceId && a.to.join(",") === b.to.join(",");
}

describe("end-to-end: HiveBot.analyze against a real ONNX model", () => {
  it("returns a legal best move and a valid win probability for the opening position", async () => {
    const modelBytes = new Uint8Array(readFileSync(MODEL_PATH));
    const bot = await HiveBot.fromModel(modelBytes, 8);

    const state = GameState.newGame(BASE_PIECE_TYPES);
    const legal = generateLegalMoves(state);

    const analysis = await bot.analyze(state);

    expect(legal.some((m) => moveEqual(m, analysis.bestMove))).toBe(true);
    expect(analysis.winProbability).toBeGreaterThanOrEqual(0);
    expect(analysis.winProbability).toBeLessThanOrEqual(1);
    expect(analysis.moveEvaluations.length).toBe(legal.length);
    const fractionSum = analysis.moveEvaluations.reduce((sum, e) => sum + e.visitFraction, 0);
    expect(fractionSum).toBeCloseTo(1, 5);
  });

  it("plays several plies without crashing, each move coming from generateLegalMoves", async () => {
    const modelBytes = new Uint8Array(readFileSync(MODEL_PATH));
    const session = await HiveNetSession.load(modelBytes);
    const mcts = new MCTS(session, { random: () => 0.5 });

    const state = GameState.newGame(BASE_PIECE_TYPES);
    for (let ply = 0; ply < 6; ply++) {
      if (state.gameOver) break;
      const legal = generateLegalMoves(state);
      const root = await mcts.run(state, 6, true);
      const move = selectMove(root, 1.0, () => 0.5);
      expect(legal.some((m) => moveEqual(m, move))).toBe(true);
      applyMove(state, move);
    }
  });
});
