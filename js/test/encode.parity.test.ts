// Numeric parity between engine/encode.ts + engine/actions.ts and the
// Python package's encode_state()/legal_action_keys(), on real sampled
// positions (see scripts/generate_js_fixtures.py's generate_encode_fixtures).

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { legalActionKeys } from "../src/engine/actions.js";
import { encodeState } from "../src/engine/encode.js";
import { generateLegalMoves } from "../src/engine/moves.js";
import type { FixtureState } from "./fixtureUtils.js";
import { deserializeState } from "./fixtureUtils.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, "fixtures");

interface EncodeSample {
  state: FixtureState;
  board: number[];
  globalFeatures: number[];
  actionKeys: [number, number, number][];
}

const samples = JSON.parse(
  readFileSync(join(FIXTURES_DIR, "encode_fixtures.json"), "utf-8"),
) as EncodeSample[];

if (samples.length === 0) {
  throw new Error("no samples in encode_fixtures.json -- run `make js-fixtures` first");
}

describe("encode/actions parity: matches the Python engine", () => {
  samples.forEach((sample, i) => {
    it(`sample ${i}`, () => {
      const state = deserializeState(sample.state);

      const encoded = encodeState(state);
      expect(encoded.board.length, `sample ${i} board length`).toBe(sample.board.length);
      for (let j = 0; j < sample.board.length; j++) {
        expect(encoded.board[j], `sample ${i} board[${j}]`).toBeCloseTo(sample.board[j], 5);
      }
      expect(
        encoded.globalFeatures.length,
        `sample ${i} globalFeatures length`,
      ).toBe(sample.globalFeatures.length);
      for (let j = 0; j < sample.globalFeatures.length; j++) {
        expect(encoded.globalFeatures[j], `sample ${i} globalFeatures[${j}]`).toBeCloseTo(
          sample.globalFeatures[j],
          5,
        );
      }

      const moves = generateLegalMoves(state);
      const actualKeys = new Set(legalActionKeys(state, moves).map((k) => k.join(",")));
      const expectedKeys = new Set(sample.actionKeys.map((k) => k.join(",")));
      expect(actualKeys, `sample ${i} action keys`).toEqual(expectedKeys);
    });
  });
});
