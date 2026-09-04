// ONNX Runtime Web wrapper around the exported HiveNet, plus a TS port of
// network.py's `score_actions`.
//
// Only `HiveNet.forward` (the CNN trunk + heads) is exported to ONNX (see
// export/onnx_export.py) -- `score_actions` itself is a small,
// parameter-free dot-product gather over a *dynamic* list of legal action
// keys, which doesn't trace to a static ONNX graph and doesn't need to.
// It's reimplemented here directly against the raw from_map/to_map/
// hand_embed/kind_bias tensors ONNX Runtime Web hands back -- see
// network.py's own `score_actions` docstring for the full rationale
// (MOVE/THROW sharing a `from` cell, disambiguated by `kindBias`).

import * as ort from "onnxruntime-web";

import type { ActionKey } from "../engine/actions.js";
import { BOARD_DIM } from "../engine/constants.js";
import { NUM_GLOBAL_FEATURES, NUM_SPATIAL_CHANNELS } from "../engine/encode.js";
import { MoveKind } from "../engine/moves.js";

const MOVE_KIND_SLOT = 0;
const THROW_KIND_SLOT = 1;

export interface NetworkOutput {
  fromMap: Float32Array; // flat [embedDim, BOARD_DIM, BOARD_DIM]
  toMap: Float32Array; // flat [embedDim, BOARD_DIM, BOARD_DIM]
  handEmbed: Float32Array; // flat [NUM_PIECE_TYPES, embedDim]
  kindBias: Float32Array; // flat [2, embedDim]
  value: number;
  embedDim: number;
  planeSize: number; // BOARD_DIM * BOARD_DIM
}

export type ModelSource = string | Uint8Array;

export class HiveNetSession {
  private constructor(private readonly session: ort.InferenceSession) {}

  /** `source` is a URL/path (browser: fetched by onnxruntime-web itself)
   * or the raw model bytes (e.g. read from disk in a Node test). Runs on
   * the single-threaded WASM backend by default -- the multi-threaded one
   * needs Cross-Origin-Opener-Policy/Cross-Origin-Embedder-Policy response
   * headers the app serving this isn't necessarily configured to send. */
  static async load(source: ModelSource): Promise<HiveNetSession> {
    ort.env.wasm.numThreads = 1;
    // Overload resolution doesn't accept a union argument even though each
    // branch individually matches one of InferenceSession.create's
    // overloads -- narrow explicitly instead.
    const session =
      typeof source === "string"
        ? await ort.InferenceSession.create(source)
        : await ort.InferenceSession.create(source);
    return new HiveNetSession(session);
  }

  async run(board: Float32Array, globalFeatures: Float32Array): Promise<NetworkOutput> {
    const boardTensor = new ort.Tensor("float32", board, [
      1,
      NUM_SPATIAL_CHANNELS,
      BOARD_DIM,
      BOARD_DIM,
    ]);
    const globalTensor = new ort.Tensor("float32", globalFeatures, [1, NUM_GLOBAL_FEATURES]);

    const results = await this.session.run({
      board: boardTensor,
      global_features: globalTensor,
    });

    const fromMapT = results.from_map;
    const toMapT = results.to_map;
    const embedDim = fromMapT.dims[1];
    const planeSize = fromMapT.dims[2] * fromMapT.dims[3];

    return {
      fromMap: fromMapT.data as Float32Array,
      toMap: toMapT.data as Float32Array,
      handEmbed: results.hand_embed.data as Float32Array,
      kindBias: results.kind_bias.data as Float32Array,
      value: (results.value.data as Float32Array)[0],
      embedDim,
      planeSize,
    };
  }
}

/** Raw (pre-softmax) scores, one per key in `keys` (same order). */
export function scoreActions(output: NetworkOutput, keys: readonly ActionKey[]): Float32Array {
  const { fromMap, toMap, handEmbed, kindBias, embedDim, planeSize } = output;
  const scores = new Float32Array(keys.length);

  for (let i = 0; i < keys.length; i++) {
    const [kind, fromIndex, toIndex] = keys[i];
    let sum = 0;
    if (kind === MoveKind.PLACE) {
      const handOffset = fromIndex * embedDim;
      for (let d = 0; d < embedDim; d++) {
        sum += handEmbed[handOffset + d] * toMap[d * planeSize + toIndex];
      }
    } else {
      const kindOffset = (kind === MoveKind.THROW ? THROW_KIND_SLOT : MOVE_KIND_SLOT) * embedDim;
      for (let d = 0; d < embedDim; d++) {
        const fromVal = fromMap[d * planeSize + fromIndex] + kindBias[kindOffset + d];
        sum += fromVal * toMap[d * planeSize + toIndex];
      }
    }
    scores[i] = sum;
  }
  return scores;
}

/** Numerically stable softmax over raw scores. */
export function softmax(scores: Float32Array): Float32Array {
  if (scores.length === 0) return scores;
  let max = -Infinity;
  for (const s of scores) if (s > max) max = s;
  const exp = new Float32Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    exp[i] = Math.exp(scores[i] - max);
    sum += exp[i];
  }
  for (let i = 0; i < exp.length; i++) exp[i] /= sum;
  return exp;
}
