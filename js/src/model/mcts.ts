// AlphaZero-style PUCT MCTS -- a direct port of the Python package's
// model/mcts.py. No random rollouts: leaf values come from HiveNet's value
// head, and the search prioritizes children by its policy head's priors
// combined with visit counts.
//
// Traversal shares a single mutable GameState, descending by calling the
// engine's applyMove and backing back out with undoMove (see engine/
// apply.ts) rather than cloning the state per node -- the same make/unmake
// approach the engine itself uses.
//
// Value convention (matches encode.ts always encoding "current player as
// me"): a node's `value` is always from the perspective of whichever
// player is to move *at that node*. Backing a leaf value up the tree
// negates it at every step, since each level up flips whose turn it is.
//
// Unlike the Python version, expansion is async (onnxruntime-web's
// inference is Promise-based, not synchronous like torch's), so `run`/
// `simulate`/`expand` all return Promises here.

import type { ActionKey } from "../engine/actions.js";
import { actionKeyToString, legalActionKeys } from "../engine/actions.js";
import type { UndoInfo } from "../engine/apply.js";
import { applyMove, undoMove } from "../engine/apply.js";
import { encodeState } from "../engine/encode.js";
import type { Move } from "../engine/moves.js";
import { generateLegalMoves } from "../engine/moves.js";
import type { GameState } from "../engine/state.js";
import { DRAW } from "../engine/state.js";
import type { HiveNetSession } from "./network.js";
import { scoreActions, softmax } from "./network.js";

export class Node {
  visitCount = 0;
  valueSum = 0;
  children: Map<string, Edge> = new Map();
  expanded = false;

  constructor(public prior: number) {}

  get value(): number {
    return this.visitCount ? this.valueSum / this.visitCount : 0;
  }
}

export interface Edge {
  move: Move;
  child: Node;
}

/** Value for `state.currentPlayer`, at a state where `state.gameOver` is
 * already true (nobody actually moves from here -- this just needs a
 * perspective to be consistent with non-terminal, network-evaluated
 * leaves for backprop). */
function terminalValue(state: GameState): number {
  if (state.winner === DRAW) return 0;
  if (state.winner === null) throw new Error("terminalValue called on a non-terminal state");
  return state.winner === state.currentPlayer ? 1 : -1;
}

export interface MCTSOptions {
  cPuct?: number;
  dirichletAlpha?: number;
  dirichletEpsilon?: number;
  /** Uniform [0, 1) random source, for Dirichlet noise sampling. Defaults
   * to Math.random; pass a seeded generator for reproducible tests. */
  random?: () => number;
}

export class MCTS {
  private readonly cPuct: number;
  private readonly dirichletAlpha: number;
  private readonly dirichletEpsilon: number;
  private readonly random: () => number;

  constructor(
    private readonly model: HiveNetSession,
    options: MCTSOptions = {},
  ) {
    this.cPuct = options.cPuct ?? 1.5;
    this.dirichletAlpha = options.dirichletAlpha ?? 0.3;
    this.dirichletEpsilon = options.dirichletEpsilon ?? 0.25;
    this.random = options.random ?? Math.random;
  }

  async run(state: GameState, numSimulations: number, addRootNoise = false): Promise<Node> {
    const root = new Node(1.0);
    if (state.gameOver) return root;
    await this.expand(root, state);
    if (addRootNoise && root.children.size > 0) this.addDirichletNoise(root);
    for (let i = 0; i < numSimulations; i++) {
      await this.simulate(root, state);
    }
    return root;
  }

  private async simulate(root: Node, state: GameState): Promise<void> {
    let node = root;
    const path: [Edge, UndoInfo][] = [];

    while (node.expanded && node.children.size > 0) {
      const edge = this.selectChild(node);
      const undo = applyMove(state, edge.move);
      path.push([edge, undo]);
      node = edge.child;
    }

    let value = state.gameOver ? terminalValue(state) : await this.expand(node, state);

    for (let i = path.length - 1; i >= 0; i--) {
      const [edge, undo] = path[i];
      edge.child.visitCount += 1;
      edge.child.valueSum += value;
      value = -value;
      undoMove(state, undo);
    }
    // `value` has now been negated once per level back up to the root,
    // i.e. it's from the root's own perspective -- record it there too so
    // `root.value` means something (a position's overall estimated
    // value), not just its children's.
    root.visitCount += 1;
    root.valueSum += value;
  }

  private async expand(node: Node, state: GameState): Promise<number> {
    const moves = generateLegalMoves(state);
    node.expanded = true;
    if (moves.length === 0) {
      // Not expected for a non-game-over state (the engine auto-passes a
      // player with no legal move), but a defensive fallback beats a
      // crash mid-search.
      return 0;
    }

    const keys: ActionKey[] = legalActionKeys(state, moves);
    const encoded = encodeState(state);
    const output = await this.model.run(encoded.board, encoded.globalFeatures);
    const scores = scoreActions(output, keys);
    const priors = softmax(scores);

    for (let i = 0; i < moves.length; i++) {
      node.children.set(actionKeyToString(keys[i]), {
        move: moves[i],
        child: new Node(priors[i]),
      });
    }
    return output.value;
  }

  private selectChild(node: Node): Edge {
    let totalVisits = 0;
    for (const edge of node.children.values()) totalVisits += edge.child.visitCount;
    const sqrtTotal = Math.sqrt(Math.max(totalVisits, 1));

    let bestEdge: Edge | null = null;
    let bestScore = -Infinity;
    for (const edge of node.children.values()) {
      const q = -edge.child.value; // child's perspective is the opponent's
      const ucb =
        q + (this.cPuct * edge.child.prior * sqrtTotal) / (1 + edge.child.visitCount);
      if (ucb > bestScore) {
        bestScore = ucb;
        bestEdge = edge;
      }
    }
    if (!bestEdge) throw new Error("selectChild called on a node with no children");
    return bestEdge;
  }

  private addDirichletNoise(root: Node): void {
    const edges = Array.from(root.children.values());
    const noise = sampleDirichlet(edges.length, this.dirichletAlpha, this.random);
    edges.forEach((edge, i) => {
      edge.child.prior =
        edge.child.prior * (1 - this.dirichletEpsilon) + noise[i] * this.dirichletEpsilon;
    });
  }
}

export function visitCounts(root: Node): [Move, number][] {
  return Array.from(root.children.values()).map((edge) => [edge.move, edge.child.visitCount]);
}

/** Sample a move from the root's visit-count distribution -- greedy
 * (highest visit count) at temperature ~0, otherwise proportional to
 * visitCount ** (1 / temperature), the standard AlphaZero self-play
 * exploration schedule. */
export function selectMove(
  root: Node,
  temperature: number,
  random: () => number = Math.random,
): Move {
  const pairs = visitCounts(root);
  if (pairs.length === 0) throw new Error("cannot select a move from a root with no children");
  if (temperature <= 1e-3) {
    return pairs.reduce((best, p) => (p[1] > best[1] ? p : best))[0];
  }

  const weights = pairs.map(([, count]) => Math.pow(count, 1 / temperature));
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  let r = random() * totalWeight;
  for (let i = 0; i < pairs.length; i++) {
    r -= weights[i];
    if (r <= 0) return pairs[i][0];
  }
  return pairs[pairs.length - 1][0]; // floating-point rounding fallback
}

// --- Dirichlet noise sampling ------------------------------------------
//
// No built-in Gamma/Dirichlet sampler in JS. Standard construction: sample
// n independent Gamma(alpha, 1) variates and normalize by their sum.
// Gamma sampling via Marsaglia & Tsang (2000), with the usual boost trick
// for alpha < 1 (true here -- the default dirichletAlpha is 0.3).

function sampleStandardNormal(random: () => number): number {
  let u1 = 0;
  while (u1 === 0) u1 = random(); // avoid log(0)
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sampleGamma(shape: number, random: () => number): number {
  if (shape < 1) {
    const u = random();
    return sampleGamma(shape + 1, random) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = sampleStandardNormal(random);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function sampleDirichlet(n: number, alpha: number, random: () => number): number[] {
  const samples = Array.from({ length: n }, () => sampleGamma(alpha, random));
  const sum = samples.reduce((a, b) => a + b, 0);
  return samples.map((s) => s / sum);
}
