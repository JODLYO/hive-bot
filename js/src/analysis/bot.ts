// Load an exported model and get a "what's the best move here, and how
// good is this position" readout for a GameState -- a direct port of the
// Python package's analysis/bot.py, the chess-engine-style analysis API
// this whole project is for.

import type { Move } from "../engine/moves.js";
import type { GameState } from "../engine/state.js";
import type { Node } from "../model/mcts.js";
import { MCTS, visitCounts } from "../model/mcts.js";
import type { ModelSource } from "../model/network.js";
import { HiveNetSession } from "../model/network.js";

export interface MoveEvaluation {
  move: Move;
  visitCount: number;
  visitFraction: number; // visitCount / total root visits -- search's confidence in this move
}

export interface PositionAnalysis {
  bestMove: Move;
  winProbability: number; // for state.currentPlayer, in [0, 1]
  moveEvaluations: MoveEvaluation[]; // sorted by visitCount, descending
}

function analysisFromRoot(root: Node): PositionAnalysis {
  const pairs = visitCounts(root);
  if (pairs.length === 0) {
    throw new Error("cannot analyze a finished (or move-less) position");
  }
  const total = pairs.reduce((sum, [, count]) => sum + count, 0);
  const evaluations = pairs
    .map(([move, visitCount]) => ({ move, visitCount, visitFraction: visitCount / total }))
    .sort((a, b) => b.visitCount - a.visitCount);

  // root.value is the search's mean backed-up value for state.currentPlayer,
  // in [-1, 1] (loss..win) -- rescale to a [0, 1] win probability.
  const winProbability = (root.value + 1.0) / 2.0;
  return {
    bestMove: evaluations[0].move,
    winProbability,
    moveEvaluations: evaluations,
  };
}

/** Load a model once, then call `analyze` on as many positions as you
 * like. */
export class HiveBot {
  private readonly mcts: MCTS;

  constructor(
    model: HiveNetSession,
    private readonly numSimulations: number = 400,
    cPuct = 1.5,
  ) {
    this.mcts = new MCTS(model, { cPuct });
  }

  static async fromModel(source: ModelSource, numSimulations = 400): Promise<HiveBot> {
    const model = await HiveNetSession.load(source);
    return new HiveBot(model, numSimulations);
  }

  async analyze(state: GameState): Promise<PositionAnalysis> {
    if (state.gameOver) throw new Error("cannot analyze a finished game");
    const root = await this.mcts.run(state, this.numSimulations);
    return analysisFromRoot(root);
  }
}
