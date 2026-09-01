"""Load a trained checkpoint and get a "what's the best move here, and how
good is this position" readout for a `GameState` -- the chess-engine-style
analysis API the project is ultimately for. Usable standalone (see
notebooks/train_colab.ipynb for a usage example) and, later, from the
Django app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..engine.moves import Move
from ..engine.state import GameState
from ..model.mcts import MCTS, Node, visit_counts
from ..model.network import HiveNet


@dataclass(slots=True)
class MoveEvaluation:
    move: Move
    visit_count: int
    visit_fraction: (
        float  # visit_count / total root visits -- search's confidence in this move
    )


@dataclass(slots=True)
class PositionAnalysis:
    best_move: Move
    win_probability: float  # for state.current_player, in [0, 1]
    move_evaluations: list[MoveEvaluation]  # sorted by visit_count, descending


def _analysis_from_root(root: Node) -> PositionAnalysis:
    pairs = visit_counts(root)
    if not pairs:
        raise ValueError("cannot analyze a finished (or move-less) position")
    total = sum(count for _, count in pairs)
    evaluations = sorted(
        (MoveEvaluation(move, count, count / total) for move, count in pairs),
        key=lambda e: e.visit_count,
        reverse=True,
    )
    # root.value is the search's mean backed-up value for state.current_player,
    # in [-1, 1] (loss..win) -- rescale to a [0, 1] win probability.
    win_probability = (root.value + 1.0) / 2.0
    return PositionAnalysis(
        best_move=evaluations[0].move,
        win_probability=win_probability,
        move_evaluations=evaluations,
    )


class HiveBot:
    """Load a checkpoint once, then call `analyze` on as many positions as
    you like."""

    def __init__(
        self, model: HiveNet, num_simulations: int = 400, c_puct: float = 1.5
    ) -> None:
        self.model = model
        self.num_simulations = num_simulations
        self.mcts = MCTS(model, c_puct=c_puct)

    @classmethod
    def from_checkpoint(
        cls, path: Path, num_simulations: int = 400, **network_kwargs: Any
    ) -> HiveBot:
        """`network_kwargs` must match whatever HiveNet(...) sizing the
        checkpoint was trained with (trunk_channels, num_blocks, ...) --
        the checkpoint only stores weights, not the architecture."""
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model = HiveNet(**network_kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model, num_simulations=num_simulations)

    def analyze(self, state: GameState) -> PositionAnalysis:
        if state.game_over:
            raise ValueError("cannot analyze a finished game")
        root = self.mcts.run(state, num_simulations=self.num_simulations)
        return _analysis_from_root(root)
