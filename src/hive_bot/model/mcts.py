"""AlphaZero-style PUCT MCTS: no random rollouts -- leaf values come from
`HiveNet`'s value head, and the search prioritizes children by its policy
head's priors combined with visit counts.

Traversal shares a single mutable `GameState`, descending by calling the
engine's `apply_move` and backing back out with `undo_move` (see
engine/apply.py) rather than cloning the state per node -- the same
make/unmake approach the engine itself uses, and for the same reason: MCTS
needs to apply/revert a great many moves per second.

Value convention (matches encode.py always encoding "current player as
me"): a node's `value` is always from the perspective of whichever player
is to move *at that node*. Backing a leaf value up the tree negates it at
every step, since each level up flips whose turn it is.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from ..engine.actions import ActionKey, legal_action_keys
from ..engine.apply import UndoInfo, apply_move, undo_move
from ..engine.encode import encode_state
from ..engine.moves import Move, generate_legal_moves
from ..engine.state import DRAW, GameState
from .network import HiveNet, score_actions


@dataclass(slots=True)
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[ActionKey, Edge] = field(default_factory=dict)
    expanded: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(slots=True)
class Edge:
    move: Move
    child: Node


def _terminal_value(state: GameState) -> float:
    """Value for `state.current_player`, at a state where `state.game_over`
    is already True (nobody actually moves from here -- this just needs a
    perspective to be consistent with non-terminal, network-evaluated
    leaves for backprop)."""
    if state.winner == DRAW:
        return 0.0
    assert state.winner is not None
    return 1.0 if state.winner == state.current_player else -1.0


class MCTS:
    def __init__(
        self,
        model: HiveNet,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._rng = rng if rng is not None else np.random.default_rng()

    def run(
        self, state: GameState, num_simulations: int, add_root_noise: bool = False
    ) -> Node:
        root = Node(prior=1.0)
        if state.game_over:
            return root
        self._expand(root, state)
        if add_root_noise and root.children:
            self._add_dirichlet_noise(root)
        for _ in range(num_simulations):
            self._simulate(root, state)
        return root

    def _simulate(self, root: Node, state: GameState) -> None:
        node = root
        path: list[tuple[Edge, UndoInfo]] = []

        while node.expanded and node.children:
            key, edge = self._select_child(node)
            undo = apply_move(state, edge.move)
            path.append((edge, undo))
            node = edge.child

        value = _terminal_value(state) if state.game_over else self._expand(node, state)

        for edge, undo in reversed(path):
            edge.child.visit_count += 1
            edge.child.value_sum += value
            value = -value
            undo_move(state, undo)
        # `value` has now been negated once per level back up to the root,
        # i.e. it's from the root's own perspective -- record it there too
        # so `root.value` means something (a position's overall estimated
        # value), not just its children's.
        root.visit_count += 1
        root.value_sum += value

    def _expand(self, node: Node, state: GameState) -> float:
        moves = generate_legal_moves(state)
        node.expanded = True
        if not moves:
            # Not expected for a non-game-over state (the engine auto-passes
            # a player with no legal move), but a defensive fallback beats
            # a crash mid-search.
            return 0.0

        keys = legal_action_keys(state, moves)
        encoded = encode_state(state)
        with torch.no_grad():
            output = self.model(
                encoded.board.unsqueeze(0), encoded.global_features.unsqueeze(0)
            )
            scores = score_actions(output, keys, batch_index=0)
            priors = torch.softmax(scores, dim=0)
            value = float(output.value.item())

        for move, key, prior in zip(moves, keys, priors.tolist(), strict=True):
            node.children[key] = Edge(move=move, child=Node(prior=prior))
        return value

    def _select_child(self, node: Node) -> tuple[ActionKey, Edge]:
        total_visits = sum(edge.child.visit_count for edge in node.children.values())
        sqrt_total = math.sqrt(max(total_visits, 1))

        best_key: ActionKey | None = None
        best_edge: Edge | None = None
        best_score = -math.inf
        for key, edge in node.children.items():
            q = -edge.child.value  # child's perspective is the opponent's
            ucb = q + self.c_puct * edge.child.prior * sqrt_total / (
                1 + edge.child.visit_count
            )
            if ucb > best_score:
                best_score = ucb
                best_key = key
                best_edge = edge

        assert best_key is not None and best_edge is not None
        return best_key, best_edge

    def _add_dirichlet_noise(self, root: Node) -> None:
        keys = list(root.children.keys())
        noise = self._rng.dirichlet([self.dirichlet_alpha] * len(keys))
        for key, n in zip(keys, noise, strict=True):
            edge = root.children[key]
            edge.child.prior = (
                edge.child.prior * (1 - self.dirichlet_epsilon) + n * self.dirichlet_epsilon
            )


def visit_counts(root: Node) -> list[tuple[Move, int]]:
    return [(edge.move, edge.child.visit_count) for edge in root.children.values()]


def select_move(root: Node, temperature: float, rng: random.Random) -> Move:
    """Sample a move from the root's visit-count distribution -- greedy
    (highest visit count) at temperature ~0, otherwise proportional to
    visit_count ** (1 / temperature), the standard AlphaZero self-play
    exploration schedule."""
    pairs = visit_counts(root)
    if not pairs:
        raise ValueError("cannot select a move from a root with no children")
    if temperature <= 1e-3:
        return max(pairs, key=lambda p: p[1])[0]

    weights = [count ** (1.0 / temperature) for _, count in pairs]
    return rng.choices([move for move, _ in pairs], weights=weights, k=1)[0]
