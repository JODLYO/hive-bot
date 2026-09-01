"""Self-play game generation: play full games with MCTS+network, recording
one training sample per ply.

Each sample's policy target is the root's visit-count distribution (the
standard AlphaZero target -- search improves on the raw prior, so training
the network to predict visit counts is what lets it get better than its own
priors over successive iterations). The value target is the game's final
outcome from whichever player was on move at that ply -- consistent with
`encode_state` always encoding "current player as me" and the network's
value head being trained to predict exactly that.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch

from ..engine.actions import ActionKey, move_to_action_key
from ..engine.apply import apply_move
from ..engine.constants import BASE_PIECE_TYPES, MAX_PLIES_BEFORE_DRAW, PieceType
from ..engine.encode import encode_state
from ..engine.state import DRAW, GameState
from ..model.mcts import MCTS, select_move, visit_counts
from ..model.network import HiveNet

# The engine itself guarantees game_over by MAX_PLIES_BEFORE_DRAW (see
# constants.py), so this loop's own cap is just a defensive fallback -- kept
# a bit above the engine's so that rule is always what actually fires.
DEFAULT_MAX_PLIES = MAX_PLIES_BEFORE_DRAW + 10
DEFAULT_TEMPERATURE_PLIES = (
    20  # sample proportional to visits before this ply, greedy after
)


@dataclass(slots=True)
class Sample:
    board: torch.Tensor
    global_features: torch.Tensor
    action_keys: list[ActionKey]
    target_policy: torch.Tensor  # aligned with action_keys, sums to 1
    value_target: float  # from the perspective of whoever was on move this ply


def play_game(
    model: HiveNet,
    num_simulations: int,
    *,
    c_puct: float = 1.5,
    enabled_types: frozenset[PieceType] = BASE_PIECE_TYPES,
    temperature_plies: int = DEFAULT_TEMPERATURE_PLIES,
    max_plies: int = DEFAULT_MAX_PLIES,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> list[Sample]:
    rng = rng if rng is not None else random.Random()
    np_rng = np_rng if np_rng is not None else np.random.default_rng()

    state = GameState.new_game(enabled_types)
    mcts = MCTS(model, c_puct=c_puct, rng=np_rng)

    pending: list[
        tuple[torch.Tensor, torch.Tensor, list[ActionKey], torch.Tensor, int]
    ] = []
    for ply in range(max_plies):
        if state.game_over:
            break
        root = mcts.run(state, num_simulations=num_simulations, add_root_noise=True)
        pairs = visit_counts(root)
        keys = [move_to_action_key(state, move) for move, _ in pairs]
        counts = torch.tensor([count for _, count in pairs], dtype=torch.float32)
        target_policy = counts / counts.sum()

        encoded = encode_state(state)
        pending.append(
            (
                encoded.board,
                encoded.global_features,
                keys,
                target_policy,
                state.current_player,
            )
        )

        temperature = 1.0 if ply < temperature_plies else 0.0
        move = select_move(root, temperature, rng=rng)
        apply_move(state, move)

    outcome = state.winner  # None if max_plies was hit without a result

    samples = []
    for board, global_features, keys, target_policy, mover in pending:
        if outcome is None or outcome == DRAW:
            value_target = 0.0
        else:
            value_target = 1.0 if outcome == mover else -1.0
        samples.append(
            Sample(
                board=board,
                global_features=global_features,
                action_keys=keys,
                target_policy=target_policy,
                value_target=value_target,
            )
        )
    return samples


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Play self-play games with a fresh network."
    )
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tiny-net",
        action="store_true",
        help="Use a small network (fast on CPU) instead of the real training size.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    net_kwargs = (
        {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}
        if args.tiny_net
        else {}
    )
    model = HiveNet(**net_kwargs)
    model.eval()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    for i in range(args.games):
        samples = play_game(
            model, args.simulations, rng=rng, np_rng=np_rng, max_plies=args.max_plies
        )
        outcome = samples[-1].value_target if samples else float("nan")
        print(f"game {i}: {len(samples)} plies, final-mover value_target={outcome:+.0f}")


if __name__ == "__main__":
    _main()
