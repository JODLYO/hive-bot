"""Self-play/train iterate loop: generate games with the current network,
push samples into a replay buffer, train on sampled batches, checkpoint,
repeat with the updated weights. This is the function the Colab notebook
calls into.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..engine.constants import BASE_PIECE_TYPES, PieceType
from ..model.network import HiveNet, score_actions
from .replay_buffer import ReplayBuffer
from .selfplay import DEFAULT_MAX_PLIES, Sample, play_game


def compute_loss(
    model: HiveNet, batch: list[Sample]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = torch.stack([s.board for s in batch])
    global_features = torch.stack([s.global_features for s in batch])
    output = model(boards, global_features)

    value_targets = torch.tensor([s.value_target for s in batch], dtype=torch.float32)
    value_loss = torch.nn.functional.mse_loss(output.value, value_targets)

    policy_losses = []
    for i, sample in enumerate(batch):
        scores = score_actions(output, sample.action_keys, batch_index=i)
        log_probs = torch.log_softmax(scores, dim=0)
        policy_losses.append(-(sample.target_policy * log_probs).sum())
    policy_loss = torch.stack(policy_losses).mean()

    return policy_loss + value_loss, policy_loss, value_loss


def train(
    iterations: int,
    games_per_iter: int,
    simulations: int,
    *,
    batch_size: int = 32,
    batches_per_iter: int = 20,
    buffer_capacity: int = 20_000,
    lr: float = 1e-3,
    enabled_types: frozenset[PieceType] = BASE_PIECE_TYPES,
    max_plies: int = DEFAULT_MAX_PLIES,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    seed: int | None = None,
    model: HiveNet | None = None,
) -> HiveNet:
    """`resume_from` continues a previous run's checkpoint (model +
    optimizer state, both needed for Adam's momentum to pick up cleanly)
    and carries its iteration count forward, so checkpoint numbering and
    training stay continuous across separate Colab sessions rather than
    resetting to 0 and overwriting old checkpoints each time this is
    called."""
    model = model if model is not None else HiveNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_iteration = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = checkpoint["iteration"] + 1

    buffer = ReplayBuffer(buffer_capacity)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    for offset in range(iterations):
        iteration = start_iteration + offset
        model.eval()
        for _ in range(games_per_iter):
            samples = play_game(
                model,
                simulations,
                enabled_types=enabled_types,
                rng=rng,
                np_rng=np_rng,
                max_plies=max_plies,
            )
            buffer.add_game(samples)

        model.train()
        last_losses = (float("nan"), float("nan"), float("nan"))
        for _ in range(batches_per_iter):
            if len(buffer) < batch_size:
                break
            batch = buffer.sample(batch_size, rng)
            total_loss, policy_loss, value_loss = compute_loss(model, batch)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            last_losses = (total_loss.item(), policy_loss.item(), value_loss.item())

        print(
            f"iteration {iteration}: buffer={len(buffer)} "
            f"loss={last_losses[0]:.4f} policy={last_losses[1]:.4f} value={last_losses[2]:.4f}"
        )

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                },
                checkpoint_dir / f"checkpoint_{iteration}.pt",
            )

    return model


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-play/train iterate loop.")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--games-per-iter", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches-per-iter", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--tiny-net",
        action="store_true",
        help="Use a small network (fast on CPU) instead of the real training size.",
    )
    args = parser.parse_args()

    net_kwargs = (
        {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}
        if args.tiny_net
        else {}
    )

    train(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        batch_size=args.batch_size,
        batches_per_iter=args.batches_per_iter,
        lr=args.lr,
        max_plies=args.max_plies,
        seed=args.seed,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        model=HiveNet(**net_kwargs),
    )


if __name__ == "__main__":
    _main()
