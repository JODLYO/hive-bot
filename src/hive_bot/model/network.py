"""Dual-headed (policy, value) CNN, AlphaZero-style, for the fast engine's
encoded board state (see engine/encode.py).

The policy head implements the dual-embedding bilinear design from the plan
doc: rather than a dense (from x to) action tensor -- intractable at
BOARD_DIM=55 -- the network produces per-cell `from`/`to` embeddings (plus a
context-dependent embedding per hand-piece-type for placements), and a
move's score is a dot product of its from/to embeddings. `score_actions`
gathers scores for exactly the legal action keys of a position (see
engine/actions.py) instead of ever materializing the full cross product.

MOVE and THROW actions can share the same `from` cell (a pillbug can throw a
piece, or that piece can move itself, from the same position) and always
share the same spatial `from` feature map here -- `kind_bias` adds a small
per-kind learned vector before the dot product so those two actions still
get distinct scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

from ..engine.actions import ActionKey
from ..engine.constants import BOARD_DIM, NUM_PIECE_TYPES
from ..engine.encode import NUM_GLOBAL_FEATURES, NUM_SPATIAL_CHANNELS
from ..engine.moves import MoveKind

_MOVE_KIND_SLOT = 0
_THROW_KIND_SLOT = 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class NetworkOutput(NamedTuple):
    from_map: torch.Tensor  # (B, D, H, W)
    to_map: torch.Tensor  # (B, D, H, W)
    hand_embed: torch.Tensor  # (B, NUM_PIECE_TYPES, D)
    kind_bias: torch.Tensor  # (2, D) -- indexed by _MOVE_KIND_SLOT / _THROW_KIND_SLOT
    value: torch.Tensor  # (B,)


class HiveNet(nn.Module):
    def __init__(
        self,
        trunk_channels: int = 64,
        num_blocks: int = 6,
        embed_dim: int = 32,
        global_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.global_mlp = nn.Sequential(
            nn.Linear(NUM_GLOBAL_FEATURES, global_hidden),
            nn.ReLU(),
        )
        self.global_to_planes = nn.Linear(global_hidden, trunk_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_SPATIAL_CHANNELS, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(trunk_channels) for _ in range(num_blocks)]
        )

        self.from_conv = nn.Conv2d(trunk_channels, embed_dim, 1)
        self.to_conv = nn.Conv2d(trunk_channels, embed_dim, 1)
        self.kind_bias = nn.Embedding(2, embed_dim)
        self.hand_head = nn.Linear(global_hidden, NUM_PIECE_TYPES * embed_dim)

        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * BOARD_DIM * BOARD_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor, global_features: torch.Tensor) -> NetworkOutput:
        """board: (B, NUM_SPATIAL_CHANNELS, H, W); global_features: (B, NUM_GLOBAL_FEATURES)."""
        global_hidden = self.global_mlp(global_features)  # (B, global_hidden)
        global_planes = self.global_to_planes(global_hidden)  # (B, trunk_channels)

        x = self.stem(board)
        x = x + global_planes[:, :, None, None]
        for block in self.blocks:
            x = block(x)

        from_map = self.from_conv(x)
        to_map = self.to_conv(x)
        hand_embed = self.hand_head(global_hidden).view(-1, NUM_PIECE_TYPES, self.embed_dim)
        value = self.value_head(x).squeeze(-1)

        return NetworkOutput(
            from_map=from_map,
            to_map=to_map,
            hand_embed=hand_embed,
            kind_bias=self.kind_bias.weight,
            value=value,
        )


def load_hivenet_from_checkpoint(path: Path, **network_kwargs: Any) -> HiveNet:
    """Load weights saved either as `train.py`'s checkpoint dict
    (`{"model": state_dict, "optimizer": ..., "iteration": ...}`) or a bare
    `model.state_dict()`. `network_kwargs` must match whatever `HiveNet(...)`
    sizing the checkpoint was trained with (trunk_channels, num_blocks,
    ...) -- the checkpoint only stores weights, not the architecture.
    Returns the model in eval mode (so BatchNorm uses its running stats,
    not batch stats -- required for correct single-sample inference)."""
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model = HiveNet(**network_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_actions(
    output: NetworkOutput, keys: Sequence[ActionKey], batch_index: int = 0
) -> torch.Tensor:
    """Raw (pre-softmax) scores, one per key in `keys` (same order), for a
    single sample in a (possibly batched) `output`."""
    if not keys:
        return torch.empty(0)

    device = output.value.device
    kinds = torch.tensor([k[0] for k in keys], dtype=torch.long, device=device)
    from_idx = torch.tensor([k[1] for k in keys], dtype=torch.long, device=device)
    to_idx = torch.tensor([k[2] for k in keys], dtype=torch.long, device=device)

    embed_dim = output.from_map.shape[1]
    to_map_flat = output.to_map[batch_index].reshape(embed_dim, -1)  # (D, H*W)
    to_vecs = to_map_flat[:, to_idx].T  # (N, D)

    from_map_flat = output.from_map[batch_index].reshape(embed_dim, -1)  # (D, H*W)
    is_throw = kinds == MoveKind.THROW
    is_place = kinds == MoveKind.PLACE

    kind_slot = torch.where(is_throw, _THROW_KIND_SLOT, _MOVE_KIND_SLOT)
    from_vecs = from_map_flat[:, from_idx].T + output.kind_bias[kind_slot]  # (N, D)

    if is_place.any():
        place_from_vecs = output.hand_embed[batch_index, from_idx[is_place]]
        from_vecs = from_vecs.clone()
        from_vecs[is_place] = place_from_vecs

    return (from_vecs * to_vecs).sum(dim=1)
