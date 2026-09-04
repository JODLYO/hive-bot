"""Numeric-parity check for the ONNX export: the exported model's outputs
must match `HiveNet.forward`'s own outputs on the same inputs, otherwise
the in-browser bot would be running a different function than the one
that was actually trained/evaluated."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime
import torch

from hive_bot.engine.constants import BOARD_DIM
from hive_bot.engine.encode import NUM_GLOBAL_FEATURES, NUM_SPATIAL_CHANNELS
from hive_bot.export.onnx_export import OUTPUT_NAMES, export_onnx
from hive_bot.model.network import HiveNet

TINY_NET_KWARGS = {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}


def test_onnx_export_matches_pytorch_forward(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = HiveNet(**TINY_NET_KWARGS)
    model.eval()

    out_path = tmp_path / "model.onnx"
    export_onnx(model, out_path)
    assert out_path.exists()
    # Single self-contained file -- no companion `.onnx.data` alongside it.
    assert not (tmp_path / "model.onnx.data").exists()

    rng = torch.Generator().manual_seed(1)
    board = torch.rand((1, NUM_SPATIAL_CHANNELS, BOARD_DIM, BOARD_DIM), generator=rng)
    global_features = torch.rand((1, NUM_GLOBAL_FEATURES), generator=rng)

    with torch.no_grad():
        expected = model(board, global_features)

    session = onnxruntime.InferenceSession(str(out_path))
    onnx_outputs = session.run(
        OUTPUT_NAMES,
        {
            "board": board.numpy(),
            "global_features": global_features.numpy(),
        },
    )

    for name, onnx_output in zip(OUTPUT_NAMES, onnx_outputs, strict=True):
        expected_tensor = getattr(expected, name).detach().numpy()
        np.testing.assert_allclose(onnx_output, expected_tensor, rtol=1e-4, atol=1e-5)
