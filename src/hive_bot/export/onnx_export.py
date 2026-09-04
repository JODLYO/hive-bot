"""Export a trained checkpoint's `HiveNet.forward` to ONNX, for the
in-browser bot (see notebooks/ or the js/ package's README for how this
gets consumed by `onnxruntime-web`).

Only `forward()` (the CNN trunk + heads) is exported. `score_actions`
(network.py) deliberately isn't part of `forward()` -- it's a small,
parameter-free dot-product gather over a *dynamic* list of legal action
keys, which doesn't trace to a static ONNX graph and doesn't need to;
that logic gets reimplemented directly in JS against this model's raw
`from_map`/`to_map`/`hand_embed`/`kind_bias` outputs.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import torch

from ..engine.constants import BOARD_DIM
from ..engine.encode import NUM_GLOBAL_FEATURES, NUM_SPATIAL_CHANNELS
from ..model.network import HiveNet, load_hivenet_from_checkpoint

INPUT_NAMES = ["board", "global_features"]
OUTPUT_NAMES = ["from_map", "to_map", "hand_embed", "kind_bias", "value"]


def export_onnx(model: HiveNet, out_path: Path, opset_version: int = 17) -> None:
    """`model` must already be in eval mode (BatchNorm needs running stats,
    not batch stats, for correct single-sample inference) -- both
    `load_hivenet_from_checkpoint` and `HiveBot.from_checkpoint` already do
    this."""
    import onnx

    dummy_board = torch.zeros(1, NUM_SPATIAL_CHANNELS, BOARD_DIM, BOARD_DIM)
    dummy_global_features = torch.zeros(1, NUM_GLOBAL_FEATURES)

    dynamic_axes = {name: {0: "batch"} for name in [*INPUT_NAMES, *OUTPUT_NAMES]}
    # kind_bias is a fixed (2, embed_dim) parameter -- doesn't depend on the
    # input batch at all, so it has no batch axis to mark dynamic.
    del dynamic_axes["kind_bias"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "model.onnx"
        torch.onnx.export(
            model,
            (dummy_board, dummy_global_features),
            str(tmp_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
        # The exporter defaults to splitting weights into a companion
        # `.onnx.data` file. Fine for most tooling, but awkward for browser
        # deployment (two files, relative-path resolution) for a model this
        # small -- collapse back into one self-contained file.
        onnx_model = onnx.load(str(tmp_path), load_external_data=True)
        onnx.save_model(onnx_model, str(out_path), save_as_external_data=False)

    _verify(out_path)


def _verify(out_path: Path) -> None:
    import onnx

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a HiveNet checkpoint to ONNX for in-browser inference."
    )
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--trunk-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--global-hidden", type=int, default=64)
    args = parser.parse_args()

    network_kwargs: dict[str, Any] = {
        "trunk_channels": args.trunk_channels,
        "num_blocks": args.num_blocks,
        "embed_dim": args.embed_dim,
        "global_hidden": args.global_hidden,
    }
    model = load_hivenet_from_checkpoint(Path(args.checkpoint), **network_kwargs)
    export_onnx(model, Path(args.out), opset_version=args.opset_version)
    print(f"exported {args.checkpoint} -> {args.out}")


if __name__ == "__main__":
    _main()
