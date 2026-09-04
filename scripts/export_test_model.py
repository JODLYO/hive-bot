"""Dev-only tool: export a tiny, deterministically-seeded (untrained)
HiveNet to ONNX, committed as js/test/fixtures/tiny_model.onnx -- a small,
fast, reproducible fixture for the Node-side smoke test in
js/test/analyze.smoke.test.ts. Not shipped as part of the hive_bot package.

Usage: uv run python scripts/export_test_model.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from hive_bot.export.onnx_export import export_onnx
from hive_bot.model.network import HiveNet

TINY_NET_KWARGS = {"trunk_channels": 8, "num_blocks": 1, "embed_dim": 4, "global_hidden": 8}


def main() -> None:
    torch.manual_seed(0)
    model = HiveNet(**TINY_NET_KWARGS)
    model.eval()

    out_path = Path("js/test/fixtures/tiny_model.onnx")
    export_onnx(model, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
