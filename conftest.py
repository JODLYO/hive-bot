"""Make sure `src/hive_bot` is importable regardless of whether the
editable install's .pth mechanism is currently working -- it's been
observed to silently no-op in this environment (Python 3.14 + uv +
hatchling editable installs), independent of anything in this project's own
code. pytest loads this before collecting any test module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
