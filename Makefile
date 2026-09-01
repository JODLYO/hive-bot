.PHONY: setup lint format typecheck test test-full selfplay train clean

# Belt-and-suspenders alongside conftest.py: the editable install's own
# .pth mechanism has been observed to silently stop working in this
# environment (Python 3.14 + uv + hatchling), independent of anything in
# this project's code. Setting PYTHONPATH directly makes `python -m ...`
# invocations (which don't go through pytest/conftest.py) robust to that
# too.
export PYTHONPATH := src

setup:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

test:
	uv run pytest

# The exhaustive engine-vs-reference-oracle parity suite (~10 min): the real
# correctness gate for the engine, too slow for routine `make test`. Run
# before trusting any change to src/hive_bot/engine.
test-full:
	uv run pytest -m slow

# Quick local smoke runs: tiny network + few simulations + a low ply cap,
# just to confirm the self-play/train plumbing works end to end. Not
# representative of real training -- that happens in the Colab notebook
# with the full-size network and much higher simulation counts.
selfplay:
	uv run python -m hive_bot.training.selfplay --games 4 --simulations 32 --max-plies 60 --tiny-net

train:
	uv run python -m hive_bot.training.train --iterations 2 --games-per-iter 4 --simulations 32 --max-plies 60 --tiny-net

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
