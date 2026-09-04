.PHONY: setup lint format typecheck test test-full selfplay train export-onnx clean \
	js-setup js-typecheck js-test js-build js-fixtures \
	scrape-setup scrape-games build-pretrain-dataset pretrain

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

# Export a trained checkpoint to ONNX for the in-browser bot, e.g.:
#   make export-onnx CHECKPOINT=~/Downloads/checkpoint_4.pt OUT=js/public/hive_net.onnx
export-onnx:
	uv run python -m hive_bot.export.onnx_export "$(CHECKPOINT)" "$(OUT)"

# js/ is the in-browser bot (TS engine + MCTS + ONNX Runtime Web), a
# standalone npm package, not part of the uv-managed Python side above.
js-setup:
	cd js && npm install

js-typecheck:
	cd js && npx tsc --noEmit -p tsconfig.test.json

js-test:
	cd js && npx vitest run

js-build:
	cd js && npm run build

# Regenerate js/test/fixtures/ from the Python engine -- run this after any
# change to src/hive_bot/engine, then `make js-test` to check the TS port
# still matches.
js-fixtures:
	uv run python scripts/generate_js_fixtures.py


# Bootstrap training from real hivegame.com games -- see the plan doc.
# scrape-games hits a small community-run free server via a real browser
# (its archive search isn't a documented public API); keep --pages/--delay
# reasonable rather than hammering it. Re-running appends new games and
# skips ones already captured (dedup'd by game_id).
scrape-setup:
	uv sync --extra scrape
	uv run playwright install chromium

scrape-games:
	uv run python scripts/scrape_hivegame_archive.py --pages 20

build-pretrain-dataset:
	uv run python scripts/build_pretrain_dataset.py

# Quick local smoke run, same spirit as `make selfplay`/`make train`: tiny
# network, a couple epochs, just confirming the pretrain plumbing runs and
# writes a checkpoint -- run `make build-pretrain-dataset` first (needs
# data/hivegame_archive/games.jsonl from `make scrape-games`).
pretrain:
	uv run python -m hive_bot.training.pretrain --epochs 2 --tiny-net

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
