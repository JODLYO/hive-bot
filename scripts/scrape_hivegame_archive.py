"""Dev-only tool: pull real human games from hivegame.com's archive for
supervised pretraining -- see the plan doc ("Bootstrap training from real
hivegame.com games"). hivegame.com (github.com/hiveboardgame/hive) has no
documented public API; its archive search page calls a Leptos server
function (`get_batch_from_options`) over CBOR. Rather than hand-roll that
request (which would need to track the server's internal request schema
and break on redeploys), this drives a real headless browser against the
live archive page and intercepts its own requests' CBOR responses --
exactly what the page itself sends and receives.

This hits a small community-run free server, not a company API with
capacity to spare -- keep the delay between page loads generous (default
below) rather than hammering it.

Usage: uv run python scripts/scrape_hivegame_archive.py --pages 50
(needs the `scrape` extra: `uv sync --extra scrape` then
`uv run playwright install chromium` once).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cbor2
from playwright.sync_api import Response, sync_playwright

ARCHIVE_URL = (
    "https://hivegame.com/archive?rated=any&expansions=false&speeds="
    "Bullet,Blitz,Rapid,Classic,Correspondence,Untimed&result_filter=any"
    "&sort_key=Date&sort_asc=false"
    # 50 is the max allowed (shared_types::games_query_options::
    # ALLOWED_BATCH_SIZES is [10, 25, 50]) -- fewer round trips for the
    # same total games than the site's own default of 10.
    "&batch_size=50"
)

# Only what replay_uhp_game / the scraper's own bookkeeping actually need --
# GameResponse (apis/src/responses/game.rs) carries a lot more (timers,
# tournament info, ...) that's irrelevant here.
_KEEP_FIELDS = (
    "game_id",
    "game_type",
    "rated",
    "speed",
    "history",
    "game_status",
    "white_player",
    "black_player",
    "white_rating",
    "black_rating",
)


def _trim(game: dict[str, Any]) -> dict[str, Any]:
    return {k: game[k] for k in _KEEP_FIELDS if k in game}


def _load_seen_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with open(out_path) as f:
        return {json.loads(line)["game_id"] for line in f if line.strip()}


def scrape(
    out_path: Path,
    max_pages: int,
    delay_seconds: float,
    url: str = ARCHIVE_URL,
) -> None:
    # Pre-populate from whatever's already on disk -- out_path is opened in
    # append mode below specifically so re-running this (e.g. to fetch
    # newer games, or resume after an interrupted run) doesn't duplicate
    # games already written by a previous run.
    seen_ids = _load_seen_ids(out_path)
    written = 0

    with (
        sync_playwright() as p,
        p.chromium.launch() as browser,
        open(out_path, "a") as out_file,
    ):
        page = browser.new_page()

        def on_response(response: Response) -> None:
            nonlocal written
            if "get_batch_from_options" not in response.url or response.status != 200:
                return
            batch = cbor2.loads(response.body())
            for game in batch.get("games", []):
                game_id = game.get("game_id")
                if game_id in seen_ids:
                    continue
                seen_ids.add(game_id)
                out_file.write(json.dumps(_trim(game), default=str) + "\n")
                written += 1

        page.on("response", on_response)
        page.goto(url, wait_until="networkidle", timeout=30_000)
        page.wait_for_timeout(int(delay_seconds * 1000))

        next_page_button = page.get_by_role("button", name=re.compile("next", re.I)).first
        for i in range(max_pages):
            if not next_page_button.is_enabled():
                print(f"reached the last page after {i} page(s)")
                break
            next_page_button.click()
            page.wait_for_timeout(int(delay_seconds * 1000))
            print(f"page {i + 1}/{max_pages}: {written} games written so far")

    print(f"done -- {written} new games written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/hivegame_archive/games.jsonl"),
        help="Output JSONL file (appended to, so re-runs can resume).",
    )
    parser.add_argument("--pages", type=int, default=20, help="Max archive pages to fetch.")
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait after each page load/click -- be a good citizen.",
    )
    parser.add_argument("--url", type=str, default=ARCHIVE_URL)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scrape(args.out, args.pages, args.delay, args.url)


if __name__ == "__main__":
    main()
