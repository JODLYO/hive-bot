"""Tests for `hive_bot.data.hivegame_archive`'s UHP-notation replay.

The two fixtures below are real, complete games pulled from hivegame.com's
archive (via the interception approach in scripts/scrape_hivegame_archive.py)
-- not hand-written, so they exercise the actual notation as hivegame.com
stores it (empty-string first move, real prefix/suffix direction symbols,
a genuine beetle-climb-onto-opponent in the second fixture) rather than a
guess at the grammar.

A sample of every recorded spider move initially failed to replay. That
turned out to be a real bug in `_slide_reachable`'s exact-3-step search
(see engine/moves.py's `_slide_reachable_exact` for the full story --
briefly, its single shared `visited` BFS could miss a valid destination
whenever a *different*, shorter path reached the same intermediate hex
first) -- not a notation-parsing issue, and not a hivegame.com quirk: the
same bug exists in the vendored reference oracle's own `can_slide_path`
(tests/reference/helpers.py), which is an unmodified copy of the real
ttbg-web-app's code. With that fixed, every real base-piece game sampled
from the archive (160/160 in one batch) replays cleanly, `"pass"` entries
included -- see build_pretrain_dataset.py if that ever regresses for a
newly-scraped batch.
"""

from __future__ import annotations

import pytest

from hive_bot.data.hivegame_archive import UhpReplayError, replay_uhp_game
from hive_bot.engine.constants import BASE_PIECE_TYPES

# White wins by move-count (25 plies), no beetle climbs -- the simple case.
_GAME_NO_CLIMB: list[tuple[str, str]] = [
    ("wG1", ""),
    ("bG1", "wG1\\"),
    ("wQ", "wG1/"),
    ("bG2", "/bG1"),
    ("wA1", "-wG1"),
    ("bQ", "bG1\\"),
    ("wA1", "bQ-"),
    ("bA1", "-bG2"),
    ("wA2", "wA1\\"),
    ("bA1", "wA2\\"),
    ("wA3", "wA1/"),
    ("bA2", "-bG2"),
    ("wA3", "/bA2"),
    ("bA3", "bG2\\"),
    ("wB1", "-wG1"),
    ("bA3", "wQ/"),
    ("wG2", "wA1/"),
    ("bB1", "bA3-"),
    ("wG2", "bQ\\"),
    ("bB1", "bA3\\"),
    ("wS1", "/wA3"),
    ("bB2", "\\bA3"),
    ("wS1", "bG2\\"),
    ("bB2", "\\wQ"),
    ("wA3", "\\wA1"),
]
_GAME_NO_CLIMB_WINNER = 0  # White

# Black wins (34 plies). Includes several beetle climbs, e.g. ("wB1",
# "bQ") -- a piece climbing directly on top of another with no direction
# symbol -- and ("bB1", "wB1") stacking onto an opponent.
_GAME_WITH_CLIMB: list[tuple[str, str]] = [
    ("wG1", ""),
    ("bG1", "wG1/"),
    ("wS1", "-wG1"),
    ("bQ", "bG1-"),
    ("wQ", "/wS1"),
    ("bA1", "\\bG1"),
    ("wA1", "\\wS1"),
    ("bA1", "-wQ"),
    ("wA1", "bQ-"),
    ("bB1", "\\bG1"),
    ("wB1", "wA1/"),
    ("bB1", "bG1"),
    ("wB1", "wA1"),
    ("bA2", "-bA1"),
    ("wB1", "bQ"),
    ("bB1", "wB1"),
    ("wA2", "wG1\\"),
    ("bG2", "\\bA1"),
    ("wA2", "\\wA1"),
    ("bA2", "wQ\\"),
    ("wG2", "wA2/"),
    ("bG2", "bA1\\"),
    ("wG2", "bG1\\"),
    ("bA3", "bA2-"),
    ("wS2", "\\wA2"),
    ("bG1", "wA1-"),
    ("wG3", "wS2/"),
    ("bA3", "wG3-"),
    ("wB2", "-wS2"),
    ("bG3", "bA2\\"),
    ("wB2", "\\bB1"),
    ("bG3", "\\wQ"),
    ("wB2", "\\wG2"),
    ("bA3", "wS1\\"),
]
_GAME_WITH_CLIMB_WINNER = 1  # Black

# White wins (37 plies). Black gets repeatedly stuck with no legal move
# once its pieces are pinned/immobilized, recording 8 real ("pass", "")
# entries (see hive_lib::History::move_is_pass) while White keeps moving.
_GAME_WITH_PASSES: list[tuple[str, str]] = [
    ("wG1", ""),
    ("bB1", "wG1\\"),
    ("wQ", "wG1/"),
    ("bQ", "bB1\\"),
    ("wA1", "-wG1"),
    ("bA1", "/bB1"),
    ("wA1", "/bA1"),
    ("bA2", "bQ/"),
    ("wA2", "-wG1"),
    ("bA2", "-wA2"),
    ("wA3", "wQ-"),
    ("bA3", "\\bA2"),
    ("wA3", "bQ-"),
    ("bA3", "wQ-"),
    ("wS1", "wA3/"),
    ("bA2", "wQ\\"),
    ("wQ", "\\bA3"),
    ("bS1", "bA3-"),
    ("wA2", "bS1-"),
    ("pass", ""),
    ("wS2", "wS1-"),
    ("pass", ""),
    ("wS2", "bQ\\"),
    ("pass", ""),
    ("wB1", "wS1\\"),
    ("pass", ""),
    ("wB1", "wS1"),
    ("pass", ""),
    ("wB2", "\\wG1"),
    ("pass", ""),
    ("wB2", "\\bA2"),
    ("bA2", "bA3\\"),
    ("wB2", "bA3"),
    ("pass", ""),
    ("wB1", "\\wA3"),
    ("pass", ""),
    ("wA1", "bA1\\"),
]
_GAME_WITH_PASSES_WINNER = 0  # White


def test_replay_no_climb_game_matches_move_count() -> None:
    result = replay_uhp_game(_GAME_NO_CLIMB, _GAME_NO_CLIMB_WINNER, BASE_PIECE_TYPES)
    assert len(result.samples) == len(_GAME_NO_CLIMB)
    assert result.final_state.game_over


def test_replay_climb_game_matches_move_count() -> None:
    result = replay_uhp_game(_GAME_WITH_CLIMB, _GAME_WITH_CLIMB_WINNER, BASE_PIECE_TYPES)
    assert len(result.samples) == len(_GAME_WITH_CLIMB)
    assert result.final_state.game_over


def test_value_targets_follow_winner_perspective() -> None:
    result = replay_uhp_game(_GAME_NO_CLIMB, _GAME_NO_CLIMB_WINNER, BASE_PIECE_TYPES)
    # Ply 0 is White's move (mover 0, the winner) -> +1.
    assert result.samples[0].value_target == pytest.approx(1.0)
    # Ply 1 is Black's move (mover 1, the loser) -> -1.
    assert result.samples[1].value_target == pytest.approx(-1.0)


def test_value_targets_are_zero_for_a_draw() -> None:
    result = replay_uhp_game(_GAME_NO_CLIMB, None, BASE_PIECE_TYPES)
    assert all(s.value_target == pytest.approx(0.0) for s in result.samples)


def test_target_policy_is_one_hot_over_full_legal_action_set() -> None:
    result = replay_uhp_game(_GAME_NO_CLIMB, _GAME_NO_CLIMB_WINNER, BASE_PIECE_TYPES)
    first = result.samples[0]
    # Turn 1's only legal move is placing the first piece at the origin.
    assert len(first.action_keys) == first.target_policy.numel()
    assert first.target_policy.sum().item() == pytest.approx(1.0)
    assert (first.target_policy == 1.0).sum().item() == 1


def test_unresolvable_reference_raises_uhp_replay_error() -> None:
    bad_history = [("wG1", ""), ("bA1", "wS1-")]  # wS1 was never placed
    with pytest.raises(UhpReplayError):
        replay_uhp_game(bad_history, None, BASE_PIECE_TYPES)


def test_replay_skips_pass_entries_and_samples_only_real_moves() -> None:
    result = replay_uhp_game(_GAME_WITH_PASSES, _GAME_WITH_PASSES_WINNER, BASE_PIECE_TYPES)
    non_pass_moves = sum(1 for label, _ in _GAME_WITH_PASSES if label != "pass")
    assert len(result.samples) == non_pass_moves
    assert result.final_state.game_over


def test_wrong_players_turn_raises_uhp_replay_error() -> None:
    # wG1 is White's first move; a second White-owned label out of turn
    # order should be rejected rather than silently misattributed.
    bad_history = [("wG1", ""), ("wA1", "wG1-")]
    with pytest.raises(UhpReplayError):
        replay_uhp_game(bad_history, None, BASE_PIECE_TYPES)
