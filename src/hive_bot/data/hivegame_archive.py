"""Replays hivegame.com's UHP-notation game histories through this engine,
turning each ply into a `training.selfplay.Sample` for supervised
pretraining -- see the plan doc ("Bootstrap training from real hivegame.com
games") for where this data comes from and how it's used.

Notation grammar (confirmed against hiveboardgame/hive's
`engine/src/direction.rs` and `engine/src/history.rs`): a position is
either "." (the game's very first placement), a piece label with no
direction symbol (climbing directly on top of that piece -- beetle/
mosquito only), or a piece label with a direction symbol as a prefix
("-", "/", "\\" for W, SW, NW respectively) or suffix (same three symbols
for E, NE, SE respectively).

This engine's own `HEX_DIRS` order has no relation to compass directions,
so the mapping below is just *a* consistent, geometrically valid labeling
(opposite pairs and cyclic adjacency preserved) -- correctness only
requires internal consistency, since every move in a game is resolved
relative to earlier moves in that same game, never compared across games.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..engine.actions import legal_action_keys
from ..engine.apply import apply_move
from ..engine.constants import PieceType
from ..engine.encode import encode_state
from ..engine.moves import Move, MoveKind, generate_legal_moves
from ..engine.state import GameState, Pos
from ..training.selfplay import Sample


def load_base_games_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Reads a scraped games.jsonl file (see scrape_hivegame_archive.py),
    keeping only base-piece games (`game_type == "Base"`, matching
    `BASE_PIECE_TYPES`). Returns the raw game dicts -- callers extract
    `history`/`winner_from_game_status(...)` themselves rather than this
    eagerly replaying anything, since replay produces the large encoded
    tensors that made holding the whole dataset in memory at once
    infeasible in the first place (see training/pretrain.py's chunked
    training loop, which replays a bounded number of games at a time
    instead)."""
    games: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            game = json.loads(line)
            if game.get("game_type") == "Base":
                games.append(game)
    return games


def winner_from_game_status(game_status: object) -> int | None:
    """`game_status` is `GameResponse.game_status` (hiveboardgame/hive's
    `GameStatus`), e.g. `{"Finished": {"Winner": "White"}}` or
    `{"Finished": {"Draw": None}}`. Returns 0/1 for White/Black winning, or
    None for a draw (or a game_status shape this doesn't recognize)."""
    if not isinstance(game_status, dict) or "Finished" not in game_status:
        return None
    finished = game_status["Finished"]
    if not isinstance(finished, dict) or "Winner" not in finished:
        return None  # draw, or a shape we don't recognize
    return 0 if finished["Winner"] == "White" else 1


_BUG_LETTER_TO_TYPE: dict[str, PieceType] = {
    "Q": PieceType.QUEEN,
    "A": PieceType.ANT,
    "S": PieceType.SPIDER,
    "B": PieceType.BEETLE,
    "G": PieceType.GRASSHOPPER,
    "M": PieceType.MOSQUITO,
    "L": PieceType.LADYBUG,
    "P": PieceType.PILLBUG,
}

_DIR_SYMBOLS = frozenset("-/\\")

# HEX_DIRS[0..5] taken in order and labeled NW, NE, E, SE, SW, W -- the
# cyclic order derived from hiveboardgame/hive's
# `Direction::adjacent_directions` (NW's neighbors are (W, NE), NE's are
# (NW, E), etc.), which makes opposite pairs 3 apart: NW<->SE, NE<->SW,
# E<->W. HEX_DIRS already guarantees index i and i+3 are negatives of each
# other, so this assignment is what actually makes the offsets internally
# consistent (an earlier version of this mapping used a guessed cyclic
# order that didn't preserve these opposite pairs, and silently produced
# wrong destinations a few moves into most real games).
_PREFIX_SYMBOL_TO_OFFSET: dict[str, Pos] = {}
_SUFFIX_SYMBOL_TO_OFFSET: dict[str, Pos] = {}


def _init_direction_tables() -> None:
    from ..engine.constants import HEX_DIRS

    nw, ne, e, se, sw, w = HEX_DIRS
    _SUFFIX_SYMBOL_TO_OFFSET.update({"-": e, "/": ne, "\\": se})
    _PREFIX_SYMBOL_TO_OFFSET.update({"-": w, "/": sw, "\\": nw})


_init_direction_tables()


class UhpReplayError(Exception):
    """Raised when a human game's recorded move can't be matched to a move
    this engine considers legal -- either a notation-parsing bug here, or a
    genuine rules disagreement worth investigating (see the plan doc's
    verification section)."""


def _parse_piece_label(label: str) -> tuple[int, PieceType]:
    color = label[0]
    bug_letter = label[1]
    if color not in "wb" or bug_letter not in _BUG_LETTER_TO_TYPE:
        raise UhpReplayError(f"unrecognized piece label {label!r}")
    owner = 0 if color == "w" else 1
    return owner, _BUG_LETTER_TO_TYPE[bug_letter]


def _resolve_position(pos_str: str, uhp_to_id: dict[str, int], state: GameState) -> Pos:
    if pos_str in (".", ""):
        # hivegame.com's stored history uses "" for the game's very first
        # placement; History::to_history_string (used e.g. for display)
        # would render it as "." -- accept either.
        return (0, 0, 0)

    if pos_str[0] in _DIR_SYMBOLS:
        symbol, ref_label = pos_str[0], pos_str[1:]
        offset = _PREFIX_SYMBOL_TO_OFFSET[symbol]
    elif pos_str[-1] in _DIR_SYMBOLS:
        symbol, ref_label = pos_str[-1], pos_str[:-1]
        offset = _SUFFIX_SYMBOL_TO_OFFSET[symbol]
    else:
        ref_label, offset = pos_str, None

    ref_id = uhp_to_id.get(ref_label)
    if ref_id is None:
        raise UhpReplayError(
            f"position {pos_str!r} references unplaced piece {ref_label!r}"
        )
    ref_pos = state.position[ref_id]
    if ref_pos is None:
        raise UhpReplayError(
            f"position {pos_str!r} references unplaced piece {ref_label!r}"
        )

    if offset is None:
        return ref_pos  # climbing directly on top of ref_label
    dq, dr, ds = offset
    return (ref_pos[0] + dq, ref_pos[1] + dr, ref_pos[2] + ds)


def _find_legal_move(
    state: GameState,
    moves: list[Move],
    piece_id: int | None,
    piece_type: PieceType,
    dest: Pos,
) -> Move:
    for move in moves:
        if move.to != dest:
            continue
        if move.kind == MoveKind.PLACE:
            if piece_id is None and state.pieces[move.piece_id].piece_type == piece_type:
                return move
        elif move.kind == MoveKind.MOVE:
            if move.piece_id == piece_id:
                return move
        else:  # THROW -- the human-visible mover is the thrown piece
            if move.thrown_piece_id == piece_id:
                return move
    kind = "PLACE" if piece_id is None else "MOVE/THROW"
    raise UhpReplayError(f"no legal {kind} move to {dest} (piece_type={piece_type.name})")


@dataclass(slots=True)
class ReplayResult:
    samples: list[Sample]
    final_state: GameState


def replay_uhp_game(
    history: list[tuple[str, str]],
    winner: int | None,
    enabled_types: frozenset[PieceType],
) -> ReplayResult:
    """`winner` is 0/1 for that player winning, or None for a draw.
    Raises `UhpReplayError` (caller's job to catch, log, and skip that game
    -- see the plan doc's verification section) if any recorded move
    doesn't match a move this engine considers legal at that point."""
    state = GameState.new_game(enabled_types)
    uhp_to_id: dict[str, int] = {}
    pending: list[tuple[Sample, int]] = []  # (sample missing value_target, mover)

    for ply_index, (piece_label, pos_str) in enumerate(history):
        if piece_label == "pass":
            # A player with no legal placement or move at all forfeits
            # their turn (recorded explicitly in hivegame.com's history --
            # see hive_lib::History::move_is_pass). This engine's own
            # apply_move already does the equivalent automatically after
            # the *previous* move (see apply.py's `_resolve_auto_pass`),
            # so by the time we reach this entry state.current_player has
            # already advanced past whoever passed -- nothing to replay
            # here. If the engine's auto-pass actually disagreed with
            # hivegame.com about whose turn is next, the very next real
            # move entry below will fail the owner-mismatch check, so
            # that's not re-validated here too.
            continue
        owner, piece_type = _parse_piece_label(piece_label)
        if owner != state.current_player:
            raise UhpReplayError(
                f"ply {ply_index}: {piece_label!r} belongs to player {owner}, "
                f"but it's player {state.current_player}'s turn"
            )
        is_placement = piece_label not in uhp_to_id
        try:
            dest = _resolve_position(pos_str, uhp_to_id, state)
            moves = generate_legal_moves(state)
            piece_id = None if is_placement else uhp_to_id[piece_label]
            move = _find_legal_move(state, moves, piece_id, piece_type, dest)
        except UhpReplayError as exc:
            raise UhpReplayError(
                f"ply {ply_index} ({piece_label} {pos_str}): {exc}"
            ) from exc
        if is_placement:
            uhp_to_id[piece_label] = move.piece_id

        encoded = encode_state(state)
        action_keys = legal_action_keys(state, moves)
        played_index = moves.index(move)
        target_policy = torch.zeros(len(action_keys))
        target_policy[played_index] = 1.0
        mover = state.current_player
        apply_move(state, move)

        sample = Sample(
            board=encoded.board,
            global_features=encoded.global_features,
            action_keys=action_keys,
            target_policy=target_policy,
            value_target=0.0,  # filled in once the game's outcome is known
        )
        pending.append((sample, mover))

    for sample, mover in pending:
        if winner is None:
            sample.value_target = 0.0
        else:
            sample.value_target = 1.0 if winner == mover else -1.0

    return ReplayResult(samples=[s for s, _ in pending], final_state=state)
