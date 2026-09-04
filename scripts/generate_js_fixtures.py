"""Dev-only tool: dump JSON fixtures from the already oracle-validated
Python fast engine, for js/test/*.parity.test.ts to replay against the
TypeScript port and check for exact behavioral parity. Not shipped as part
of the hive_bot package -- see the plan doc's Phase 5.

Usage: uv run python scripts/generate_js_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from hive_bot.engine.actions import legal_action_keys
from hive_bot.engine.apply import apply_move
from hive_bot.engine.constants import BASE_PIECE_TYPES, EXPANSION_PIECE_TYPES, PieceType
from hive_bot.engine.encode import encode_state
from hive_bot.engine.moves import Move, generate_legal_moves
from hive_bot.engine.state import GameState


def serialize_move(move: Move) -> dict[str, Any]:
    return {
        "kind": int(move.kind),
        "pieceId": move.piece_id,
        "to": list(move.to),
        "thrownPieceId": move.thrown_piece_id,
    }


def serialize_state(state: GameState) -> dict[str, Any]:
    return {
        "pieces": [
            {"id": pid, "pieceType": int(p.piece_type), "owner": p.owner}
            for pid, p in state.pieces.items()
        ],
        # "q,r,s" string keys -- matches engine/state.ts's posKey() format
        # exactly, so the TS side never needs a separate parsing scheme.
        # `list(stack)` etc. below are real copies, not aliases -- state.board's
        # stacks, state.hand's sublists, and state.queen_placed are all
        # mutated in place by apply_move, so storing the bare references
        # would let every earlier snapshot silently mutate along with them.
        "board": {f"{q},{r},{s}": list(stack) for (q, r, s), stack in state.board.items()},
        "position": {
            str(pid): (list(pos) if pos is not None else None)
            for pid, pos in state.position.items()
        },
        "hand": [list(state.hand[0]), list(state.hand[1])],
        "queenPlaced": list(state.queen_placed),
        "currentPlayer": state.current_player,
        "turnNo": state.turn_no,
        "ply": state.ply,
        "lastMovedPieceId": state.last_moved_piece_id,
        "lastMovedPly": state.last_moved_ply,
        "gameOver": state.game_over,
        "winner": state.winner,
    }


def generate_game(
    seed: int, enabled_types: frozenset[PieceType], max_plies: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    state = GameState.new_game(enabled_types)
    steps = []
    for _ in range(max_plies):
        if state.game_over:
            break
        legal = generate_legal_moves(state)
        if not legal:
            break
        applied_index = rng.randrange(len(legal))
        steps.append(
            {
                "state": serialize_state(state),
                "legalMoves": [serialize_move(m) for m in legal],
                "appliedMoveIndex": applied_index,
            }
        )
        apply_move(state, legal[applied_index])
    # Only append a trailing "terminal" entry if the game actually ended --
    # if the loop just ran out of max_plies mid-game, `state` still has
    # real legal moves (a live, non-terminal position), so a synthetic
    # empty-legalMoves entry for it would be wrong, not just superfluous.
    if state.game_over:
        steps.append(
            {"state": serialize_state(state), "legalMoves": [], "appliedMoveIndex": None}
        )
    return {
        "enabledTypes": sorted(int(t) for t in enabled_types),
        "steps": steps,
    }


def serialize_encoded_sample(state: GameState) -> dict[str, Any]:
    """A single (state, tensor, action keys) snapshot -- for
    encode.parity.test.ts to check engine/encode.ts and engine/actions.ts
    numerically against encode_state()/legal_action_keys(). Kept separate
    from the per-ply game fixtures above (rather than adding this to every
    step) since a full board tensor is ~54k floats -- fine for a couple
    dozen sampled snapshots, not for every ply of every game."""
    encoded = encode_state(state)
    legal = generate_legal_moves(state)
    keys = legal_action_keys(state, legal)
    return {
        "state": serialize_state(state),
        "board": encoded.board.flatten().tolist(),
        "globalFeatures": encoded.global_features.tolist(),
        "actionKeys": [list(k) for k in keys],
    }


def generate_encode_fixtures(num_samples: int, max_plies: int) -> list[dict[str, Any]]:
    samples = []
    seed = 10_000
    while len(samples) < num_samples:
        rng = random.Random(seed)
        enabled_types = (
            BASE_PIECE_TYPES if seed % 2 == 0 else BASE_PIECE_TYPES | EXPANSION_PIECE_TYPES
        )
        state = GameState.new_game(enabled_types)
        for _ in range(rng.randint(1, max_plies)):
            if state.game_over:
                break
            legal = generate_legal_moves(state)
            if not legal:
                break
            apply_move(state, legal[rng.randrange(len(legal))])
        if not state.game_over:
            samples.append(serialize_encoded_sample(state))
        seed += 1
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default="js/test/fixtures")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--expansion-games", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=60)
    parser.add_argument("--encode-samples", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.json"):
        path.unlink()

    for seed in range(args.games):
        game = generate_game(seed, BASE_PIECE_TYPES, args.max_plies)
        out_path = out_dir / f"base_game_{seed}.json"
        out_path.write_text(json.dumps(game))
        print(f"wrote {out_path} ({len(game['steps'])} steps)")

    full_set = BASE_PIECE_TYPES | EXPANSION_PIECE_TYPES
    for seed in range(args.expansion_games):
        game = generate_game(seed, full_set, args.max_plies)
        out_path = out_dir / f"expansion_game_{seed}.json"
        out_path.write_text(json.dumps(game))
        print(f"wrote {out_path} ({len(game['steps'])} steps)")

    encode_samples = generate_encode_fixtures(args.encode_samples, args.max_plies)
    out_path = out_dir / "encode_fixtures.json"
    out_path.write_text(json.dumps(encode_samples))
    print(f"wrote {out_path} ({len(encode_samples)} samples)")


if __name__ == "__main__":
    main()
