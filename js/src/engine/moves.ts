// Legal move generation for the in-browser Hive engine -- a direct port
// of the Python package's engine/moves.py. See that file's docstring for
// the overall design (single-BFS `slideReachable` instead of validating
// one candidate at a time, placement actions keyed by piece *type* not
// instance). Behavior must stay identical to the Python engine, which is
// itself validated against the reference Django app -- see the plan doc's
// Phase 5 (parity fixtures) for how this file gets checked against it.

import { HEX_DIRS, PieceType, TURN_NUMBER_QUEEN_MUST_BE_PLACED } from "./constants.js";
import type { GameState, Owner, Pos } from "./state.js";
import { neighbors, parsePosKey, positionsConnected, posKey, sharedNeighbors } from "./state.js";

export enum MoveKind {
  PLACE = 0,
  MOVE = 1,
  THROW = 2,
}

export interface Move {
  readonly kind: MoveKind;
  readonly pieceId: number; // placed/moved piece, or the pillbug itself for THROW
  readonly to: Pos;
  readonly thrownPieceId?: number; // THROW only: the piece being lifted
}

export function generateLegalMoves(state: GameState): Move[] {
  if (state.gameOver) return [];
  const moves = placementMoves(state);
  if (state.queenPlaced[state.currentPlayer]) {
    moves.push(...boardMoves(state));
    moves.push(...pillbugThrowMoves(state));
  }
  return moves;
}

// --- placement -------------------------------------------------------------

function placementMoves(state: GameState): Move[] {
  const owner = state.currentPlayer;
  const hand = state.hand[owner];
  if (hand.length === 0) return [];
  const positions = placementPositions(state, owner);
  if (positions.size === 0) return [];
  const mustPlaceQueen =
    !state.queenPlaced[owner] && state.turnNo === TURN_NUMBER_QUEEN_MUST_BE_PLACED;

  const moves: Move[] = [];
  const seenTypes = new Set<PieceType>();
  for (const pieceId of hand) {
    const pieceType = state.pieces.get(pieceId)!.pieceType;
    if (seenTypes.has(pieceType)) continue;
    if (mustPlaceQueen && pieceType !== PieceType.QUEEN) continue;
    seenTypes.add(pieceType);
    for (const posKeyStr of positions) {
      moves.push({ kind: MoveKind.PLACE, pieceId, to: parsePosKey(posKeyStr) });
    }
  }
  return moves;
}

function placementPositions(state: GameState, owner: Owner): Set<string> {
  const occupied = state.occupied();
  if (occupied.size === 0) return new Set([posKey([0, 0, 0])]);

  const opponent: Owner = owner === 0 ? 1 : 0;
  if (state.turnNo === 1) {
    // Each side's very first placement just needs to touch the hive so
    // far, which (on turn 1) can only be the opponent's lone piece.
    const anchors = state.boardPositions(opponent);
    const result = new Set<string>();
    for (const anchorKey of anchors) {
      for (const nb of neighbors(parsePosKey(anchorKey))) {
        const key = posKey(nb);
        if (!occupied.has(key)) result.add(key);
      }
    }
    return result;
  }

  const ownPositions = state.boardPositions(owner);
  const oppPositions = state.boardPositions(opponent);
  const candidates = new Set<string>();
  for (const ownKey of ownPositions) {
    for (const nb of neighbors(parsePosKey(ownKey))) {
      const key = posKey(nb);
      if (!occupied.has(key)) candidates.add(key);
    }
  }
  const result = new Set<string>();
  for (const candidateKey of candidates) {
    const touchesOpponent = neighbors(parsePosKey(candidateKey)).some((nb) =>
      oppPositions.has(posKey(nb)),
    );
    if (!touchesOpponent) result.add(candidateKey);
  }
  return result;
}

// --- on-board moves ----------------------------------------------------------

function boardMoves(state: GameState): Move[] {
  const owner = state.currentPlayer;
  const moves: Move[] = [];
  for (const pieceId of state.piecesOnBoard(owner)) {
    const pos = state.position.get(pieceId)!;
    if (state.topPieceAt(pos) !== pieceId) continue; // buried, can't move
    const pieceType = state.pieces.get(pieceId)!.pieceType;
    for (const destKey of destinationsFor(state, pieceId, pieceType)) {
      const dest = parsePosKey(destKey);
      if (moveKeepsHiveConnected(state, pieceId, dest)) {
        moves.push({ kind: MoveKind.MOVE, pieceId, to: dest });
      }
    }
  }
  return moves;
}

function destinationsFor(state: GameState, pieceId: number, pieceType: PieceType): Set<string> {
  switch (pieceType) {
    case PieceType.QUEEN:
    case PieceType.PILLBUG:
      return slideDestinations(state, pieceId, 1);
    case PieceType.ANT:
      return slideDestinations(state, pieceId, null);
    case PieceType.SPIDER:
      return slideDestinations(state, pieceId, 3, 3);
    case PieceType.GRASSHOPPER:
      return grasshopperDestinations(state, pieceId);
    case PieceType.BEETLE:
      return beetleDestinations(state, pieceId);
    case PieceType.LADYBUG:
      return ladybugDestinations(state, pieceId);
    case PieceType.MOSQUITO:
      return mosquitoDestinations(state, pieceId);
    default:
      throw new Error(`unhandled piece type ${pieceType as number}`);
  }
}

/** Occupied-position set after hypothetically moving `pieceId` to `dest`.
 * Only vacates the start cell if this piece was its sole occupant -- a
 * piece stacked under a beetle/mosquito keeps that position "occupied"
 * for connectivity purposes when the top piece leaves. */
function resultingOccupied(state: GameState, pieceId: number, dest: Pos): Set<string> {
  const start = state.position.get(pieceId)!;
  const occ = new Set(state.occupied());
  if (state.stackHeightAt(start) === 1) occ.delete(posKey(start));
  occ.add(posKey(dest));
  return occ;
}

/** The official "One Hive Rule" requires connectivity to hold both
 * *during* a move (the instant the piece is lifted, before it lands --
 * real Hive rules forbid a move that would momentarily split the hive,
 * even if the piece's destination would reconnect it) and *after* it (the
 * final board). Only matters "during" when the start cell truly empties
 * (stack height 1); a beetle/mosquito climbing off a taller stack never
 * affects connectivity by leaving, since something remains there. */
function moveKeepsHiveConnected(state: GameState, pieceId: number, dest: Pos): boolean {
  const start = state.position.get(pieceId)!;
  if (state.stackHeightAt(start) === 1) {
    const during = new Set(state.occupied());
    during.delete(posKey(start));
    if (!positionsConnected(during)) return false;
  }
  return positionsConnected(resultingOccupied(state, pieceId, dest));
}

function slideDestinations(
  state: GameState,
  pieceId: number,
  maxSteps: number | null,
  exactSteps: number | null = null,
): Set<string> {
  const start = state.position.get(pieceId)!;
  const occupiedWithoutStart = new Set(state.occupied());
  occupiedWithoutStart.delete(posKey(start));
  return slideReachable(occupiedWithoutStart, start, maxSteps, exactSteps);
}

/** All destinations reachable from `start` by sliding across `occupied`
 * (which must NOT include `start` itself -- the piece is lifted first).
 * Exact-step sliding (spider) needs a different search than
 * unbounded/max-step sliding (ant/queen) -- see `slideReachableExact` for
 * why -- so this just dispatches to whichever is correct. */
function slideReachable(
  occupied: ReadonlySet<string>,
  start: Pos,
  maxSteps: number | null,
  exactSteps: number | null,
): Set<string> {
  if (exactSteps !== null) return slideReachableExact(occupied, start, exactSteps);

  // A single BFS pass computes every reachable destination instead of
  // validating one candidate at a time: since the wedge/gate check and
  // the "hive stays connected" check at each step only depend on the
  // current node and the single candidate next node (not on the path
  // taken to get there), and *any* non-self-crossing path to a node
  // within maxSteps is equally good enough (there's no "must be exactly
  // this many steps" requirement here), a node's reachability doesn't
  // depend on which path first discovered it -- so a single shared
  // `visited` set across the whole search is exact, not an
  // approximation. (This reasoning does NOT extend to exact-step sliding
  // -- see `slideReachableExact`.)
  const visited = new Set<string>([posKey(start)]);
  const reachable = new Set<string>();
  const queue: [Pos, number][] = [[start, 0]];
  let head = 0;
  while (head < queue.length) {
    const [current, steps] = queue[head++];
    for (const nb of neighbors(current)) {
      const nbKey = posKey(nb);
      if (visited.has(nbKey)) continue;
      visited.add(nbKey);
      if (occupied.has(nbKey)) continue;
      if (!positionsConnected(new Set([...occupied, nbKey]))) continue;
      const [n1, n2] = sharedNeighbors(current, nb);
      if (occupied.has(posKey(n1)) && occupied.has(posKey(n2))) continue; // wedged

      const newSteps = steps + 1;
      if (maxSteps === null || newSteps <= maxSteps) reachable.add(nbKey);
      if (maxSteps === null || newSteps < maxSteps) queue.push([nb, newSteps]);
    }
  }
  return reachable;
}

/** Destinations reachable from `start` in *exactly* `exactSteps` slides
 * (spider), without ever crossing a hex already visited earlier in that
 * same slide.
 *
 * Unlike max-step sliding, this can't reuse one shared `visited` set
 * across the whole search: a hex first reached via a *shorter* path
 * would then block rediscovering it via a *different*, longer,
 * non-self-crossing path that legitimately reaches it in exactly
 * `exactSteps` -- "no revisiting a hex" is a per-path rule, not a global
 * one across every path from `start`. A real game found this concretely:
 * a destination the shared-visited BFS marked unreachable (some other
 * path had already visited it after 2 steps) but which a valid 3-step
 * path -- through different intermediate hexes -- does reach. So this
 * instead explores every non-self-crossing path explicitly; cheap in
 * practice since `exactSteps` is always 3 (only spider uses this) with
 * branching factor at most 6.
 *
 * This exact bug is present in the Python engine's vendored reference
 * oracle's own `can_slide_path` (tests/reference/helpers.py, in the
 * hive-bot repo -- an unmodified copy of the real ttbg-web-app's own
 * code), so it isn't just a bug this port could have introduced -- it's
 * a real, reachable bug in the shipped Django app itself. See
 * engine/moves.py's `_slide_reachable_exact` (the Python side of this
 * same fix) for the full writeup. */
function slideReachableExact(
  occupied: ReadonlySet<string>,
  start: Pos,
  exactSteps: number,
): Set<string> {
  const reachable = new Set<string>();

  function dfs(current: Pos, path: ReadonlySet<string>, steps: number): void {
    if (steps === exactSteps) {
      reachable.add(posKey(current));
      return;
    }
    for (const nb of neighbors(current)) {
      const nbKey = posKey(nb);
      if (path.has(nbKey) || occupied.has(nbKey)) continue;
      if (!positionsConnected(new Set([...occupied, nbKey]))) continue;
      const [n1, n2] = sharedNeighbors(current, nb);
      if (occupied.has(posKey(n1)) && occupied.has(posKey(n2))) continue; // wedged
      dfs(nb, new Set([...path, nbKey]), steps + 1);
    }
  }

  dfs(start, new Set([posKey(start)]), 0);
  return reachable;
}

function grasshopperDestinations(state: GameState, pieceId: number): Set<string> {
  const start = state.position.get(pieceId)!;
  const occupied = state.occupied();
  const destinations = new Set<string>();
  for (const [dq, dr, ds] of HEX_DIRS) {
    let [q, r, s] = start;
    let hoppedOver = 0;
    for (;;) {
      q += dq;
      r += dr;
      s += ds;
      const key = posKey([q, r, s]);
      if (!occupied.has(key)) {
        if (hoppedOver > 0) destinations.add(key);
        break;
      }
      hoppedOver++;
    }
  }
  return destinations;
}

function beetleDestinations(state: GameState, pieceId: number): Set<string> {
  const start = state.position.get(pieceId)!;
  const pieceLevel = state.stackIndexOf(pieceId);
  const destinations = new Set<string>();
  for (const end of neighbors(start)) {
    const endStackLen = state.stackHeightAt(end);
    if (pieceLevel !== endStackLen) {
      // Climbing on/off a stack always changes level -- no gate check.
      destinations.add(posKey(end));
      continue;
    }
    const [n1, n2] = sharedNeighbors(start, end);
    const h1 = state.stackHeightAt(n1) - 1; // top-of-stack index, -1 if empty
    const h2 = state.stackHeightAt(n2) - 1;
    if (h1 >= pieceLevel && h2 >= pieceLevel) continue; // wedged between two taller (or equal) neighbors
    destinations.add(posKey(end));
  }
  return destinations;
}

function ladybugDestinations(state: GameState, pieceId: number): Set<string> {
  const start = state.position.get(pieceId)!;
  const occupied = new Set(state.occupied());
  occupied.delete(posKey(start));
  const destinations = new Set<string>();
  for (const a of neighbors(start)) {
    if (!occupied.has(posKey(a))) continue;
    for (const b of neighbors(a)) {
      if (posKey(b) === posKey(start) || !occupied.has(posKey(b))) continue;
      for (const c of neighbors(b)) {
        const cKey = posKey(c);
        if (cKey === posKey(start) || cKey === posKey(a) || occupied.has(cKey)) continue;
        destinations.add(cKey);
      }
    }
  }
  return destinations;
}

function mosquitoDestinations(state: GameState, pieceId: number): Set<string> {
  if (state.stackIndexOf(pieceId) > 0) {
    // Having climbed up (by copying a beetle) it now only moves as one.
    return beetleDestinations(state, pieceId);
  }
  const start = state.position.get(pieceId)!;
  const adjacentTypes = new Set<PieceType>();
  for (const nb of neighbors(start)) {
    const top = state.topPieceAt(nb);
    if (top !== null) adjacentTypes.add(state.pieces.get(top)!.pieceType);
  }
  adjacentTypes.delete(PieceType.MOSQUITO);
  const destinations = new Set<string>();
  for (const pieceType of adjacentTypes) {
    for (const key of destinationsFor(state, pieceId, pieceType)) destinations.add(key);
  }
  return destinations;
}

// --- pillbug throw -----------------------------------------------------------

function pillbugThrowMoves(state: GameState): Move[] {
  const owner = state.currentPlayer;
  const moves: Move[] = [];
  for (const pillbugId of state.piecesOnBoard(owner)) {
    if (state.pieces.get(pillbugId)!.pieceType !== PieceType.PILLBUG) continue;
    const pos = state.position.get(pillbugId)!;
    if (state.topPieceAt(pos) !== pillbugId) continue;
    const occupied = state.occupied();
    const emptyDests = neighbors(pos).filter((nb) => !occupied.has(posKey(nb)));
    if (emptyDests.length === 0) continue;

    for (const targetPos of neighbors(pos)) {
      const targetId = state.topPieceAt(targetPos);
      if (targetId === null || targetId === pillbugId) continue;
      if (state.stackHeightAt(targetPos) > 1) continue; // only a lone piece on the ground can be thrown
      if (state.isFrozen(targetId)) continue;
      for (const dest of emptyDests) {
        if (throwKeepsHiveConnected(state, targetId, dest)) {
          moves.push({ kind: MoveKind.THROW, pieceId: pillbugId, to: dest, thrownPieceId: targetId });
        }
      }
    }
  }
  return moves;
}

function throwKeepsHiveConnected(state: GameState, targetId: number, dest: Pos): boolean {
  const targetPos = state.position.get(targetId)!;
  const occ = new Set(state.occupied());
  occ.delete(posKey(targetPos)); // verified height 1 by the caller
  if (!positionsConnected(occ)) return false;
  occ.add(posKey(dest));
  return positionsConnected(occ);
}
