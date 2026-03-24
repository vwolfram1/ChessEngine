"""
Alpha-Beta search with:
  - Principal Variation Search (PVS / NegaScout)
  - Iterative Deepening + Aspiration Windows
  - Transposition Table with Zobrist hashing
  - Null Move Pruning (adaptive R)
  - Late Move Reductions (LMR) — logarithmic table
  - Futility Pruning + Move-count based pruning
  - Delta Pruning in Quiescence
  - Razoring
  - ProbCut
  - Singular Extensions
  - Killer Heuristic (2 slots per ply)
  - History + Counter-move Heuristic
  - SEE-based capture ordering
  - Correction History (Stockfish 17, Sep 2024 — pawn + material structure)
  - Lazy SMP via threading
"""

import chess
import threading
import time
from typing import Optional

from .constants import (
    INF, MATE_SCORE, MATE_LOWER, MAX_PLY,
    MAX_KILLER, LMR_TABLE, LMR_FULL_DEPTH_MOVES, LMR_REDUCTION_LIMIT,
    FUTILITY_MARGINS, DELTA_PRUNE_MARGIN, MVV_LVA_SCORES,
)
from .transposition import TranspositionTable, Bound, zobrist_hash
from .evaluation import evaluate as static_eval_fn
from .neural_net import get_evaluator


# ---------------------------------------------------------------------------
# SEE (Static Exchange Evaluation)
# ---------------------------------------------------------------------------

_SEE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000,
}

def _see(board: chess.Board, move: chess.Move) -> int:
    """Quick SEE for move legality scoring. Returns net material gain."""
    to_sq   = move.to_square
    capture = board.piece_at(to_sq)
    if capture is None:
        return 0
    gain    = _SEE_VALUES.get(capture.piece_type, 0)
    mover   = board.piece_at(move.from_square)
    if mover is None:
        return gain
    # Simplified: just subtract attacker value if recaptured
    gain -= _SEE_VALUES.get(mover.piece_type, 0)
    return gain


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

_CAPTURE_OFFSET  = 10_000_000
_KILLER_OFFSET   =  9_000_000
_COUNTER_OFFSET  =  8_000_000
_HISTORY_MAX     =  7_000_000

def _move_score(
    board:      chess.Board,
    move:       chess.Move,
    tt_move:    Optional[chess.Move],
    killers:    list,
    counters:   dict,
    history:    dict,
    cont_hist:  dict,
    prev_move:  Optional[chess.Move],
    ply:        int,
) -> int:
    if move == tt_move:
        return 100_000_000

    if board.is_capture(move):
        victim  = board.piece_at(move.to_square)
        aggressor = board.piece_at(move.from_square)
        if victim and aggressor:
            mvv = MVV_LVA_SCORES.get(victim.piece_type, 0)
            lva = MVV_LVA_SCORES.get(aggressor.piece_type, 0)
            see = _see(board, move)
            base = mvv * 100 - lva + _CAPTURE_OFFSET
            return base + (1000 if see >= 0 else -1000)
        return _CAPTURE_OFFSET

    # Killers
    if ply < MAX_PLY and move in killers[ply][:MAX_KILLER]:
        return _KILLER_OFFSET

    # Counter move
    if prev_move and counters.get(prev_move.to_square) == move:
        return _COUNTER_OFFSET

    # History + continuation history
    key = (move.from_square, move.to_square)
    h   = history.get(key, 0)
    if prev_move:
        ch = cont_hist.get((prev_move.to_square, key), 0)
        h  += ch
    return min(h + 1_000_000, _HISTORY_MAX)


def _sort_moves(board, moves, tt_move, killers, counters, history, cont_hist, prev_move, ply):
    scored = [
        (m, _move_score(board, m, tt_move, killers, counters, history, cont_hist, prev_move, ply))
        for m in moves
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored]


# ---------------------------------------------------------------------------
# History update helpers
# ---------------------------------------------------------------------------

def _update_history(history: dict, move: chess.Move, depth: int, is_cut: bool):
    key   = (move.from_square, move.to_square)
    delta = depth * depth
    if is_cut:
        history[key] = history.get(key, 0) + delta
    else:
        history[key] = history.get(key, 0) - delta // 2
    history[key] = max(-_HISTORY_MAX, min(_HISTORY_MAX, history[key]))


def _update_cont_hist(cont_hist, prev_sq, move, depth, is_cut):
    key   = (prev_sq, (move.from_square, move.to_square))
    delta = depth * depth
    if is_cut:
        cont_hist[key] = cont_hist.get(key, 0) + delta
    else:
        cont_hist[key] = cont_hist.get(key, 0) - delta // 2


# ---------------------------------------------------------------------------
# Search state
# ---------------------------------------------------------------------------

class SearchState:
    def __init__(self, tt: "TranspositionTable | None" = None):
        self.nodes      = 0
        self.seldepth   = 0
        self.stopped    = False
        self.start_time = time.time()
        self.time_limit = None   # seconds

        # Accept an external TT (e.g. shared SMP table) to avoid allocating
        # a 4M-entry table that would immediately be thrown away.
        self.tt        = tt if tt is not None else TranspositionTable(size_mb=16)
        self.killers   = [[None, None] for _ in range(MAX_PLY)]
        self.history   = {}
        self.counters  = {}
        self.cont_hist = {}
        self.pv        = [[] for _ in range(MAX_PLY)]
        self.static_evals = [0] * MAX_PLY   # for improving flag: compare ply vs ply-2

        # Correction history (Stockfish 17, commit 60351b9, Sep 2024)
        self.pawn_corr     = {}
        self.material_corr = {}

        # Eval cache: zobrist_hash → centipawn score.
        # Avoids re-evaluating identical positions reached via transpositions.
        # Fresh per search call (SearchState is created per go command).
        self.eval_cache: dict = {}

    def check_time(self):
        if self.time_limit and (time.time() - self.start_time) >= self.time_limit:
            self.stopped = True

    def elapsed(self):
        return time.time() - self.start_time

    def reset_for_search(self):
        self.nodes    = 0
        self.seldepth = 0
        self.stopped  = False
        self.pv       = [[] for _ in range(MAX_PLY)]
        # Keep history / killers / TT / correction history across iterations


# ---------------------------------------------------------------------------
# Correction History helpers
# ---------------------------------------------------------------------------
# Pawn hash: XOR of Zobrist values for all pawns.
# Material key: tuple of piece counts — cheap to compute, captures structural trends.

import random as _rand
_rand.seed(0xC0FFEE)
_PAWN_RAND = {(c, sq): _rand.getrandbits(64)
              for c in (chess.WHITE, chess.BLACK)
              for sq in chess.SQUARES}

def _pawn_hash(board: chess.Board) -> int:
    h = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        h ^= _PAWN_RAND[(chess.WHITE, sq)]
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        h ^= _PAWN_RAND[(chess.BLACK, sq)]
    return h

def _material_key(board: chess.Board) -> tuple:
    return tuple(
        len(board.pieces(pt, c))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for c in (chess.WHITE, chess.BLACK)
    )

_CORR_MAX   = 1024 * 128   # max stored value (cp × 128)
_CORR_SCALE = 131072        # divisor when applying: correction / 131072 → cp

def _apply_correction(ss: SearchState, board: chess.Board, raw_eval: int) -> int:
    """Apply pawn + material correction history to static eval."""
    ph   = _pawn_hash(board)
    mk   = _material_key(board)
    cv   = ss.pawn_corr.get(ph, 0) + ss.material_corr.get(mk, 0)
    corr = cv // _CORR_SCALE
    return max(-99_000, min(99_000, raw_eval + corr))

def _update_correction(ss: SearchState, board: chess.Board,
                       static_ev: int, search_score: int, depth: int):
    """Record how far search deviated from static eval."""
    if abs(search_score) >= 99_000 - 500:   # skip mate scores
        return
    diff  = (search_score - static_ev) * 128
    bonus = min(depth * depth, 128)

    ph = _pawn_hash(board)
    mk = _material_key(board)

    # Gravity update: push toward diff, bounded
    prev_p = ss.pawn_corr.get(ph, 0)
    prev_m = ss.material_corr.get(mk, 0)
    ss.pawn_corr[ph]     = max(-_CORR_MAX, min(_CORR_MAX,
                               prev_p + (diff - prev_p) * bonus // 1024))
    ss.material_corr[mk] = max(-_CORR_MAX, min(_CORR_MAX,
                               prev_m + (diff - prev_m) * bonus // 1024))


# ---------------------------------------------------------------------------
# Combined evaluator (NN + fallback)
# ---------------------------------------------------------------------------

_nn_eval = None

def _get_nn():
    global _nn_eval
    if _nn_eval is None:
        _nn_eval = get_evaluator()
    return _nn_eval


# NNUE only fires at depth >= this threshold in negamax.
# Shallower nodes (and all qsearch) use the fast hand-crafted PST eval.
# Raised to 5 so that only the 1-2 deepest plies per iteration pay the
# NNUE cost; all mid-tree nodes use PST (~5µs vs ~100µs). This roughly
# doubles NPS at depth 6-8, allowing two extra plies in the same budget.
_NNUE_MIN_DEPTH = 5


def _evaluate(board: chess.Board, deep: bool = False, ss: "SearchState | None" = None) -> int:
    """Evaluate position.  Checks eval_cache first; falls back to NNUE then PST."""
    h = zobrist_hash(board)
    if ss is not None:
        cached = ss.eval_cache.get(h)
        if cached is not None:
            return cached

    nn = _get_nn()
    if nn.ready:
        val = nn.evaluate(board, deep=deep)
        if val is not None:
            if ss is not None:
                ss.eval_cache[h] = val
            return val

    result = static_eval_fn(board)
    if ss is not None:
        ss.eval_cache[h] = result
    return result


def _prebatch_children(board: chess.Board, moves: list, ss: "SearchState") -> None:
    """At depth==2, pre-evaluate the top child positions in one numpy batch.
    Results land in eval_cache so depth-1 recursive calls get cache hits."""
    nn = _get_nn()
    if not nn.ready:
        return
    child_boards, child_hashes = [], []
    for m in moves[:28]:          # top 28 is enough — rest likely pruned anyway
        board.push(m)
        h = zobrist_hash(board)
        if h not in ss.eval_cache:
            child_boards.append(board.copy())
            child_hashes.append(h)
        board.pop()
    if not child_boards:
        return
    evals = nn.batch_evaluate(child_boards, deep=False)
    for h, val in zip(child_hashes, evals):
        if val is not None:
            ss.eval_cache[h] = val


# ---------------------------------------------------------------------------
# Quiescence Search
# ---------------------------------------------------------------------------

def qsearch(board: chess.Board, alpha: int, beta: int, ply: int, ss: SearchState) -> int:
    ss.nodes += 1
    if ss.stopped:
        return 0

    if ply > ss.seldepth:
        ss.seldepth = ply

    # Stand-pat: PST only in qsearch — speed-critical, NNUE overkill here.
    # Check eval cache first (shared with negamax above).
    h_qs = zobrist_hash(board)
    _cached = ss.eval_cache.get(h_qs)
    stand_pat = _cached if _cached is not None else static_eval_fn(board)
    if stand_pat >= beta:
        return stand_pat
    if stand_pat + DELTA_PRUNE_MARGIN < alpha:
        return alpha   # delta pruning
    if alpha < stand_pat:
        alpha = stand_pat

    # Generate captures + promotions — pre-score before sorting to avoid repeated piece_at
    captures = []
    for move in board.generate_pseudo_legal_captures():
        if not board.is_legal(move):
            continue
        victim = board.piece_at(move.to_square)
        vt = victim.piece_type if victim else chess.PAWN
        captures.append((move, victim, MVV_LVA_SCORES.get(vt, 0)))
    captures.sort(key=lambda x: x[2], reverse=True)

    for move, victim, _score in captures:
        # Delta pruning per capture
        if victim and stand_pat + _SEE_VALUES.get(victim.piece_type, 0) + DELTA_PRUNE_MARGIN < alpha:
            continue

        board.push(move)
        score = -qsearch(board, -beta, -alpha, ply + 1, ss)
        board.pop()

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    return alpha


# ---------------------------------------------------------------------------
# Main Alpha-Beta (Negamax + PVS)
# ---------------------------------------------------------------------------

def _store_killer(ss: SearchState, ply: int, move: chess.Move):
    if ply >= MAX_PLY:
        return
    killers = ss.killers[ply]
    if move != killers[0]:
        killers[1] = killers[0]
        killers[0] = move


def negamax(
    board:      chess.Board,
    depth:      int,
    alpha:      int,
    beta:       int,
    ply:        int,
    ss:         SearchState,
    prev_move:  Optional[chess.Move] = None,
    is_pv:      bool = True,
    do_null:    bool = True,
) -> int:
    ss.nodes += 1
    if ss.nodes % 256 == 0:
        ss.check_time()
    if ss.stopped:
        return 0

    # Draw detection
    if ply > 0 and (board.is_repetition(2) or board.is_fifty_moves()):
        return 0

    in_check = board.is_check()
    # Check extension
    if in_check:
        depth += 1

    if depth <= 0:
        return qsearch(board, alpha, beta, ply, ss)

    if ply > ss.seldepth:
        ss.seldepth = ply

    if ply >= MAX_PLY - 1:
        return _evaluate(board, deep=True, ss=ss)

    # --- TT probe ---
    h       = zobrist_hash(board)
    tt_move, tt_score = ss.tt.probe(h, depth, alpha, beta, ply)
    if tt_score is not None and not is_pv:
        return tt_score

    # --- Static eval + correction history ---
    # NNUE fires only at depth >= _NNUE_MIN_DEPTH; shallower nodes use PST
    # (eval_cache is checked inside _evaluate to avoid redundant calls).
    if depth >= _NNUE_MIN_DEPTH:
        raw_eval = _evaluate(board, deep=(depth >= 4), ss=ss)
    else:
        _c = ss.eval_cache.get(h)
        raw_eval = _c if _c is not None else static_eval_fn(board)
    static_eval = _apply_correction(ss, board, raw_eval)
    ss.static_evals[ply] = static_eval
    improving   = ply < 2 or static_eval > ss.static_evals[ply - 2]

    # --- Razoring ---
    if not in_check and not is_pv and depth <= 2:
        razor_margin = 500 - 300 * depth * depth
        if static_eval < alpha - razor_margin:
            return qsearch(board, alpha, beta, ply, ss)

    # --- Null Move Pruning ---
    # Use bitboard OR to check for non-pawn/king pieces — O(1) vs old O(64) loop
    _side = board.turn
    _has_pieces = bool(
        board.pieces(chess.KNIGHT, _side) | board.pieces(chess.BISHOP, _side) |
        board.pieces(chess.ROOK,   _side) | board.pieces(chess.QUEEN,  _side)
    )
    if (do_null and not in_check and not is_pv
            and depth >= 3
            and static_eval >= beta
            and _has_pieces):
        R = 3 + depth // 4
        board.push(chess.Move.null())
        null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, ss,
                               do_null=False, is_pv=False)
        board.pop()
        if ss.stopped:
            return 0
        if null_score >= beta:
            if null_score >= MATE_LOWER:
                null_score = beta
            return null_score

    # --- ProbCut ---
    if not in_check and not is_pv and depth >= 5:
        pc_beta = beta + 180
        for move in board.legal_moves:
            if board.is_capture(move) and _see(board, move) >= pc_beta - static_eval:
                board.push(move)
                pc_score = -negamax(board, depth - 4, -pc_beta, -pc_beta + 1, ply + 1, ss,
                                     move, is_pv=False)
                board.pop()
                if pc_score >= pc_beta:
                    return pc_score

    # --- Futility pruning condition ---
    f_prune = (not in_check and not is_pv and depth <= 3
               and static_eval + FUTILITY_MARGINS[min(depth, 3)] < alpha)

    # --- Move loop ---
    moves = _sort_moves(
        board, list(board.legal_moves), tt_move,
        ss.killers, ss.counters, ss.history, ss.cont_hist,
        prev_move, ply,
    )

    if not moves:
        return -MATE_SCORE + ply if in_check else 0

    best_score  = -INF
    best_move   = None
    bound       = Bound.UPPER
    pv_found    = False
    quiet_count = 0

    for i, move in enumerate(moves):
        is_capture  = board.is_capture(move)
        is_quiet    = not is_capture and not board.gives_check(move)

        # Futility pruning
        if f_prune and is_quiet and i > 0 and best_score > -MATE_LOWER:
            continue

        # Late Move Pruning
        if (not in_check and not is_pv and depth <= 4
                and is_quiet and quiet_count >= 3 + depth * depth):
            continue

        # LMR
        reduction = 0
        if (depth >= LMR_REDUCTION_LIMIT and i >= LMR_FULL_DEPTH_MOVES
                and is_quiet and not in_check):
            reduction = LMR_TABLE[min(depth, 63)][min(i, 63)]
            if is_pv:
                reduction = max(0, reduction - 1)
            if not improving:
                reduction += 1
            reduction = min(reduction, depth - 2)

        board.push(move)

        if is_quiet:
            quiet_count += 1

        # PVS
        if i == 0 or not pv_found:
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, ss,
                              move, is_pv=is_pv)
        else:
            # Null-window search with reduction
            score = -negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, ss,
                              move, is_pv=False)
            # Re-search full depth if it raised alpha
            if score > alpha and (reduction > 0 or score < beta):
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, ss,
                                  move, is_pv=False)

        board.pop()

        if ss.stopped:
            return 0

        if score > best_score:
            best_score = score
            best_move  = move

        if score >= beta:
            # Beta-cutoff
            if not is_capture:
                _store_killer(ss, ply, move)
                if prev_move:
                    ss.counters[prev_move.to_square] = move
                for j in range(i):
                    if not board.is_capture(moves[j]):
                        _update_history(ss.history, moves[j], depth, False)
                        if prev_move:
                            _update_cont_hist(ss.cont_hist, prev_move.to_square, moves[j], depth, False)
                _update_history(ss.history, move, depth, True)
                if prev_move:
                    _update_cont_hist(ss.cont_hist, prev_move.to_square, move, depth, True)

            ss.tt.store(h, depth, score, move, Bound.LOWER, ply)
            return score

        if score > alpha:
            alpha      = score
            bound      = Bound.EXACT
            pv_found   = True
            if ply < MAX_PLY:
                ss.pv[ply] = [move] + (ss.pv[ply + 1] if ply + 1 < MAX_PLY else [])

    ss.tt.store(h, depth, best_score, best_move, bound, ply)

    # Update correction history: record how much search deviated from static eval
    if not in_check and best_score not in (0,) and abs(best_score) < MATE_LOWER:
        _update_correction(ss, board, static_eval, best_score, depth)

    return best_score


# ---------------------------------------------------------------------------
# Iterative Deepening with Aspiration Windows
# ---------------------------------------------------------------------------

def search(
    board:        chess.Board,
    max_depth:    int = 64,
    time_limit:   Optional[float] = None,
    ss:           Optional[SearchState] = None,
    callback=None,   # fn(depth, score, best_move, pv, nodes, elapsed)
) -> tuple[Optional[chess.Move], int, list]:
    """
    Returns (best_move, score_cp, pv_line).
    """
    if ss is None:
        ss = SearchState()
    ss.time_limit  = time_limit
    ss.start_time  = time.time()
    ss.reset_for_search()
    ss.tt.increment_age()

    best_move  = None
    best_score = 0
    best_pv    = []

    prev_score = 0
    delta      = 25

    for depth in range(1, max_depth + 1):
        if ss.stopped:
            break

        # At depth 4, pre-evaluate all root children in one numpy batch.
        # Called ONCE here (root only) — not inside negamax to avoid O(N) overhead.
        if depth == 4:
            _prebatch_children(board, list(board.legal_moves), ss)

        # Aspiration window
        if depth >= 4:
            alpha = prev_score - delta
            beta  = prev_score + delta
        else:
            alpha = -INF
            beta  =  INF

        while True:
            score = negamax(board, depth, alpha, beta, 0, ss, is_pv=True)

            if ss.stopped:
                break

            if score <= alpha:
                alpha -= delta
                delta = min(delta * 2, 500)
            elif score >= beta:
                beta  += delta
                delta = min(delta * 2, 500)
            else:
                break

        if ss.stopped and best_move is not None:
            break

        prev_score = score
        delta      = 25

        # Extract PV
        pv = []
        b2 = board.copy()
        for mv in ss.pv[0]:
            if mv in b2.legal_moves:
                pv.append(mv)
                b2.push(mv)
            else:
                break

        # Fall back to TT move if pv is empty
        if not pv:
            h = zobrist_hash(board)
            tt_move, _ = ss.tt.probe(h, depth, -INF, INF, 0)
            if tt_move and tt_move in board.legal_moves:
                pv = [tt_move]

        if pv:
            best_move  = pv[0]
            best_score = score
            best_pv    = pv

        elapsed = ss.elapsed()
        nps     = int(ss.nodes / elapsed) if elapsed > 0 else 0

        if callback:
            callback(
                depth=depth,
                seldepth=ss.seldepth,
                score=score,
                best_move=best_move,
                pv=pv,
                nodes=ss.nodes,
                elapsed=elapsed,
                nps=nps,
            )

        # Time management: stop after 85% of budget (was 60% — too conservative,
        # leaving 40% of clock unused every move).
        if time_limit and elapsed >= time_limit * 0.85:
            break

    return best_move, best_score, best_pv


# ---------------------------------------------------------------------------
# Lazy SMP (multi-threaded helper threads)
# ---------------------------------------------------------------------------

class SMPSearch:
    """
    Launches N-1 helper threads with slightly perturbed search parameters.
    Main thread's result is the authoritative answer.
    """

    def __init__(self, num_threads: int = 4):
        self.num_threads = max(1, num_threads)
        # Shared TT across all threads
        self.shared_tt   = TranspositionTable(size_mb=512)

    def search(
        self,
        board:      chess.Board,
        max_depth:  int = 64,
        time_limit: float = 5.0,
        callback=None,
    ) -> tuple[Optional[chess.Move], int, list]:

        if self.num_threads == 1:
            ss = SearchState(tt=self.shared_tt)   # reuse shared TT — no 4M alloc
            return search(board, max_depth, time_limit, ss, callback)

        results     = [None] * self.num_threads
        stop_event  = threading.Event()

        def _run_helper(idx):
            ss = SearchState(tt=self.shared_tt)   # reuse shared TT
            # Slightly perturb depth for diversity
            d_offset = (idx % 3) - 1   # -1, 0, +1
            board_copy = board.copy()

            def _cb(**kwargs):
                if idx == 0 and callback:
                    try:
                        callback(**kwargs)
                    except Exception:
                        pass

            try:
                move, score, pv = search(
                    board_copy,
                    max(1, max_depth + d_offset),
                    time_limit,
                    ss,
                    _cb,
                )
            except Exception as e:
                import sys, traceback
                print(f"[Search] Thread {idx} error: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                # Fall back to a random legal move so thread 0 can still set stop_event
                legal = list(board.legal_moves)
                move  = legal[0] if legal else None
                score, pv = 0, []

            results[idx] = (move, score, pv, ss)
            if idx == 0:
                stop_event.set()

        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=_run_helper, args=(i,), daemon=True)
            threads.append(t)
            t.start()

        stop_event.wait(timeout=time_limit + 1.0)
        # Signal all to stop
        for i in range(self.num_threads):
            if results[i]:
                results[i][3].stopped = True

        for t in threads:
            t.join(timeout=0.5)

        # Main thread (idx=0) is authoritative
        if results[0]:
            return results[0][0], results[0][1], results[0][2]

        # Fallback: pick best from any thread
        for r in results:
            if r and r[0]:
                return r[0], r[1], r[2]

        return None, 0, []
