"""
Main engine class. Exposes a clean API for the UI and UCI protocol.
Uses the Cython bitboard engine (CSearch) when available for maximum NPS,
falling back to the pure-Python SMP engine otherwise.
"""

import chess
import chess.polyglot
import os
import sys
import threading
from typing import Optional, Callable

from .search import SearchState, SMPSearch, search
from .evaluation import evaluate as static_eval
from .neural_net import get_evaluator

try:
    from .cx import CSearch, CYTHON_AVAILABLE
except ImportError:
    CSearch = None
    CYTHON_AVAILABLE = False

BOOK_PATH = os.path.join(os.path.dirname(__file__), "data", "komodo.bin")
TB_PATH   = os.path.join(os.path.dirname(__file__), "data", "syzygy")


class ChessEngine:
    def __init__(self, num_threads: int = 8, tt_mb: int = 512):
        self.board        = chess.Board()
        self.num_threads  = num_threads
        self.smp          = SMPSearch(num_threads=num_threads)
        self._lock        = threading.Lock()
        self._search_thread: Optional[threading.Thread] = None
        self._ss: Optional[SearchState] = None

        # Cython bitboard engine (primary search when available)
        self._csearch: Optional["CSearch"] = None
        if CYTHON_AVAILABLE:
            try:
                self._csearch = CSearch(tt_mb)
                print("[Engine] Cython bitboard engine active (~1M NPS)", file=sys.stderr)
            except Exception as e:
                print(f"[Engine] CSearch init failed: {e}", file=sys.stderr)

        # Syzygy tablebases
        self._tb_reader = None
        if os.path.isdir(TB_PATH):
            try:
                self._tb_reader = chess.syzygy.open_tablebase(TB_PATH)
                print(f"[Engine] Syzygy tablebases loaded from {TB_PATH}", file=sys.stderr)
            except Exception as e:
                print(f"[Engine] No Syzygy tablebases: {e}", file=sys.stderr)

        # Opening book
        self._book = None
        if os.path.exists(BOOK_PATH):
            try:
                self._book = chess.polyglot.open_reader(BOOK_PATH)
                print(f"[Engine] Opening book loaded: {BOOK_PATH}", file=sys.stderr)
            except Exception as e:
                print(f"[Engine] No opening book: {e}", file=sys.stderr)

        # Warm up neural net
        self._nn = get_evaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_position(self, fen: str = chess.STARTING_FEN, moves: list[str] = None):
        with self._lock:
            self.board = chess.Board(fen)
            for uci in (moves or []):
                self.board.push_uci(uci)
            if self._csearch:
                self._csearch.set_position(fen, list(moves or []))

    def get_fen(self) -> str:
        return self.board.fen()

    def make_move(self, uci: str) -> bool:
        with self._lock:
            try:
                move = chess.Move.from_uci(uci)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    return True
            except Exception:
                pass
            return False

    def undo_move(self) -> bool:
        with self._lock:
            if self.board.move_stack:
                self.board.pop()
                return True
            return False

    def get_legal_moves(self) -> list[str]:
        return [m.uci() for m in self.board.legal_moves]

    def get_evaluation(self) -> int:
        """Static evaluation of current position (centipawns, side-to-move POV)."""
        return static_eval(self.board.copy())

    def is_game_over(self) -> dict:
        b = self.board
        if b.is_checkmate():
            winner = "black" if b.turn == chess.WHITE else "white"
            return {"over": True, "reason": "checkmate", "winner": winner}
        if b.is_stalemate():
            return {"over": True, "reason": "stalemate", "winner": None}
        if b.is_insufficient_material():
            return {"over": True, "reason": "insufficient_material", "winner": None}
        if b.is_fifty_moves():
            return {"over": True, "reason": "fifty_moves", "winner": None}
        if b.is_repetition(3):
            return {"over": True, "reason": "repetition", "winner": None}
        return {"over": False}

    def best_move(
        self,
        time_limit: float = 5.0,
        depth_limit: int = 64,
        callback: Optional[Callable] = None,
    ) -> Optional[str]:
        """Blocking best-move search. Returns UCI string."""
        board_copy = self.board.copy()

        # Book move
        book_move = self._try_book(board_copy)
        if book_move:
            return book_move.uci()

        # Tablebase probe at root
        tb_move = self._try_tablebase(board_copy)
        if tb_move:
            return tb_move.uci()

        move, score, pv = self.smp.search(
            board_copy,
            max_depth=depth_limit,
            time_limit=time_limit,
            callback=callback,
        )
        return move.uci() if move else None

    def start_async_search(
        self,
        time_limit: float = 5.0,
        depth_limit: int = 64,
        on_depth: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
    ):
        """Non-blocking search. on_done(uci_move, score, pv) is called when finished."""
        self.stop_search()
        board_copy = self.board.copy()

        if self._csearch:
            def _run_cx():
                try:
                    # Book / tablebase checks still use python-chess
                    book_move = self._try_book(board_copy)
                    if book_move:
                        if on_done:
                            on_done(book_move.uci(), 0, [book_move.uci()])
                        return
                    tb_move = self._try_tablebase(board_copy)
                    if tb_move:
                        if on_done:
                            on_done(tb_move.uci(), 0, [tb_move.uci()])
                        return

                    def _cb(depth, seldepth, score, move_uci, nodes, nps, elapsed):
                        if on_depth:
                            # Adapt to the (depth,seldepth,score,best_move,pv,nodes,elapsed,nps) signature
                            on_depth(depth, seldepth, score, move_uci,
                                     [move_uci], nodes, elapsed, nps)

                    best_uci, score, depth = self._csearch.search(
                        time_limit, max_depth=depth_limit, callback=_cb
                    )
                    if on_done:
                        if not best_uci or best_uci == "0000":
                            legal = list(board_copy.legal_moves)
                            best_uci = legal[0].uci() if legal else "0000"
                        on_done(best_uci, score, [best_uci])
                except Exception as e:
                    print(f"[Engine] CSearch error: {e}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    if on_done:
                        legal = list(board_copy.legal_moves)
                        on_done(legal[0].uci() if legal else "0000", 0, [])

            self._search_thread = threading.Thread(target=_run_cx, daemon=True)
            self._search_thread.start()
            return

        # Python fallback
        def _run():
            try:
                book_move = self._try_book(board_copy)
                if book_move:
                    if on_done:
                        on_done(book_move.uci(), 0, [book_move.uci()])
                    return

                tb_move = self._try_tablebase(board_copy)
                if tb_move:
                    if on_done:
                        on_done(tb_move.uci(), 0, [tb_move.uci()])
                    return

                move, score, pv = self.smp.search(
                    board_copy,
                    max_depth=depth_limit,
                    time_limit=time_limit,
                    callback=on_depth,
                )
                if on_done:
                    pv_uci = [m.uci() for m in pv]
                    if not move:
                        legal = list(board_copy.legal_moves)
                        move = legal[0] if legal else None
                    on_done(move.uci() if move else "0000", score, pv_uci)
            except Exception as e:
                print(f"[Engine] Search error: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                if on_done:
                    legal = list(board_copy.legal_moves)
                    fallback = legal[0].uci() if legal else "0000"
                    on_done(fallback, 0, [])

        self._search_thread = threading.Thread(target=_run, daemon=True)
        self._search_thread.start()

    def stop_search(self):
        if self._ss:
            self._ss.stopped = True
        if self._csearch:
            self._csearch.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_book(self, board: chess.Board) -> Optional[chess.Move]:
        if not self._book:
            return None
        try:
            entry = self._book.weighted_choice(board)
            return entry.move
        except Exception:
            return None

    def _try_tablebase(self, board: chess.Board) -> Optional[chess.Move]:
        if not self._tb_reader:
            return None
        # Only probe with ≤5 pieces for speed
        if len(board.piece_map()) > 5:
            return None
        try:
            best_move = None
            best_dtz  = None
            for move in board.legal_moves:
                board.push(move)
                try:
                    dtz = self._tb_reader.get_dtz(board)
                    if dtz is not None:
                        if best_dtz is None or -dtz < best_dtz:
                            best_dtz  = -dtz
                            best_move = move
                except Exception:
                    pass
                finally:
                    board.pop()
            return best_move
        except Exception:
            return None

    def perft(self, depth: int) -> int:
        """Node count for testing move generation correctness."""
        if depth == 0:
            return 1
        count = 0
        for move in self.board.legal_moves:
            self.board.push(move)
            count += self.perft(depth - 1)
            self.board.pop()
        return count
