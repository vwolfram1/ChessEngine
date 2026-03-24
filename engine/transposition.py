"""
Transposition Table with Zobrist hashing.
Uses a fixed-size array with aging for replacement strategy.
"""

import chess
import random
from enum import IntEnum


class Bound(IntEnum):
    EXACT = 0
    LOWER = 1   # beta-cutoff (fail-high)
    UPPER = 2   # all-node  (fail-low)


class TTEntry:
    __slots__ = ("key", "depth", "score", "move", "bound", "age")

    def __init__(self, key=0, depth=-1, score=0, move=None, bound=Bound.EXACT, age=0):
        self.key   = key
        self.depth = depth
        self.score = score
        self.move  = move
        self.bound = bound
        self.age   = age


class TranspositionTable:
    """
    Power-of-two sized transposition table.
    Each slot holds one entry (simple replacement with aging preference).
    """

    def __init__(self, size_mb: int = 256):
        self.num_entries = (size_mb * 1024 * 1024) // 64   # ~64 bytes per logical slot
        # Round down to power of 2 for fast masking
        self.num_entries = 1 << (self.num_entries.bit_length() - 1)
        self.mask        = self.num_entries - 1
        self._table      = [TTEntry() for _ in range(self.num_entries)]
        self.age         = 0
        self.hits        = 0
        self.stores      = 0

    def probe(self, key: int, depth: int, alpha: int, beta: int, ply: int):
        """
        Returns (tt_move, score_or_None).
        score_or_None is set when we can use the TT score directly.
        """
        entry = self._table[key & self.mask]
        if entry.key != key:
            return None, None
        self.hits += 1
        tt_move = entry.move
        if entry.depth >= depth:
            score = self._score_from_tt(entry.score, ply)
            if entry.bound == Bound.EXACT:
                return tt_move, score
            if entry.bound == Bound.LOWER and score >= beta:
                return tt_move, score
            if entry.bound == Bound.UPPER and score <= alpha:
                return tt_move, score
        return tt_move, None

    def store(self, key: int, depth: int, score: int, move, bound: Bound, ply: int):
        idx   = key & self.mask
        entry = self._table[idx]
        # Replacement: prefer same key, always replace older entries, prefer deeper
        if (entry.key == key
                or entry.age != self.age
                or depth >= entry.depth - 3):
            entry.key   = key
            entry.depth = depth
            entry.score = self._score_to_tt(score, ply)
            entry.move  = move
            entry.bound = bound
            entry.age   = self.age
            self.stores += 1

    def increment_age(self):
        self.age = (self.age + 1) & 0xFF

    def clear(self):
        for i in range(self.num_entries):
            self._table[i] = TTEntry()
        self.age = 0

    # Mate scores are stored relative to the root; adjust by ply
    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        from .constants import MATE_LOWER
        if score > MATE_LOWER:
            return score + ply
        if score < -MATE_LOWER:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        from .constants import MATE_LOWER
        if score > MATE_LOWER:
            return score - ply
        if score < -MATE_LOWER:
            return score + ply
        return score


# ---------------------------------------------------------------------------
# Zobrist Hashing
# ---------------------------------------------------------------------------

random.seed(0xDEADBEEF)

_PIECE_RANDOMS = {}
for color in (chess.WHITE, chess.BLACK):
    for pt in chess.PIECE_TYPES:
        for sq in chess.SQUARES:
            _PIECE_RANDOMS[(color, pt, sq)] = random.getrandbits(64)

_SIDE_RANDOM    = random.getrandbits(64)
_CASTLE_RANDOMS = [random.getrandbits(64) for _ in range(16)]
_EP_RANDOMS     = [random.getrandbits(64) for _ in range(8)]   # one per file


def zobrist_hash(board: chess.Board) -> int:
    h = 0
    for sq, piece in board.piece_map().items():
        h ^= _PIECE_RANDOMS[(piece.color, piece.piece_type, sq)]
    if board.turn == chess.WHITE:
        h ^= _SIDE_RANDOM
    h ^= _CASTLE_RANDOMS[board.castling_rights & 0xF]
    if board.ep_square is not None:
        h ^= _EP_RANDOMS[chess.square_file(board.ep_square)]
    return h


def incremental_hash(h: int, board: chess.Board, move: chess.Move) -> int:
    """
    Approximate incremental hash update (handles most cases).
    We fall back to full recompute after the move is pushed.
    """
    return zobrist_hash(board)
