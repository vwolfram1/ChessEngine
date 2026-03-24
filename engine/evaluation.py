"""
Hybrid evaluation: Tapered piece-square tables (PeSTO) + hand-crafted bonuses.
Falls back gracefully when neural network is unavailable.
"""

import chess
from .constants import (
    MG_VALUE, EG_VALUE, MG_TABLES, EG_TABLES,
    PHASE_WEIGHTS, MAX_PHASE,
)


def _pst_square(sq: int, color: chess.Color) -> int:
    """Map chess.Square to PST index (white = rank-flipped)."""
    if color == chess.WHITE:
        rank = 7 - chess.square_rank(sq)
    else:
        rank = chess.square_rank(sq)
    return rank * 8 + chess.square_file(sq)


def _phase(board: chess.Board) -> int:
    phase = 0
    for pt in chess.PIECE_TYPES:
        count = len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
        phase += PHASE_WEIGHTS.get(pt, 0) * count
    return min(phase, MAX_PHASE)


def _tapered(mg: int, eg: int, phase: int) -> int:
    return (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE


# ---------------------------------------------------------------------------
# Hand-crafted bonuses
# ---------------------------------------------------------------------------

BISHOP_PAIR_BONUS_MG = 40
BISHOP_PAIR_BONUS_EG = 60

ROOK_OPEN_FILE_MG  = 25
ROOK_OPEN_FILE_EG  = 15
ROOK_SEMI_OPEN_MG  = 12
ROOK_SEMI_OPEN_EG  = 8

PASSED_PAWN_BONUS = [0, 10, 15, 25, 40, 60, 90, 0]  # indexed by rank (from own side)
ISOLATED_PAWN_PENALTY = 15
DOUBLED_PAWN_PENALTY  = 10

KING_PAWN_SHIELD_MG = 10   # per shielding pawn
KING_OPEN_FILE_PENALTY = 40

MOBILITY_BONUS_MG = [0, 4, 0, 0, 2, 1, 0]   # per piece type (PAWN=1..KING=6)
MOBILITY_BONUS_EG = [0, 4, 0, 0, 3, 2, 0]


def _pawn_structure(board: chess.Board, color: chess.Color, phase: int) -> int:
    score = 0
    pawns       = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    # Pre-compute file data in a single O(n) pass — avoids O(n²) per-pawn scan
    file_count: dict[int, int] = {}
    for sq in pawns:
        f = chess.square_file(sq)
        file_count[f] = file_count.get(f, 0) + 1
    files_with_pawns = set(file_count)

    # Pre-compute enemy pawn (file, adjusted_rank) pairs for passed-pawn check
    enemy_info = []
    for esq in enemy_pawns:
        ef    = chess.square_file(esq)
        erank = chess.square_rank(esq) if color == chess.BLACK else 7 - chess.square_rank(esq)
        enemy_info.append((ef, erank))

    for sq in pawns:
        f    = chess.square_file(sq)
        rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)

        # Isolated pawn
        if (f - 1) not in files_with_pawns and (f + 1) not in files_with_pawns:
            score -= ISOLATED_PAWN_PENALTY

        # Doubled pawn — O(1) via pre-computed count
        if file_count[f] > 1:
            score -= DOUBLED_PAWN_PENALTY

        # Passed pawn
        adj_files = {max(0, f - 1), f, min(7, f + 1)}
        is_passed = not any(ef in adj_files and erank > rank for ef, erank in enemy_info)
        if is_passed:
            score += _tapered(PASSED_PAWN_BONUS[rank], PASSED_PAWN_BONUS[rank] * 2, phase)

    return score


def _rook_bonuses(board: chess.Board, color: chess.Color, phase: int) -> int:
    score = 0
    own_pawn_files   = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
    enemy_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, not color)}

    for sq in board.pieces(chess.ROOK, color):
        f = chess.square_file(sq)
        if f not in own_pawn_files:
            if f not in enemy_pawn_files:
                score += _tapered(ROOK_OPEN_FILE_MG, ROOK_OPEN_FILE_EG, phase)
            else:
                score += _tapered(ROOK_SEMI_OPEN_MG, ROOK_SEMI_OPEN_EG, phase)
    return score


def _king_safety(board: chess.Board, color: chess.Color, phase: int) -> int:
    if phase < 12:  # only apply in middlegame
        return 0
    score     = 0
    king_sq   = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    # Pawn shield
    shield_rank = king_rank + (1 if color == chess.WHITE else -1)
    if 0 <= shield_rank <= 7:
        for df in (-1, 0, 1):
            sf = king_file + df
            if 0 <= sf <= 7:
                shield_sq = chess.square(sf, shield_rank)
                if board.piece_at(shield_sq) == chess.Piece(chess.PAWN, color):
                    score += KING_PAWN_SHIELD_MG

    # Penalty for open file near king
    pawns = board.pieces(chess.PAWN, color)
    for df in (0, 1, -1):
        f = king_file + df
        if 0 <= f <= 7:
            if not any(chess.square_file(s) == f for s in pawns):
                score -= KING_OPEN_FILE_PENALTY // (1 if df == 0 else 2)

    return (score * phase) // MAX_PHASE


def evaluate(board: chess.Board) -> int:
    """
    Returns evaluation in centipawns from the perspective of the side to move.
    Positive = good for side to move.
    """
    if board.is_checkmate():
        return -99_000 + board.ply()
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = _phase(board)
    mg_w = mg_b = eg_w = eg_b = 0

    # Material + PST — piece_map() only visits occupied squares (~20-32 vs 64)
    for sq, piece in board.piece_map().items():
        pt  = piece.piece_type
        idx = _pst_square(sq, piece.color)
        mg  = MG_VALUE[pt] + MG_TABLES[pt][idx]
        eg  = EG_VALUE[pt] + EG_TABLES[pt][idx]
        if piece.color == chess.WHITE:
            mg_w += mg; eg_w += eg
        else:
            mg_b += mg; eg_b += eg

    # Bishop pair
    for color, mg_ref, eg_ref in ((chess.WHITE, 'w', 'w'), (chess.BLACK, 'b', 'b')):
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            if color == chess.WHITE:
                mg_w += BISHOP_PAIR_BONUS_MG
                eg_w += BISHOP_PAIR_BONUS_EG
            else:
                mg_b += BISHOP_PAIR_BONUS_MG
                eg_b += BISHOP_PAIR_BONUS_EG

    mg_score = mg_w - mg_b
    eg_score = eg_w - eg_b
    score    = _tapered(mg_score, eg_score, phase)

    # Structural bonuses
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        score += sign * _pawn_structure(board, color, phase)
        score += sign * _rook_bonuses(board, color, phase)
        score += sign * _king_safety(board, color, phase)

    # Flip for side to move
    return score if board.turn == chess.WHITE else -score
