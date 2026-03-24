# cchess.pyx  –  NexusChess Cython bitboard engine
# Covers: board representation, move generation, PST evaluation,
#         transposition table, negamax + PVS, iterative deepening.
#
# Language level: Python 3 (Cython 3.x)
# Compile:  python setup.py build_ext --inplace

# cython: language_level=3

cimport cython
from libc.stdint  cimport uint64_t, uint32_t, int32_t, int16_t, uint8_t
from libc.stdlib  cimport malloc, free, rand, srand
from libc.string  cimport memset, memcpy
from libc.time    cimport time as c_time

import time as pytime
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Compile-time constants (DEF = C macros, zero runtime overhead)
# ─────────────────────────────────────────────────────────────────────────────

# Piece types
DEF PAWN   = 0
DEF KNIGHT = 1
DEF BISHOP = 2
DEF ROOK   = 3
DEF QUEEN  = 4
DEF KING   = 5
DEF NO_PT  = 6

# Colors
DEF WHITE = 0
DEF BLACK = 1

# Move-flag encoding (stored in bits 12-15 of the uint32 move)
DEF MF_QUIET     = 0
DEF MF_DBL_PUSH  = 1   # double pawn push
DEF MF_KS_CASTLE = 2   # king-side castle
DEF MF_QS_CASTLE = 3   # queen-side castle
DEF MF_CAPTURE   = 4   # ordinary capture
DEF MF_EP        = 5   # en-passant capture
# 6,7 unused
DEF MF_N_PROMO   = 8   # knight promotion
DEF MF_B_PROMO   = 9   # bishop promotion
DEF MF_R_PROMO   = 10  # rook promotion
DEF MF_Q_PROMO   = 11  # queen promotion
DEF MF_N_PROMO_C = 12  # knight promo + capture
DEF MF_B_PROMO_C = 13
DEF MF_R_PROMO_C = 14
DEF MF_Q_PROMO_C = 15

# Castling right bits
DEF W_KS = 1
DEF W_QS = 2
DEF B_KS = 4
DEF B_QS = 8

# Search / eval constants
DEF INF        = 100000
DEF MATE_SCORE = 99000
DEF MATE_LOWER = 98000
DEF MAX_PLY    = 128

# TT bound types
DEF BOUND_EXACT = 0
DEF BOUND_LOWER = 1   # fail-high (score >= beta)
DEF BOUND_UPPER = 2   # fail-low  (score <= alpha)

# ─────────────────────────────────────────────────────────────────────────────
# C structs
# ─────────────────────────────────────────────────────────────────────────────

cdef struct UndoInfo:
    uint64_t zobrist_before
    int      ep_before           # en-passant square before the move (-1 = none)
    uint8_t  castling_before
    int      halfmove_before
    int      captured_type       # piece type of captured piece (-1 = none)
    int      captured_sq         # square piece was removed from (EP differs from to_sq)

cdef struct CBoard:
    uint64_t pieces[2][6]        # pieces[color][piece_type]
    uint64_t occ[2]              # aggregate occupancy per color
    uint64_t all_occ             # all pieces
    int      stm                 # side to move: WHITE=0, BLACK=1
    int      ep_sq               # en-passant target square (-1 = none)
    uint8_t  castling            # W_KS | W_QS | B_KS | B_QS
    int      halfmove
    int      fullmove
    uint64_t zobrist
    UndoInfo undo[1024]          # undo stack
    int      undo_top
    uint64_t hash_history[1024]  # for repetition detection
    int      hist_top

cdef struct TTEntry:
    uint64_t key
    int32_t  score
    uint32_t move           # best move in this position (0 = none)
    int16_t  depth
    uint8_t  bound          # BOUND_EXACT / LOWER / UPPER
    uint8_t  age

# ─────────────────────────────────────────────────────────────────────────────
# Module-level precomputed tables (filled once at import)
# ─────────────────────────────────────────────────────────────────────────────

cdef uint64_t PAWN_ATTACKS[2][64]    # [color][sq] → attack bitboard
cdef uint64_t KNIGHT_ATTACKS[64]
cdef uint64_t KING_ATTACKS[64]

# Zobrist random numbers
cdef uint64_t ZOB_PIECES[2][6][64]
cdef uint64_t ZOB_SIDE                # XOR when WHITE to move
cdef uint64_t ZOB_CASTLING[16]
cdef uint64_t ZOB_EP[9]              # [file 0-7], index 8 = no EP

# PST tables: [piece_type][square], from WHITE's perspective.
# Black uses vertically mirrored square (rank flip).
# Values are (mg_value + mg_pst, eg_value + eg_pst) packed as (mg<<16 | (eg & 0xFFFF))
# We store them separately for clarity.
cdef int MG_PST[6][64]
cdef int EG_PST[6][64]
cdef int MG_VAL[6]    # piece material values (midgame)
cdef int EG_VAL[6]    # piece material values (endgame)
cdef int PHASE_WT[6]  # phase weight per piece

# ─────────────────────────────────────────────────────────────────────────────
# Table initialisation (called once at module load)
# ─────────────────────────────────────────────────────────────────────────────

cdef uint64_t _xorshift64(uint64_t* state) nogil:
    state[0] ^= state[0] << 13
    state[0] ^= state[0] >> 7
    state[0] ^= state[0] << 17
    return state[0]

cdef void _init_tables():
    cdef int sq, f, r, tsq, color
    cdef uint64_t bb, rng_state = 0xDEADF00DCAFEBABE

    # ── pawn attacks ──────────────────────────────────────────────────────────
    for sq in range(64):
        f = sq & 7; r = sq >> 3
        # White attacks: NW and NE from sq
        bb = 0
        if r < 7:
            if f > 0: bb |= (<uint64_t>1 << (sq + 7))
            if f < 7: bb |= (<uint64_t>1 << (sq + 9))
        PAWN_ATTACKS[WHITE][sq] = bb
        # Black attacks: SW and SE from sq
        bb = 0
        if r > 0:
            if f > 0: bb |= (<uint64_t>1 << (sq - 9))
            if f < 7: bb |= (<uint64_t>1 << (sq - 7))
        PAWN_ATTACKS[BLACK][sq] = bb

    # ── knight attacks ────────────────────────────────────────────────────────
    cdef int[8] KN_DR = [-17, -15, -10, -6, 6, 10, 15, 17]
    cdef int[8] KN_DF = [-1,   1,  -2,  2, -2,  2, -1,  1]
    for sq in range(64):
        f = sq & 7; r = sq >> 3
        bb = 0
        for i in range(8):
            nf = f + KN_DF[i]; nr = r + KN_DR[i] // 8
            # Use actual delta
            tsq = sq + KN_DR[i]
            if 0 <= tsq < 64:
                tf = tsq & 7
                if abs(tf - f) <= 2:   # guard file wrap
                    bb |= (<uint64_t>1 << tsq)
        KNIGHT_ATTACKS[sq] = bb

    # ── king attacks ──────────────────────────────────────────────────────────
    cdef int[8] KG_D = [-9, -8, -7, -1, 1, 7, 8, 9]
    for sq in range(64):
        f = sq & 7
        bb = 0
        for i in range(8):
            tsq = sq + KG_D[i]
            if 0 <= tsq < 64:
                tf = tsq & 7
                if abs(tf - f) <= 1:
                    bb |= (<uint64_t>1 << tsq)
        KING_ATTACKS[sq] = bb

    # ── Zobrist ──────────────────────────────────────────────────────────────
    for color in range(2):
        for pt in range(6):
            for sq in range(64):
                ZOB_PIECES[color][pt][sq] = _xorshift64(&rng_state)
    ZOB_SIDE = _xorshift64(&rng_state)
    for i in range(16):
        ZOB_CASTLING[i] = _xorshift64(&rng_state)
    for i in range(9):
        ZOB_EP[i] = _xorshift64(&rng_state)

    # ── PST tables (PeSTO values) ─────────────────────────────────────────────
    _init_pst()

# ──  PeSTO piece-square tables  ──────────────────────────────────────────────
# Stored rank-0 = rank 1 (a1 = index 0). White perspective = rank flipped for PST.
# fmt: off
_MG_PAWN = [
    0,   0,   0,   0,   0,   0,   0,   0,
   98, 134,  61,  95,  68, 126,  34, -11,
   -6,   7,  26,  31,  65,  56,  25, -20,
  -14,  13,   6,  21,  23,  12,  17, -23,
  -27,  -2,  -5,  12,  17,   6,  10, -25,
  -26,  -4,  -4, -10,   3,   3,  33, -12,
  -35,  -1, -20, -23, -15,  24,  38, -22,
    0,   0,   0,   0,   0,   0,   0,   0,
]
_EG_PAWN = [
    0,   0,   0,   0,   0,   0,   0,   0,
  178, 173, 158, 134, 147, 132, 165, 187,
   94, 100,  85,  67,  56,  53,  82,  84,
   32,  24,  13,   5,  -2,   4,  17,  17,
   13,   9,  -3,  -7,  -7,  -8,   3,  -1,
    4,   7,  -6,   1,   0,  -5,  -1,  -8,
   13,   8,   8,  10,  13,   0,   2,  -7,
    0,   0,   0,   0,   0,   0,   0,   0,
]
_MG_KNIGHT = [
 -167, -89, -34, -49,  61, -97, -15,-107,
  -73, -41,  72,  36,  23,  62,   7, -17,
  -47,  60,  37,  65,  84, 129,  73,  44,
   -9,  17,  19,  53,  37,  69,  18,  22,
  -13,   4,  16,  13,  28,  19,  21,  -8,
  -23,  -9,  12,  10,  19,  17,  25, -16,
  -29, -53, -12,  -3,  -1,  18, -14, -19,
 -105, -21, -58, -33, -17, -28, -19, -23,
]
_EG_KNIGHT = [
  -58, -38, -13, -28, -31, -27, -63, -99,
  -25,  -8, -25,  -2,  -9, -25, -24, -52,
  -24, -20,  10,   9,  -1,  -9, -19, -41,
  -17,   3,  22,  22,  22,  11,   8, -18,
  -18,  -6,  16,  25,  16,  17,   4, -18,
  -23,  -3,  -1,  15,  10,  -3, -20, -22,
  -42, -20, -10,  -5,  -2, -20, -23, -44,
  -29, -51, -23, -15, -22, -18, -50, -64,
]
_MG_BISHOP = [
  -29,   4, -82, -37, -25, -42,   7,  -8,
  -26,  16, -18, -13,  30,  59,  18, -47,
  -16,  37,  43,  40,  35,  50,  37,  -2,
   -4,   5,  19,  50,  37,  37,   7,  -2,
   -6,  13,  13,  26,  34,  12,  10,   4,
    0,  15,  15,  15,  14,  27,  18,  10,
    4,  15,  16,   0,   7,  21,  33,   1,
  -33,  -3, -14, -21, -13, -12, -39, -21,
]
_EG_BISHOP = [
  -14, -21, -11,  -8,  -7,  -9, -17, -24,
   -8,  -4,   7, -12,  -3, -13,  -4, -14,
    2,  -8,   0,  -1,  -2,   6,   0,   4,
   -3,   9,  12,   9,  14,  10,   3,   2,
   -6,   3,  13,  19,   7,  10,  -3,  -9,
  -12,  -3,   8,  10,  13,   3,  -7, -15,
  -14, -18,  -7,  -1,   4,  -9, -15, -27,
  -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]
_MG_ROOK = [
   32,  42,  32,  51,  63,   9,  31,  43,
   27,  32,  58,  62,  80,  67,  26,  44,
   -5,  19,  26,  36,  17,  45,  61,  16,
  -24, -11,   7,  26,  24,  35,  -8, -20,
  -36, -26, -12,  -1,   9,  -7,   6, -23,
  -45, -25, -16, -17,   3,   0,  -5, -33,
  -44, -16, -20,  -9,  -1,  11,  -6, -71,
  -19, -13,   1,  17,  16,   7, -37, -26,
]
_EG_ROOK = [
   13,  10,  18,  15,  12,  12,   8,   5,
   11,  13,  13,  11,  -3,   3,   8,   3,
    7,   7,   7,   5,   4,  -3,  -5,  -3,
    4,   3,  13,   1,   2,   1,  -1,   2,
    3,   5,   8,   4,  -5,  -6,  -8, -11,
   -4,   0,  -5,  -1,  -7, -12,  -8, -16,
   -6,  -6,   0,   2,  -9,  -9, -11,  -3,
   -9,   2,   3,  -1,  -5, -13,   4, -20,
]
_MG_QUEEN = [
  -28,   0,  29,  12,  59,  44,  43,  45,
  -24, -39,  -5,   1, -16,  57,  28,  54,
  -13, -17,   7,   8,  29,  56,  47,  57,
  -27, -27, -16, -16,  -1,  17,  -2,   1,
   -9, -26,  -9, -10,  -2,  -4,   3,  -3,
  -14,   2, -11,  -2,  -5,   2,  14,   5,
  -35,  -8,  11,   2,   8,  15,  -3,   1,
   -1, -18,  -9,  10, -15, -25, -31, -50,
]
_EG_QUEEN = [
   -9,  22,  22,  27,  27,  19,  10,  20,
  -17,  20,  32,  41,  58,  25,  30,   0,
  -20,   6,   9,  49,  47,  35,  19,   9,
    3,  22,  24,  45,  57,  40,  57,  36,
  -18,  28,  19,  47,  31,  34,  39,  23,
  -16, -27,  15,   6,   9,  17,  10,   5,
  -22, -23, -30, -16, -16, -23, -36, -32,
  -33, -28, -22, -43,  -5, -32, -20, -41,
]
_MG_KING = [
  -65,  23,  16, -15, -56, -34,   2,  13,
   29,  -1, -20,  -7,  -8,  -4, -38, -29,
   -9,  24,   2, -16, -20,   6,  22, -22,
  -17, -20, -12, -27, -30, -25, -14, -36,
  -49,  -1, -27, -39, -46, -44, -33, -51,
  -14, -14, -22, -46, -44, -30, -15, -27,
    1,   7,  -8, -64, -43, -16,   9,   8,
  -15,  36,  12, -54,   8, -28,  24,  14,
]
_EG_KING = [
  -74, -35, -18, -18, -11,  15,   4, -17,
  -12,  17,  14,  17,  17,  38,  23,  11,
   10,  17,  23,  15,  20,  45,  44,  13,
   -8,  22,  24,  27,  26,  33,  26,   3,
  -18,  -4,  21,  24,  27,  23,   9, -11,
  -19,  -3,  11,  21,  23,  16,   7,  -9,
  -27, -11,   4,  13,  14,   4,  -5, -17,
  -53, -34, -21, -11, -28, -14, -24, -43,
]
# fmt: on

_ALL_PST = [
    (_MG_PAWN,   _EG_PAWN),
    (_MG_KNIGHT, _EG_KNIGHT),
    (_MG_BISHOP, _EG_BISHOP),
    (_MG_ROOK,   _EG_ROOK),
    (_MG_QUEEN,  _EG_QUEEN),
    (_MG_KING,   _EG_KING),
]

cdef void _init_pst():
    cdef int pt, sq
    _mg_vals = [82,  337, 365, 477, 1025, 0]
    _eg_vals = [94,  281, 297, 512,  936, 0]
    _ph_wts  = [ 0,    1,   1,   2,    4, 0]

    for pt in range(6):
        MG_VAL[pt] = _mg_vals[pt]
        EG_VAL[pt] = _eg_vals[pt]
        PHASE_WT[pt] = _ph_wts[pt]
        mg_tbl, eg_tbl = _ALL_PST[pt]
        for sq in range(64):
            # PeSTO tables are stored with a8=0 in the Python list.
            # Our squares: a1=0..h8=63 → rank = sq>>3, file = sq&7
            # White PST: flip rank so rank-7 (rank 8) maps to row 0 of table
            rank = sq >> 3
            file_ = sq & 7
            white_idx = (7 - rank) * 8 + file_
            MG_PST[pt][sq] = mg_tbl[white_idx]
            EG_PST[pt][sq] = eg_tbl[white_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Inline bit helpers
# ─────────────────────────────────────────────────────────────────────────────

cdef inline int lsb(uint64_t bb) nogil:
    """Index of least-significant set bit.  UB if bb==0."""
    cdef int n = 0
    if (bb & <uint64_t>0xFFFFFFFF) == 0: n += 32; bb >>= 32
    if (bb & <uint64_t>0x0000FFFF) == 0: n += 16; bb >>= 16
    if (bb & <uint64_t>0x000000FF) == 0: n +=  8; bb >>=  8
    if (bb & <uint64_t>0x0000000F) == 0: n +=  4; bb >>=  4
    if (bb & <uint64_t>0x00000003) == 0: n +=  2; bb >>=  2
    if (bb & <uint64_t>0x00000001) == 0: n +=  1
    return n

cdef inline int popcount(uint64_t bb) nogil:
    bb -= (bb >> 1) & <uint64_t>0x5555555555555555
    bb  = (bb & <uint64_t>0x3333333333333333) + ((bb >> 2) & <uint64_t>0x3333333333333333)
    bb  = (bb + (bb >> 4)) & <uint64_t>0x0F0F0F0F0F0F0F0F
    return <int>((bb * <uint64_t>0x0101010101010101) >> 56)

cdef inline uint64_t bit(int sq) nogil:
    return <uint64_t>1 << sq


# ─────────────────────────────────────────────────────────────────────────────
# Sliding piece attack generators (classical ray scanning)
# ─────────────────────────────────────────────────────────────────────────────

cdef inline uint64_t rook_attacks(int sq, uint64_t occ) nogil:
    cdef uint64_t attacks = 0, b
    cdef int s
    cdef int rank_start = (sq >> 3) << 3     # first square on rank
    cdef int rank_end   = rank_start + 8     # one past last square on rank

    # North (+8)
    s = sq + 8
    while s < 64:
        attacks |= bit(s)
        if occ & bit(s): break
        s += 8
    # South (-8)
    s = sq - 8
    while s >= 0:
        attacks |= bit(s)
        if occ & bit(s): break
        s -= 8
    # East (+1, stay on same rank)
    s = sq + 1
    while s < rank_end:
        attacks |= bit(s)
        if occ & bit(s): break
        s += 1
    # West (-1, stay on same rank)
    s = sq - 1
    while s >= rank_start:
        attacks |= bit(s)
        if occ & bit(s): break
        s -= 1
    return attacks

cdef inline uint64_t bishop_attacks(int sq, uint64_t occ) nogil:
    cdef uint64_t attacks = 0
    cdef int s, f = sq & 7, r = sq >> 3
    # NE (+9)
    s = sq + 9
    while s < 64 and (s & 7) > (s - 9 & 7):
        attacks |= bit(s)
        if occ & bit(s): break
        s += 9
    # NW (+7)
    s = sq + 7
    while s < 64 and (s & 7) < (s - 7 & 7):
        attacks |= bit(s)
        if occ & bit(s): break
        s += 7
    # SE (-7)
    s = sq - 7
    while s >= 0 and (s & 7) > (s + 7 & 7):
        attacks |= bit(s)
        if occ & bit(s): break
        s -= 7
    # SW (-9)
    s = sq - 9
    while s >= 0 and (s & 7) < (s + 9 & 7):
        attacks |= bit(s)
        if occ & bit(s): break
        s -= 9
    return attacks

cdef inline uint64_t queen_attacks(int sq, uint64_t occ) nogil:
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)


# ─────────────────────────────────────────────────────────────────────────────
# Square-attack detection
# ─────────────────────────────────────────────────────────────────────────────

cdef inline bint is_sq_attacked(CBoard* b, int sq, int by_color) nogil:
    """Is square `sq` attacked by any piece of `by_color`?"""
    cdef uint64_t occ = b.all_occ
    cdef int opp = by_color

    # Pawn attacks
    if PAWN_ATTACKS[by_color ^ 1][sq] & b.pieces[opp][PAWN]:
        return True
    # Knight
    if KNIGHT_ATTACKS[sq] & b.pieces[opp][KNIGHT]:
        return True
    # King
    if KING_ATTACKS[sq] & b.pieces[opp][KING]:
        return True
    # Bishops / Queens (diagonal)
    if bishop_attacks(sq, occ) & (b.pieces[opp][BISHOP] | b.pieces[opp][QUEEN]):
        return True
    # Rooks / Queens (orthogonal)
    if rook_attacks(sq, occ) & (b.pieces[opp][ROOK] | b.pieces[opp][QUEEN]):
        return True
    return False

cdef inline bint is_in_check(CBoard* b) nogil:
    cdef uint64_t king_bb = b.pieces[b.stm][KING]
    if not king_bb: return False
    return is_sq_attacked(b, lsb(king_bb), b.stm ^ 1)


# ─────────────────────────────────────────────────────────────────────────────
# FEN parser
# ─────────────────────────────────────────────────────────────────────────────

cdef void board_clear(CBoard* b) noexcept nogil:
    memset(b, 0, sizeof(CBoard))
    b.ep_sq    = -1
    b.halfmove =  0
    b.fullmove =  1

cdef int _pt_from_char(char c) nogil:
    if c == 112 or c == 80:  return PAWN    # 'p'/'P'
    if c == 110 or c == 78:  return KNIGHT  # 'n'/'N'
    if c == 98  or c == 66:  return BISHOP  # 'b'/'B'
    if c == 114 or c == 82:  return ROOK    # 'r'/'R'
    if c == 113 or c == 81:  return QUEEN   # 'q'/'Q'
    if c == 107 or c == 75:  return KING    # 'k'/'K'
    return -1

cdef bint parse_fen(CBoard* b, str fen_str):
    """Parse a FEN string into board b.  Returns True on success."""
    board_clear(b)
    cdef bytes fen_bytes = fen_str.encode('ascii')
    cdef const char* fen = fen_bytes
    cdef int i = 0, sq = 56, pt, color, c, ep_file, ep_rank

    # Piece placement
    while fen[i] != 32 and fen[i] != 0:   # 32 = ' '
        c = fen[i]
        if 49 <= c <= 56:          # '1'..'8'
            sq += c - 48           # c - '0'
        elif c == 47:              # '/'
            sq -= 16
        else:
            color = WHITE if (65 <= c <= 90) else BLACK   # uppercase A-Z
            pt = _pt_from_char(<char>c)
            if pt >= 0:
                b.pieces[color][pt] |= bit(sq)
                sq += 1
        i += 1
    i += 1  # skip space

    # Side to move
    b.stm = WHITE if fen[i] == 119 else BLACK   # 119 = 'w'
    i += 2  # skip char + space

    # Castling
    b.castling = 0
    while fen[i] != 32 and fen[i] != 0:
        c = fen[i]
        if   c == 75:  b.castling |= W_KS   # 'K'
        elif c == 81:  b.castling |= W_QS   # 'Q'
        elif c == 107: b.castling |= B_KS   # 'k'
        elif c == 113: b.castling |= B_QS   # 'q'
        i += 1
    i += 1  # skip space

    # En passant
    b.ep_sq = -1
    if fen[i] != 45:               # '-'
        ep_file = fen[i] - 97      # 'a'
        ep_rank = fen[i+1] - 49   # '1'
        b.ep_sq = ep_rank * 8 + ep_file
        i += 2
    else:
        i += 1
    i += 1  # skip space

    # Halfmove clock
    b.halfmove = 0
    while fen[i] != 32 and fen[i] != 0:
        b.halfmove = b.halfmove * 10 + (fen[i] - 48)   # '0'
        i += 1
    i += 1

    # Fullmove number
    b.fullmove = 0
    while fen[i] != 32 and fen[i] != 0:
        b.fullmove = b.fullmove * 10 + (fen[i] - 48)
        i += 1

    # Rebuild aggregate bitboards
    for c in range(2):
        b.occ[c] = 0
        for pt in range(6):
            b.occ[c] |= b.pieces[c][pt]
    b.all_occ = b.occ[WHITE] | b.occ[BLACK]

    # Compute Zobrist hash from scratch
    b.zobrist = 0
    for c in range(2):
        for pt in range(6):
            bb = b.pieces[c][pt]
            while bb:
                sq = lsb(bb)
                b.zobrist ^= ZOB_PIECES[c][pt][sq]
                bb &= bb - 1
    if b.stm == WHITE:
        b.zobrist ^= ZOB_SIDE
    b.zobrist ^= ZOB_CASTLING[b.castling & 0xF]
    if b.ep_sq >= 0:
        b.zobrist ^= ZOB_EP[b.ep_sq & 7]

    b.undo_top = 0
    b.hist_top = 0
    return True

cdef str board_to_fen(CBoard* b):
    """Convert board to FEN string."""
    cdef int sq, empty, pt, c
    rows = []
    for rank in range(7, -1, -1):
        empty = 0
        row = ""
        for file_ in range(8):
            sq = rank * 8 + file_
            found = False
            for c in range(2):
                for pt in range(6):
                    if b.pieces[c][pt] & bit(sq):
                        if empty:
                            row += str(empty)
                            empty = 0
                        ch = "pnbrqk"[pt] if c == BLACK else "PNBRQK"[pt]
                        row += ch
                        found = True
                        break
                if found: break
            if not found:
                empty += 1
        if empty: row += str(empty)
        rows.append(row)
    fen = "/".join(rows)
    fen += " " + ("w" if b.stm == WHITE else "b")
    castling = ""
    if b.castling & W_KS: castling += "K"
    if b.castling & W_QS: castling += "Q"
    if b.castling & B_KS: castling += "k"
    if b.castling & B_QS: castling += "q"
    fen += " " + (castling if castling else "-")
    if b.ep_sq >= 0:
        fen += " " + "abcdefgh"[b.ep_sq & 7] + str((b.ep_sq >> 3) + 1)
    else:
        fen += " -"
    fen += f" {b.halfmove} {b.fullmove}"
    return fen


# ─────────────────────────────────────────────────────────────────────────────
# Move encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

cdef inline uint32_t make_move_code(int frm, int to, int flags) nogil:
    return <uint32_t>(frm | (to << 6) | (flags << 12))

cdef inline int move_from(uint32_t mv)  nogil: return  mv        & 0x3F
cdef inline int move_to(uint32_t mv)    nogil: return (mv >>  6) & 0x3F
cdef inline int move_flags(uint32_t mv) nogil: return (mv >> 12) & 0xF

cdef str move_to_uci(uint32_t mv):
    if mv == 0: return "0000"
    cdef int frm   = move_from(mv)
    cdef int to    = move_to(mv)
    cdef int flags = move_flags(mv)
    s = "abcdefgh"[frm & 7] + str((frm >> 3) + 1)
    s += "abcdefgh"[to & 7] + str((to >> 3) + 1)
    if flags >= MF_N_PROMO:
        s += "nbrq"[(flags & 3)]
    return s

cdef uint32_t uci_to_move_on_board(CBoard* b, str uci_str):
    """Convert UCI string to the move code matching the current position."""
    if len(uci_str) < 4: return 0
    cdef int frm = (ord(uci_str[0]) - ord('a')) + (int(uci_str[1]) - 1) * 8
    cdef int to  = (ord(uci_str[2]) - ord('a')) + (int(uci_str[3]) - 1) * 8
    cdef uint32_t moves[256]
    cdef int n = gen_legal_moves(b, moves)
    for i in range(n):
        if move_from(moves[i]) == frm and move_to(moves[i]) == to:
            # For promotions, check the piece
            if len(uci_str) == 5:
                promo_ch = uci_str[4]
                flags = move_flags(moves[i])
                if flags < MF_N_PROMO: continue
                promo_pt = flags & 3  # 0=N,1=B,2=R,3=Q
                if promo_ch == 'n' and promo_pt == 0: return moves[i]
                if promo_ch == 'b' and promo_pt == 1: return moves[i]
                if promo_ch == 'r' and promo_pt == 2: return moves[i]
                if promo_ch == 'q' and promo_pt == 3: return moves[i]
            else:
                return moves[i]
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# make_move / unmake_move
# ─────────────────────────────────────────────────────────────────────────────

# Castling-rights update table: indexed by square, contains mask to AND with castling
# Any move from/to these squares strips the corresponding castling right.
cdef uint8_t CASTLE_MASK[64]

cdef void _init_castle_mask():
    for i in range(64):
        CASTLE_MASK[i] = 0xF  # full rights preserved by default
    CASTLE_MASK[0]  = ~W_QS & 0xF   # a1 – white queen-side rook
    CASTLE_MASK[7]  = ~W_KS & 0xF   # h1 – white king-side rook
    CASTLE_MASK[4]  = ~(W_KS | W_QS) & 0xF   # e1 – white king
    CASTLE_MASK[56] = ~B_QS & 0xF   # a8 – black queen-side rook
    CASTLE_MASK[63] = ~B_KS & 0xF   # h8 – black king-side rook
    CASTLE_MASK[60] = ~(B_KS | B_QS) & 0xF   # e8 – black king

cdef void do_make_move(CBoard* b, uint32_t mv) noexcept nogil:
    """Apply move to board, updating all state including Zobrist."""
    cdef int frm   = move_from(mv)
    cdef int to    = move_to(mv)
    cdef int flags = move_flags(mv)
    cdef int cap_sq, promo_pt, rook_frm, rook_to
    cdef int us    = b.stm
    cdef int them  = us ^ 1
    cdef UndoInfo* u = &b.undo[b.undo_top]

    # Save undo info
    u.zobrist_before  = b.zobrist
    u.ep_before       = b.ep_sq
    u.castling_before = b.castling
    u.halfmove_before = b.halfmove
    u.captured_type   = -1
    u.captured_sq     = -1
    b.undo_top += 1

    # Save hash in history for repetition detection
    b.hash_history[b.hist_top] = b.zobrist
    b.hist_top += 1

    # Undo previous zobrist contributions we'll change
    b.zobrist ^= ZOB_SIDE  # flip side (XOR is symmetric)
    b.zobrist ^= ZOB_CASTLING[b.castling & 0xF]
    if b.ep_sq >= 0:
        b.zobrist ^= ZOB_EP[b.ep_sq & 7]

    # Identify moving piece
    cdef int moving_pt = -1
    for pt in range(6):
        if b.pieces[us][pt] & bit(frm):
            moving_pt = pt
            break

    # Handle capture
    if flags == MF_CAPTURE or flags >= MF_N_PROMO_C:
        for pt in range(6):
            if b.pieces[them][pt] & bit(to):
                u.captured_type = pt
                u.captured_sq   = to
                b.pieces[them][pt] &= ~bit(to)
                b.zobrist ^= ZOB_PIECES[them][pt][to]
                break
    elif flags == MF_EP:
        # Captured pawn is on the same rank as moving pawn, same file as destination
        cap_sq = to - 8 if us == WHITE else to + 8
        u.captured_type = PAWN
        u.captured_sq   = cap_sq
        b.pieces[them][PAWN] &= ~bit(cap_sq)
        b.zobrist ^= ZOB_PIECES[them][PAWN][cap_sq]

    # Move piece from frm to to
    b.pieces[us][moving_pt] &= ~bit(frm)
    b.zobrist ^= ZOB_PIECES[us][moving_pt][frm]

    if flags >= MF_N_PROMO:
        # Promotion: place promoted piece
        promo_pt = KNIGHT + (flags & 3)   # 0=N,1=B,2=R,3=Q → KNIGHT..QUEEN
        b.pieces[us][promo_pt] |= bit(to)
        b.zobrist ^= ZOB_PIECES[us][promo_pt][to]
    else:
        b.pieces[us][moving_pt] |= bit(to)
        b.zobrist ^= ZOB_PIECES[us][moving_pt][to]

    # Castling: move rook too
    if flags == MF_KS_CASTLE:
        if us == WHITE: rook_frm = 7;  rook_to = 5
        else:           rook_frm = 63; rook_to = 61
        b.pieces[us][ROOK] &= ~bit(rook_frm)
        b.pieces[us][ROOK] |=  bit(rook_to)
        b.zobrist ^= ZOB_PIECES[us][ROOK][rook_frm]
        b.zobrist ^= ZOB_PIECES[us][ROOK][rook_to]
    elif flags == MF_QS_CASTLE:
        if us == WHITE: rook_frm = 0;  rook_to = 3
        else:           rook_frm = 56; rook_to = 59
        b.pieces[us][ROOK] &= ~bit(rook_frm)
        b.pieces[us][ROOK] |=  bit(rook_to)
        b.zobrist ^= ZOB_PIECES[us][ROOK][rook_frm]
        b.zobrist ^= ZOB_PIECES[us][ROOK][rook_to]

    # Update en-passant square
    b.ep_sq = -1
    if flags == MF_DBL_PUSH:
        b.ep_sq = (frm + to) >> 1   # middle square

    # Update castling rights
    b.castling &= CASTLE_MASK[frm] & CASTLE_MASK[to]

    # Halfmove clock
    if moving_pt == PAWN or flags == MF_CAPTURE or flags >= MF_N_PROMO_C or flags == MF_EP:
        b.halfmove = 0
    else:
        b.halfmove += 1

    # Fullmove
    if us == BLACK:
        b.fullmove += 1

    # Update aggregate occupancies
    for c in range(2):
        b.occ[c] = 0
        for pt in range(6):
            b.occ[c] |= b.pieces[c][pt]
    b.all_occ = b.occ[WHITE] | b.occ[BLACK]

    # Finish Zobrist
    b.zobrist ^= ZOB_CASTLING[b.castling & 0xF]
    if b.ep_sq >= 0:
        b.zobrist ^= ZOB_EP[b.ep_sq & 7]
    # ZOB_SIDE already applied above (we flipped at start)

    b.stm = them

cdef void do_unmake_move(CBoard* b, uint32_t mv) noexcept nogil:
    """Undo the last move."""
    b.undo_top -= 1
    b.hist_top -= 1
    cdef UndoInfo* u = &b.undo[b.undo_top]
    cdef int frm   = move_from(mv)
    cdef int to    = move_to(mv)
    cdef int flags = move_flags(mv)
    cdef int them  = b.stm      # the side that just moved is now opponent
    cdef int us    = them ^ 1

    b.stm      = us
    b.ep_sq    = u.ep_before
    b.castling = u.castling_before
    b.halfmove = u.halfmove_before
    b.zobrist  = u.zobrist_before
    if us == BLACK:
        b.fullmove -= 1

    # Move piece back from to → frm
    cdef int moved_pt = -1
    cdef int promo_pt
    if flags >= MF_N_PROMO:
        # Piece at 'to' is the promoted piece; restore as pawn at 'frm'
        promo_pt = KNIGHT + (flags & 3)
        b.pieces[us][promo_pt] &= ~bit(to)
        b.pieces[us][PAWN]     |=  bit(frm)
    else:
        for pt in range(6):
            if b.pieces[us][pt] & bit(to):
                moved_pt = pt
                break
        b.pieces[us][moved_pt] &= ~bit(to)
        b.pieces[us][moved_pt] |=  bit(frm)

    # Restore captured piece
    if u.captured_type >= 0:
        b.pieces[them][u.captured_type] |= bit(u.captured_sq)

    # Undo castling rook move
    if flags == MF_KS_CASTLE:
        if us == WHITE: b.pieces[us][ROOK] &= ~bit(5);  b.pieces[us][ROOK] |= bit(7)
        else:           b.pieces[us][ROOK] &= ~bit(61); b.pieces[us][ROOK] |= bit(63)
    elif flags == MF_QS_CASTLE:
        if us == WHITE: b.pieces[us][ROOK] &= ~bit(3);  b.pieces[us][ROOK] |= bit(0)
        else:           b.pieces[us][ROOK] &= ~bit(59); b.pieces[us][ROOK] |= bit(56)

    # Rebuild aggregates
    for c in range(2):
        b.occ[c] = 0
        for pt in range(6):
            b.occ[c] |= b.pieces[c][pt]
    b.all_occ = b.occ[WHITE] | b.occ[BLACK]


# ─────────────────────────────────────────────────────────────────────────────
# Move generation
# ─────────────────────────────────────────────────────────────────────────────

cdef inline void _add(uint32_t* moves, int* n, int frm, int to, int flags) noexcept nogil:
    moves[n[0]] = make_move_code(frm, to, flags)
    n[0] += 1

cdef inline void _add_promotions(uint32_t* moves, int* n, int frm, int to, bint is_cap) noexcept nogil:
    cdef int base = MF_N_PROMO_C if is_cap else MF_N_PROMO
    _add(moves, n, frm, to, base + 3)   # queen first (most likely best)
    _add(moves, n, frm, to, base + 2)   # rook
    _add(moves, n, frm, to, base + 0)   # knight
    _add(moves, n, frm, to, base + 1)   # bishop

cdef int gen_pseudo_legal(CBoard* b, uint32_t* moves) nogil:
    """Generate pseudo-legal moves (may leave king in check). Returns count."""
    cdef int n = 0
    cdef int us = b.stm, them = us ^ 1
    cdef uint64_t our = b.occ[us], their = b.occ[them]
    cdef uint64_t occ = b.all_occ
    cdef uint64_t empty = ~occ
    cdef uint64_t bb, targets, att
    cdef int sq, tsq, frm

    # ── Pawns ────────────────────────────────────────────────────────────────
    bb = b.pieces[us][PAWN]
    while bb:
        sq = lsb(bb); bb &= bb - 1
        if us == WHITE:
            # Single push
            tsq = sq + 8
            if tsq < 64 and not (occ & bit(tsq)):
                if sq >> 3 == 6:  # rank 7 → promotion
                    _add_promotions(moves, &n, sq, tsq, False)
                else:
                    _add(moves, &n, sq, tsq, MF_QUIET)
                    # Double push from rank 2
                    if sq >> 3 == 1:
                        tsq2 = sq + 16
                        if not (occ & bit(tsq2)):
                            _add(moves, &n, sq, tsq2, MF_DBL_PUSH)
            # Captures
            att = PAWN_ATTACKS[WHITE][sq] & their
            while att:
                tsq = lsb(att); att &= att - 1
                if sq >> 3 == 6:
                    _add_promotions(moves, &n, sq, tsq, True)
                else:
                    _add(moves, &n, sq, tsq, MF_CAPTURE)
            # En passant
            if b.ep_sq >= 0 and PAWN_ATTACKS[WHITE][sq] & bit(b.ep_sq):
                _add(moves, &n, sq, b.ep_sq, MF_EP)
        else:  # BLACK
            tsq = sq - 8
            if tsq >= 0 and not (occ & bit(tsq)):
                if sq >> 3 == 1:  # rank 2 → promotion
                    _add_promotions(moves, &n, sq, tsq, False)
                else:
                    _add(moves, &n, sq, tsq, MF_QUIET)
                    if sq >> 3 == 6:
                        tsq2 = sq - 16
                        if not (occ & bit(tsq2)):
                            _add(moves, &n, sq, tsq2, MF_DBL_PUSH)
            att = PAWN_ATTACKS[BLACK][sq] & their
            while att:
                tsq = lsb(att); att &= att - 1
                if sq >> 3 == 1:
                    _add_promotions(moves, &n, sq, tsq, True)
                else:
                    _add(moves, &n, sq, tsq, MF_CAPTURE)
            if b.ep_sq >= 0 and PAWN_ATTACKS[BLACK][sq] & bit(b.ep_sq):
                _add(moves, &n, sq, b.ep_sq, MF_EP)

    # ── Knights ──────────────────────────────────────────────────────────────
    bb = b.pieces[us][KNIGHT]
    while bb:
        sq = lsb(bb); bb &= bb - 1
        targets = KNIGHT_ATTACKS[sq] & ~our
        while targets:
            tsq = lsb(targets); targets &= targets - 1
            _add(moves, &n, sq, tsq, MF_CAPTURE if (bit(tsq) & their) else MF_QUIET)

    # ── Bishops ──────────────────────────────────────────────────────────────
    bb = b.pieces[us][BISHOP]
    while bb:
        sq = lsb(bb); bb &= bb - 1
        targets = bishop_attacks(sq, occ) & ~our
        while targets:
            tsq = lsb(targets); targets &= targets - 1
            _add(moves, &n, sq, tsq, MF_CAPTURE if (bit(tsq) & their) else MF_QUIET)

    # ── Rooks ────────────────────────────────────────────────────────────────
    bb = b.pieces[us][ROOK]
    while bb:
        sq = lsb(bb); bb &= bb - 1
        targets = rook_attacks(sq, occ) & ~our
        while targets:
            tsq = lsb(targets); targets &= targets - 1
            _add(moves, &n, sq, tsq, MF_CAPTURE if (bit(tsq) & their) else MF_QUIET)

    # ── Queens ───────────────────────────────────────────────────────────────
    bb = b.pieces[us][QUEEN]
    while bb:
        sq = lsb(bb); bb &= bb - 1
        targets = queen_attacks(sq, occ) & ~our
        while targets:
            tsq = lsb(targets); targets &= targets - 1
            _add(moves, &n, sq, tsq, MF_CAPTURE if (bit(tsq) & their) else MF_QUIET)

    # ── King (normal moves) ──────────────────────────────────────────────────
    cdef uint64_t king_bb = b.pieces[us][KING]
    if king_bb:
        sq = lsb(king_bb)
        targets = KING_ATTACKS[sq] & ~our
        while targets:
            tsq = lsb(targets); targets &= targets - 1
            _add(moves, &n, sq, tsq, MF_CAPTURE if (bit(tsq) & their) else MF_QUIET)

        # Castling (pseudo-legal; legality checked separately)
        if us == WHITE:
            if (b.castling & W_KS) and not (occ & 0x60) and not (b.pieces[them][KING]):
                # squares f1 (5) and g1 (6) must be empty (already checked via 0x60)
                _add(moves, &n, 4, 6, MF_KS_CASTLE)
            if (b.castling & W_QS) and not (occ & 0xE):
                _add(moves, &n, 4, 2, MF_QS_CASTLE)
        else:
            if (b.castling & B_KS) and not (occ & (<uint64_t>0x60 << 56)):
                _add(moves, &n, 60, 62, MF_KS_CASTLE)
            if (b.castling & B_QS) and not (occ & (<uint64_t>0xE << 56)):
                _add(moves, &n, 60, 58, MF_QS_CASTLE)

    return n

cdef int gen_legal_moves(CBoard* b, uint32_t* moves) nogil:
    """Generate fully legal moves.  Returns count."""
    cdef uint32_t pseudo[256]
    cdef int count = gen_pseudo_legal(b, pseudo)
    cdef int legal_n = 0
    cdef int i, frm, to, flags
    cdef int them = b.stm ^ 1
    cdef uint64_t us_king
    cdef bint illegal

    for i in range(count):
        frm   = move_from(pseudo[i])
        to    = move_to(pseudo[i])
        flags = move_flags(pseudo[i])

        # For castling, verify king doesn't start in or pass through check
        if flags == MF_KS_CASTLE:
            if is_sq_attacked(b, frm, them): continue      # e1/e8 in check
            if is_sq_attacked(b, frm + 1, them): continue  # f1/f8 attacked
            if is_sq_attacked(b, to, them): continue        # g1/g8 attacked
        elif flags == MF_QS_CASTLE:
            if is_sq_attacked(b, frm, them): continue
            if is_sq_attacked(b, frm - 1, them): continue   # d1/d8 attacked
            if is_sq_attacked(b, to, them): continue         # c1/c8 attacked

        do_make_move(b, pseudo[i])
        # After make: b.stm has flipped to opponent.  Check if the original
        # mover's king (b.stm^1) is attacked by the new b.stm (opponent).
        us_king = b.pieces[b.stm ^ 1][KING]
        illegal = False
        if us_king:
            illegal = is_sq_attacked(b, lsb(us_king), b.stm)
        do_unmake_move(b, pseudo[i])

        if not illegal:
            moves[legal_n] = pseudo[i]
            legal_n += 1

    return legal_n

cdef int gen_legal_captures(CBoard* b, uint32_t* moves) nogil:
    """Generate only legal captures (for qsearch)."""
    cdef uint32_t all_moves[256]
    cdef int total = gen_legal_moves(b, all_moves)
    cdef int n = 0, i, flags
    for i in range(total):
        flags = move_flags(all_moves[i])
        if flags == MF_CAPTURE or flags == MF_EP or flags >= MF_N_PROMO_C:
            moves[n] = all_moves[i]
            n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (tapered PST + basic structural bonuses)
# ─────────────────────────────────────────────────────────────────────────────

cdef int c_evaluate(CBoard* b) nogil:
    """Returns centipawns from the perspective of the side to move."""
    cdef int mg_w = 0, eg_w = 0, mg_b = 0, eg_b = 0
    cdef int phase = 0
    cdef uint64_t bb
    cdef int sq, pst_sq

    for pt in range(6):
        # White pieces
        bb = b.pieces[WHITE][pt]
        while bb:
            sq = lsb(bb); bb &= bb - 1
            pst_sq = sq  # white uses direct index (rank already correct)
            mg_w += MG_VAL[pt] + MG_PST[pt][pst_sq]
            eg_w += EG_VAL[pt] + EG_PST[pt][pst_sq]
            phase += PHASE_WT[pt]
        # Black pieces (mirror rank)
        bb = b.pieces[BLACK][pt]
        while bb:
            sq = lsb(bb); bb &= bb - 1
            pst_sq = ((7 - (sq >> 3)) << 3) | (sq & 7)   # rank-flip
            mg_b += MG_VAL[pt] + MG_PST[pt][pst_sq]
            eg_b += EG_VAL[pt] + EG_PST[pt][pst_sq]
            phase += PHASE_WT[pt]

    if phase > 24: phase = 24
    cdef int mg_score = mg_w - mg_b
    cdef int eg_score = eg_w - eg_b
    cdef int score = (mg_score * phase + eg_score * (24 - phase)) // 24

    # Bishop pair bonus
    if popcount(b.pieces[WHITE][BISHOP]) >= 2: score += 40 * phase // 24 + 60 * (24 - phase) // 24
    if popcount(b.pieces[BLACK][BISHOP]) >= 2: score -= 40 * phase // 24 + 60 * (24 - phase) // 24

    return score if b.stm == WHITE else -score


# ─────────────────────────────────────────────────────────────────────────────
# Transposition Table
# ─────────────────────────────────────────────────────────────────────────────

cdef class TTable:
    cdef TTEntry* _data
    cdef int      _size      # number of entries (power of 2)
    cdef int      _mask
    cdef uint8_t  _age

    def __cinit__(self, int size_mb=256):
        cdef int n = (size_mb * 1024 * 1024) // sizeof(TTEntry)
        # Round down to power of 2
        cdef int p = 1
        while p * 2 <= n: p *= 2
        self._size = p
        self._mask = p - 1
        self._data = <TTEntry*>malloc(p * sizeof(TTEntry))
        memset(self._data, 0, p * sizeof(TTEntry))
        self._age  = 0

    def __dealloc__(self):
        if self._data:
            free(self._data)

    cdef void store(self, uint64_t key, int depth, int score, uint32_t mv,
                    int bound, int ply) nogil:
        cdef int idx = <int>(key & self._mask)
        cdef TTEntry* e = &self._data[idx]
        # Mate score adjustment
        if score >  MATE_LOWER: score += ply
        if score < -MATE_LOWER: score -= ply
        if e.key != key or e.age != self._age or depth >= e.depth - 3:
            e.key   = key
            e.depth = <int16_t>depth
            e.score = <int32_t>score
            e.move  = mv
            e.bound = <uint8_t>bound
            e.age   = self._age

    cdef void probe(self, uint64_t key, int depth, int alpha, int beta, int ply,
                    int* out_score, uint32_t* out_move) nogil:
        """Sets out_score to INF+1 if no usable entry (sentinel for 'miss')."""
        cdef int idx = <int>(key & self._mask)
        cdef TTEntry* e = &self._data[idx]
        out_move[0] = 0
        if e.key != key:
            out_score[0] = INF + 1
            return
        out_move[0] = e.move
        if e.depth < depth:
            out_score[0] = INF + 1
            return
        cdef int s = e.score
        if s >  MATE_LOWER: s -= ply
        if s < -MATE_LOWER: s += ply
        if e.bound == BOUND_EXACT:
            out_score[0] = s
        elif e.bound == BOUND_LOWER and s >= beta:
            out_score[0] = s
        elif e.bound == BOUND_UPPER and s <= alpha:
            out_score[0] = s
        else:
            out_score[0] = INF + 1

    cdef void increment_age(self) nogil:
        self._age = (self._age + 1) & 0xFF

    cdef void clear(self) nogil:
        memset(self._data, 0, self._size * sizeof(TTEntry))
        self._age = 0


# ─────────────────────────────────────────────────────────────────────────────
# Move ordering (for search)
# ─────────────────────────────────────────────────────────────────────────────

# MVV-LVA scores indexed by piece type
cdef int MVV_LVA[6][6]   # [victim][attacker]

cdef void _init_mvvlva():
    cdef int victim_val[6]
    victim_val[PAWN]   = 100
    victim_val[KNIGHT] = 300
    victim_val[BISHOP] = 300
    victim_val[ROOK]   = 500
    victim_val[QUEEN]  = 900
    victim_val[KING]   = 10000
    for v in range(6):
        for a in range(6):
            MVV_LVA[v][a] = victim_val[v] * 10 - victim_val[a]

cdef int _score_move(CBoard* b, uint32_t mv, uint32_t tt_move,
                      int killers0, int killers1,
                      int* history, int ply) nogil:
    """Score a move for ordering (higher = search first)."""
    cdef int flags = move_flags(mv)
    cdef int frm   = move_from(mv)
    cdef int to    = move_to(mv)
    cdef int vict_pt, att_pt, pt

    if mv == tt_move: return 10000000

    if flags == MF_CAPTURE or flags >= MF_N_PROMO_C or flags == MF_EP:
        # Find victim and attacker piece types
        vict_pt = -1; att_pt = -1
        for pt in range(6):
            if b.pieces[b.stm ^ 1][pt] & bit(to): vict_pt = pt; break
        if flags == MF_EP: vict_pt = PAWN
        for pt in range(6):
            if b.pieces[b.stm][pt] & bit(frm): att_pt = pt; break
        if vict_pt >= 0 and att_pt >= 0:
            return 1000000 + MVV_LVA[vict_pt][att_pt]
        return 1000000

    if mv == <uint32_t>killers0: return 900000
    if mv == <uint32_t>killers1: return 800000

    # History (simple flat array indexed by from*64+to)
    return history[frm * 64 + to]

cdef void _sort_moves(uint32_t* moves, int n, int* scores) noexcept nogil:
    """Insertion sort by descending score (small n, so fine)."""
    cdef int i, j
    cdef uint32_t tmp_m
    cdef int tmp_s
    for i in range(1, n):
        tmp_m = moves[i]; tmp_s = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < tmp_s:
            moves[j+1] = moves[j]; scores[j+1] = scores[j]
            j -= 1
        moves[j+1] = tmp_m; scores[j+1] = tmp_s


# ─────────────────────────────────────────────────────────────────────────────
# Search state (per-thread)
# ─────────────────────────────────────────────────────────────────────────────

cdef struct SearchState:
    int  killers[MAX_PLY][2]   # killer moves per ply
    int  history[64*64]        # history heuristic [from*64+to]
    int  nodes
    int  seldepth
    bint stopped
    double start_time
    double time_limit
    int  static_evals[MAX_PLY]


# ─────────────────────────────────────────────────────────────────────────────
# Quiescence search
# ─────────────────────────────────────────────────────────────────────────────

cdef int qsearch(CBoard* b, int alpha, int beta, int ply, SearchState* ss,
                 TTable tt):
    ss.nodes += 1
    if (ss.nodes & 4095) == 0:
        if pytime.time() - ss.start_time >= ss.time_limit:
            ss.stopped = True
    if ss.stopped: return 0
    if ply > ss.seldepth: ss.seldepth = ply

    cdef int stand_pat = c_evaluate(b)
    if stand_pat >= beta: return stand_pat
    if stand_pat + 200 < alpha: return alpha   # delta pruning
    if alpha < stand_pat: alpha = stand_pat

    cdef uint32_t caps[128]
    cdef int ncaps = gen_legal_captures(b, caps)
    cdef int scores[128]
    cdef int i, score

    for i in range(ncaps):
        scores[i] = _score_move(b, caps[i], 0, 0, 0, ss.history, ply)
    _sort_moves(caps, ncaps, scores)

    for i in range(ncaps):
        do_make_move(b, caps[i])
        score = -qsearch(b, -beta, -alpha, ply + 1, ss, tt)
        do_unmake_move(b, caps[i])
        if ss.stopped: return 0
        if score >= beta: return score
        if score > alpha: alpha = score

    return alpha


# ─────────────────────────────────────────────────────────────────────────────
# Negamax (PVS)
# ─────────────────────────────────────────────────────────────────────────────

cdef int negamax(CBoard* b, int depth, int alpha, int beta, int ply,
                 SearchState* ss, TTable tt, bint do_null):
    # All cdef declarations at function scope (Cython requirement)
    cdef int tt_score, static_eval, null_score, razor_score, nm, score
    cdef int best_score, bound, quiet_count, reduction, R, old_ep, hkey, i, j
    cdef uint32_t tt_move, best_move
    cdef uint32_t moves[256]
    cdef int move_scores[256]
    cdef bint in_check, improving, has_pieces, f_prune, is_cap, is_quiet

    ss.nodes += 1
    if (ss.nodes & 1023) == 0:
        if pytime.time() - ss.start_time >= ss.time_limit:
            ss.stopped = True
    if ss.stopped: return 0

    # Repetition / fifty-move draw
    if ply > 0:
        if b.halfmove >= 100: return 0
        for i in range(b.hist_top - 2, b.hist_top - b.halfmove - 1, -2):
            if i < 0: break
            if b.hash_history[i] == b.zobrist: return 0

    if depth <= 0:
        return qsearch(b, alpha, beta, ply, ss, tt)
    if ply >= MAX_PLY - 1:
        return c_evaluate(b)
    if ply > ss.seldepth: ss.seldepth = ply

    # TT probe
    tt_score = INF + 1
    tt_move  = 0
    tt.probe(b.zobrist, depth, alpha, beta, ply, &tt_score, &tt_move)
    if tt_score != INF + 1 and ply > 0:
        return tt_score

    # Check extension
    in_check = is_in_check(b)
    if in_check: depth += 1

    # Static eval
    static_eval = c_evaluate(b)
    ss.static_evals[ply] = static_eval
    improving = ply < 2 or static_eval > ss.static_evals[ply - 2]

    # Null-move pruning
    has_pieces = (b.pieces[b.stm][KNIGHT] | b.pieces[b.stm][BISHOP] |
                  b.pieces[b.stm][ROOK]   | b.pieces[b.stm][QUEEN]) != 0
    if do_null and not in_check and depth >= 3 and static_eval >= beta and has_pieces:
        R = 3 + depth // 4
        b.stm ^= 1
        b.zobrist ^= ZOB_SIDE
        old_ep = b.ep_sq
        b.ep_sq = -1
        null_score = -negamax(b, depth - 1 - R, -beta, -beta + 1, ply + 1, ss, tt, False)
        b.stm ^= 1
        b.zobrist ^= ZOB_SIDE
        b.ep_sq = old_ep
        if ss.stopped: return 0
        if null_score >= beta:
            if null_score >= MATE_LOWER: null_score = beta
            return null_score

    # Razoring
    if not in_check and depth <= 2 and static_eval < alpha - 400:
        razor_score = qsearch(b, alpha, beta, ply, ss, tt)
        if razor_score <= alpha: return razor_score

    # Futility pruning condition
    f_prune = (not in_check and depth <= 3 and static_eval + 200 * depth < alpha)

    # Generate and sort moves
    nm = gen_legal_moves(b, moves)
    if nm == 0:
        return -MATE_SCORE + ply if in_check else 0

    for i in range(nm):
        move_scores[i] = _score_move(b, moves[i], tt_move,
                                      ss.killers[ply][0], ss.killers[ply][1],
                                      ss.history, ply)
    _sort_moves(moves, nm, move_scores)

    best_score  = -INF
    best_move   = 0
    bound       = BOUND_UPPER
    quiet_count = 0

    for i in range(nm):
        is_cap   = (move_flags(moves[i]) == MF_CAPTURE or
                    move_flags(moves[i]) == MF_EP or
                    move_flags(moves[i]) >= MF_N_PROMO_C)
        is_quiet = (not is_cap and move_flags(moves[i]) != MF_KS_CASTLE and
                    move_flags(moves[i]) != MF_QS_CASTLE)

        # Futility pruning
        if f_prune and is_quiet and i > 0 and best_score > -MATE_LOWER:
            continue

        # Late move pruning
        if not in_check and depth <= 4 and is_quiet and quiet_count >= 3 + depth * depth:
            continue

        # LMR
        reduction = 0
        if depth >= 3 and i >= 4 and is_quiet and not in_check:
            if depth >= 8:   reduction = 2
            elif depth >= 5: reduction = 1
            else:            reduction = 1
            if not improving: reduction += 1
            if reduction > depth - 2: reduction = depth - 2

        do_make_move(b, moves[i])
        if is_quiet: quiet_count += 1

        # PVS
        if i == 0:
            score = -negamax(b, depth - 1, -beta, -alpha, ply + 1, ss, tt, True)
        else:
            score = -negamax(b, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, ss, tt, True)
            if score > alpha and (reduction > 0 or score < beta):
                score = -negamax(b, depth - 1, -beta, -alpha, ply + 1, ss, tt, True)

        do_unmake_move(b, moves[i])
        if ss.stopped: return 0

        if score > best_score:
            best_score = score
            best_move  = moves[i]

        if score >= beta:
            if is_quiet and ply < MAX_PLY:
                # Killer moves
                if moves[i] != <uint32_t>ss.killers[ply][0]:
                    ss.killers[ply][1] = ss.killers[ply][0]
                    ss.killers[ply][0] = moves[i]
                # History
                hkey = move_from(moves[i]) * 64 + move_to(moves[i])
                ss.history[hkey] += depth * depth
                if ss.history[hkey] > 1000000:
                    for j in range(64 * 64): ss.history[j] //= 2
            tt.store(b.zobrist, depth, score, moves[i], BOUND_LOWER, ply)
            return score

        if score > alpha:
            alpha = score
            bound = BOUND_EXACT

    tt.store(b.zobrist, depth, best_score, best_move, bound, ply)
    return best_score


# ─────────────────────────────────────────────────────────────────────────────
# Iterative deepening
# ─────────────────────────────────────────────────────────────────────────────

cdef tuple _iterative_deepening(CBoard* b, double time_limit, int max_depth,
                                 TTable tt, object callback):
    cdef SearchState ss
    cdef int depth, score, prev_score, delta, alpha, beta
    cdef int tt_score_out
    cdef uint32_t best_move, tt_move_out
    cdef double elapsed
    cdef int nps

    memset(&ss, 0, sizeof(SearchState))
    ss.start_time = pytime.time()
    ss.time_limit = time_limit
    ss.stopped    = False

    tt.increment_age()

    score = 0; prev_score = 0; delta = 25; best_move = 0

    for depth in range(1, max_depth + 1):
        if ss.stopped: break

        ss.nodes    = 0
        ss.seldepth = 0

        if depth >= 4:
            alpha = prev_score - delta
            beta  = prev_score + delta
        else:
            alpha = -INF; beta = INF

        while True:
            score = negamax(b, depth, alpha, beta, 0, &ss, tt, True)
            if ss.stopped: break
            if score <= alpha:
                alpha -= delta
                delta = min(delta * 2, 500)
            elif score >= beta:
                beta  += delta
                delta  = min(delta * 2, 500)
            else:
                break

        if ss.stopped and best_move != 0:
            break

        # Retrieve best move from TT
        tt_score_out = INF + 1
        tt_move_out  = 0
        tt.probe(b.zobrist, depth, -INF, INF, 0, &tt_score_out, &tt_move_out)
        if tt_move_out:
            best_move = tt_move_out
        prev_score = score
        delta      = 25

        elapsed = pytime.time() - ss.start_time
        nps = <int>(ss.nodes / elapsed) if elapsed > 0 else 0

        if callback:
            try:
                callback(
                    depth    = depth,
                    seldepth = ss.seldepth,
                    score    = score,
                    move_uci = move_to_uci(best_move),
                    nodes    = ss.nodes,
                    nps      = nps,
                    elapsed  = elapsed,
                )
            except StopIteration:
                ss.stopped = True
                break

        if time_limit > 0 and elapsed >= time_limit * 0.85:
            break

    return move_to_uci(best_move), score, depth


# ─────────────────────────────────────────────────────────────────────────────
# Python-accessible wrapper class
# ─────────────────────────────────────────────────────────────────────────────

cdef class CSearch:
    """
    Python-accessible search interface.
    Maintains board state, TT, and can be called from the UCI handler.
    """
    cdef CBoard _board
    cdef TTable _tt
    cdef public bint _stop_requested   # set from Python to interrupt search

    def __cinit__(self, int tt_mb=256):
        board_clear(&self._board)
        self._tt = TTable(tt_mb)
        self._stop_requested = False

    def stop(self):
        """Signal the search to stop at the next depth boundary."""
        self._stop_requested = True

    def set_position(self, str fen, list moves=None):
        """Set up position from FEN, then apply list of UCI move strings."""
        parse_fen(&self._board, fen)
        if moves:
            for uci_str in moves:
                mv = uci_to_move_on_board(&self._board, uci_str)
                if mv:
                    do_make_move(&self._board, mv)

    def new_game(self):
        """Reset TT age and stop flag (call on ucinewgame)."""
        self._tt.clear()
        self._stop_requested = False

    def search(self, double time_limit, int max_depth=64, callback=None):
        """
        Run iterative deepening search.
        Returns (best_move_uci: str, score_cp: int, depth: int).
        callback(depth, seldepth, score, move_uci, nodes, nps, elapsed) called per depth.
        """
        self._stop_requested = False
        cs_self = self   # capture for closure

        def _cb_wrapper(**kw):
            if cs_self._stop_requested:
                raise StopIteration   # caught by _iterative_deepening to stop search
            if callback:
                callback(**kw)

        # Always pass wrapper so stop() works even without a user callback
        return _iterative_deepening(&self._board, time_limit, max_depth, self._tt,
                                    _cb_wrapper)

    def get_fen(self):
        return board_to_fen(&self._board)

    def legal_moves(self):
        """Return list of legal move UCI strings."""
        cdef uint32_t moves[256]
        cdef int n = gen_legal_moves(&self._board, moves)
        return [move_to_uci(moves[i]) for i in range(n)]

    def perft(self, int depth):
        """Perft test. Returns node count."""
        return _perft(&self._board, depth)

cdef int _perft(CBoard* b, int depth) nogil:
    if depth == 0: return 1
    cdef uint32_t moves[256]
    cdef int n = gen_legal_moves(b, moves)
    cdef int total = 0
    for i in range(n):
        do_make_move(b, moves[i])
        total += _perft(b, depth - 1)
        do_unmake_move(b, moves[i])
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Module initialisation
# ─────────────────────────────────────────────────────────────────────────────

_init_tables()
_init_castle_mask()
_init_mvvlva()
