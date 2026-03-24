"""
NNUE-inspired dual neural network evaluation (PyTorch, GPU-accelerated).

Architecture:
  Big  net: 768 → 512 → SCReLU → 32 → ReLU → 1   (16 output buckets by material)
  Small net: 768 → 128 → SCReLU → 32 → ReLU → 1   (used at shallow nodes)

Input features: HalfKP-style binary 12×64 = 768 features per side (concatenated = 1536).
We simplify to a single 768-feature view (material-symmetric) for portability.

When no trained weights are found, the module returns None and the engine
falls back to the hand-crafted evaluator.
"""

from __future__ import annotations
import os
import sys
import chess
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "nnue.pt")
DEVICE = None  # determined at init


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_PIECE_INDEX = {
    (chess.WHITE, chess.PAWN):   0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK):   3,
    (chess.WHITE, chess.QUEEN):  4,
    (chess.WHITE, chess.KING):   5,
    (chess.BLACK, chess.PAWN):   6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK):   9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING):  11,
}


def board_to_features(board: chess.Board) -> np.ndarray:
    """Return a float32 array of shape (768,) — 12 piece planes × 64 squares."""
    feat = np.zeros(768, dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            idx = _PIECE_INDEX[(piece.color, piece.piece_type)]
            feat[idx * 64 + sq] = 1.0
    return feat


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

if TORCH_OK:
    class SCReLU(nn.Module):
        """Squared Clipped ReLU: f(x) = clamp(x, 0, 1)^2"""
        def forward(self, x):
            return torch.clamp(x, 0.0, 1.0).pow(2)

    class NNUENet(nn.Module):
        def __init__(self, hidden: int = 512, buckets: int = 16):
            super().__init__()
            self.hidden  = hidden
            self.buckets = buckets
            self.fc1     = nn.Linear(768, hidden)
            self.act1    = SCReLU()
            self.fc2     = nn.Linear(hidden, 32)
            self.act2    = nn.ReLU()
            # Per-bucket output heads
            self.heads   = nn.ModuleList([nn.Linear(32, 1) for _ in range(buckets)])

        def forward(self, x, bucket_idx=None):
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            if bucket_idx is None:
                # Use head 0 by default
                return self.heads[0](x)
            # bucket_idx: (batch,) long tensor
            outputs = torch.stack([h(x) for h in self.heads], dim=1)  # (B, buckets, 1)
            idx     = bucket_idx.view(-1, 1, 1).expand(-1, 1, 1)
            return outputs.gather(1, idx).squeeze(1)

    class SmallNNUENet(nn.Module):
        def __init__(self, hidden: int = 128):
            super().__init__()
            self.fc1  = nn.Linear(768, hidden)
            self.act1 = SCReLU()
            self.fc2  = nn.Linear(hidden, 32)
            self.act2 = nn.ReLU()
            self.out  = nn.Linear(32, 1)

        def forward(self, x):
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            return self.out(x)


# ---------------------------------------------------------------------------
# Material bucket helper
# ---------------------------------------------------------------------------

_MATERIAL_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

def _material_bucket(board: chess.Board, num_buckets: int = 16) -> int:
    total = 0
    for pt in chess.PIECE_TYPES:
        count = len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
        total += _MATERIAL_VALUES[pt] * count
    # Map 0-78 -> 0..num_buckets-1
    return min(int(total * num_buckets / 78), num_buckets - 1)


# ---------------------------------------------------------------------------
# NNUEEvaluator — public interface
# ---------------------------------------------------------------------------

class NNUEEvaluator:
    """
    Wraps big + small networks.
    Call evaluate(board, deep=True/False) → centipawns (side-to-move perspective).
    Returns None if not initialized.
    """

    def __init__(self):
        self.big_net   = None
        self.small_net = None
        self.device    = None
        self._ready    = False
        # Numpy weight arrays for zero-overhead inference (no PyTorch at search time)
        self._np_s_W1 = self._np_s_b1 = None  # small net fc1
        self._np_s_W2 = self._np_s_b2 = None  # small net fc2
        self._np_s_W3 = self._np_s_b3 = None  # small net out
        self._np_b_W1 = self._np_b_b1 = None  # big net fc1
        self._np_b_W2 = self._np_b_b2 = None  # big net fc2
        self._np_b_Wh = self._np_b_bh = None  # big net bucket heads (16, 32)

    def load_or_init(self):
        global DEVICE
        if not TORCH_OK:
            return False
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[NNUE] GPU detected: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        else:
            self.device = torch.device("cpu")
            print("[NNUE] No GPU found, using CPU", file=sys.stderr)
        DEVICE = self.device

        self.big_net   = NNUENet(hidden=512, buckets=16).to(self.device)
        self.small_net = SmallNNUENet(hidden=128).to(self.device)

        if os.path.exists(WEIGHTS_PATH):
            try:
                ckpt = torch.load(WEIGHTS_PATH, map_location=self.device)
                self.big_net.load_state_dict(ckpt["big"])
                self.small_net.load_state_dict(ckpt["small"])
                print(f"[NNUE] Loaded weights from {WEIGHTS_PATH}", file=sys.stderr)
                self._ready = True
            except Exception as e:
                print(f"[NNUE] Could not load weights ({e}), using PST fallback.", file=sys.stderr)
                self._ready = False
        else:
            print("[NNUE] No weights file found. Using hand-crafted evaluator.", file=sys.stderr)
            self._ready = False

        if self._ready:
            self._build_numpy_weights()
            print("[NNUE] Numpy inference ready", file=sys.stderr)

        return self._ready

    def _build_numpy_weights(self):
        """Extract model weights as contiguous float32 numpy arrays.
        At search time we use pure numpy — no PyTorch overhead, fully thread-safe."""
        # Small net  768 → 128 → SCReLU → 32 → ReLU → 1
        self._np_s_W1 = np.ascontiguousarray(
            self.small_net.fc1.weight.detach().cpu().numpy().T, dtype=np.float32)  # (768, 128)
        self._np_s_b1 = self.small_net.fc1.bias.detach().cpu().numpy().astype(np.float32)
        self._np_s_W2 = np.ascontiguousarray(
            self.small_net.fc2.weight.detach().cpu().numpy().T, dtype=np.float32)  # (128, 32)
        self._np_s_b2 = self.small_net.fc2.bias.detach().cpu().numpy().astype(np.float32)
        self._np_s_W3 = self.small_net.out.weight.detach().cpu().numpy().flatten().astype(np.float32)  # (32,)
        self._np_s_b3 = float(self.small_net.out.bias.detach().cpu().numpy()[0])

        # Big net  768 → 512 → SCReLU → 32 → ReLU → per-bucket(1)
        self._np_b_W1 = np.ascontiguousarray(
            self.big_net.fc1.weight.detach().cpu().numpy().T, dtype=np.float32)  # (768, 512)
        self._np_b_b1 = self.big_net.fc1.bias.detach().cpu().numpy().astype(np.float32)
        self._np_b_W2 = np.ascontiguousarray(
            self.big_net.fc2.weight.detach().cpu().numpy().T, dtype=np.float32)  # (512, 32)
        self._np_b_b2 = self.big_net.fc2.bias.detach().cpu().numpy().astype(np.float32)
        # Stack 16 bucket heads → (16, 32) and (16,)
        self._np_b_Wh = np.stack([
            h.weight.detach().cpu().numpy().flatten().astype(np.float32)
            for h in self.big_net.heads
        ])
        self._np_b_bh = np.array(
            [float(h.bias.detach().cpu().numpy()[0]) for h in self.big_net.heads],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Fast numpy forward passes (no PyTorch overhead at search time)
    # ------------------------------------------------------------------

    def _infer_small(self, feat: np.ndarray) -> float:
        """Single-position inference for small net. feat: (768,) float32."""
        x = feat @ self._np_s_W1 + self._np_s_b1          # (128,)
        x = np.clip(x, 0.0, 1.0); x *= x                  # SCReLU
        x = x @ self._np_s_W2 + self._np_s_b2             # (32,)
        np.maximum(x, 0.0, out=x)                          # ReLU
        return float(np.dot(x, self._np_s_W3) + self._np_s_b3)

    def _infer_big(self, feat: np.ndarray, bucket: int) -> float:
        """Single-position inference for big net. feat: (768,) float32."""
        x = feat @ self._np_b_W1 + self._np_b_b1          # (512,)
        x = np.clip(x, 0.0, 1.0); x *= x                  # SCReLU
        x = x @ self._np_b_W2 + self._np_b_b2             # (32,)
        np.maximum(x, 0.0, out=x)                          # ReLU
        return float(np.dot(x, self._np_b_Wh[bucket]) + self._np_b_bh[bucket])

    @property
    def ready(self) -> bool:
        return self._ready

    def evaluate(self, board: chess.Board, deep: bool = True) -> int | None:
        if not self._ready:
            return None
        feat = board_to_features(board)
        raw  = self._infer_big(feat, _material_bucket(board)) if deep else self._infer_small(feat)
        cp   = int(raw * 600)
        return cp if board.turn == chess.WHITE else -cp

    def batch_evaluate(self, boards: list, deep: bool = False) -> list:
        """Evaluate N positions in a single numpy matrix pass — amortises overhead."""
        if not self._ready or not boards:
            return [None] * len(boards)
        feats = np.stack([board_to_features(b) for b in boards])  # (N, 768)

        if deep:
            buckets = np.array([_material_bucket(b) for b in boards], dtype=np.int32)
            x = feats @ self._np_b_W1 + self._np_b_b1          # (N, 512)
            x = np.clip(x, 0.0, 1.0); x *= x
            x = x @ self._np_b_W2 + self._np_b_b2              # (N, 32)
            np.maximum(x, 0.0, out=x)
            W = self._np_b_Wh[buckets]                          # (N, 32)
            raws = np.einsum('ij,ij->i', x, W) + self._np_b_bh[buckets]
        else:
            x = feats @ self._np_s_W1 + self._np_s_b1          # (N, 128)
            x = np.clip(x, 0.0, 1.0); x *= x
            x = x @ self._np_s_W2 + self._np_s_b2              # (N, 32)
            np.maximum(x, 0.0, out=x)
            raws = x @ self._np_s_W3 + self._np_s_b3           # (N,)

        out = []
        for b, raw in zip(boards, raws):
            cp = int(float(raw) * 600)
            out.append(cp if b.turn == chess.WHITE else -cp)
        return out



# Singleton
_evaluator: NNUEEvaluator | None = None

def get_evaluator() -> NNUEEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = NNUEEvaluator()
        _evaluator.load_or_init()
    return _evaluator
