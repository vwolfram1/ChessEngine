"""
Cython bitboard engine bridge.
Tries to import the compiled cchess extension; falls back gracefully if unavailable.
"""

try:
    from .cchess import CSearch  # noqa: F401
    CYTHON_AVAILABLE = True
except ImportError:
    CSearch = None
    CYTHON_AVAILABLE = False

__all__ = ["CSearch", "CYTHON_AVAILABLE"]
