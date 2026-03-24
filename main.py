"""
NexusChess — Entry point.

Usage:
  python main.py            → Launch web UI (http://127.0.0.1:8765)
  python main.py --uci      → Run as UCI engine (for Arena, CuteChess, etc.)
  python main.py --perft 5  → Perft test from start position
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    args = sys.argv[1:]

    if "--uci" in args:
        from engine.uci import run_uci
        run_uci()
        return

    if "--perft" in args:
        idx   = args.index("--perft")
        depth = int(args[idx + 1]) if idx + 1 < len(args) else 5
        from engine import ChessEngine
        eng  = ChessEngine()
        import time
        t0   = time.time()
        n    = eng.perft(depth)
        elapsed = time.time() - t0
        print(f"Perft({depth}): {n:,} nodes in {elapsed:.2f}s ({int(n/elapsed):,} n/s)")
        return

    # Default: launch web UI
    print("=" * 60)
    print("  NexusChess — Sophisticated Chess Engine")
    print("=" * 60)
    print()
    print("  Initializing engine (8 threads, 512 MB TT)...")

    # Warm up engine in background
    import threading
    from engine import ChessEngine
    engine_ready = threading.Event()

    def _warm():
        ChessEngine(num_threads=8, tt_mb=512)
        engine_ready.set()

    threading.Thread(target=_warm, daemon=True).start()

    print("  Starting web server at http://127.0.0.1:8765")
    print()
    print("  Open your browser and navigate to:")
    print("    http://127.0.0.1:8765")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    import webbrowser
    import time
    time.sleep(0.5)
    webbrowser.open("http://127.0.0.1:8765")

    from ui.app import start_server
    start_server(host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
