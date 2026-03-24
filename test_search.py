"""Quick test: does the search produce a move?"""
import sys
import chess
sys.path.insert(0, '.')

print("Importing engine modules...", flush=True)
from engine.search import search, SearchState, SMPSearch
from engine.neural_net import get_evaluator

print("Loading NNUE...", flush=True)
nn = get_evaluator()
print(f"NNUE ready: {nn.ready}", flush=True)

board = chess.Board()
print(f"Position: {board.fen()}", flush=True)

def cb(**kw):
    print(f"  depth={kw['depth']} score={kw['score']} move={kw['best_move']} nodes={kw['nodes']} elapsed={kw['elapsed']:.2f}s", flush=True)

print("\nTesting single-thread search (1s)...", flush=True)
ss = SearchState()
move, score, pv = search(board, max_depth=10, time_limit=1.0, ss=ss, callback=cb)
print(f"Best move: {move}, score: {score}", flush=True)

print("\nTesting SMPSearch (2 threads, 2s)...", flush=True)
smp = SMPSearch(num_threads=2)
move, score, pv = smp.search(board, max_depth=10, time_limit=2.0, callback=cb)
print(f"Best move: {move}, score: {score}", flush=True)

print("\nAll done!", flush=True)
