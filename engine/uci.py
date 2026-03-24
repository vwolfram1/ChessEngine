"""
UCI (Universal Chess Interface) protocol handler.
Allows the engine to be used with any UCI-compatible GUI (Arena, Cute Chess, etc.)
"""

import sys
import threading
import chess
from .engine import ChessEngine

ENGINE_NAME    = "NexusChess"
ENGINE_AUTHOR  = "Nexus"
ENGINE_VERSION = "1.0"

OPTIONS = {
    "Hash":       {"type": "spin", "default": 512, "min": 8,   "max": 16384, "value": 512},
    "Threads":    {"type": "spin", "default": 8,   "min": 1,   "max": 256,   "value": 8},
    "MultiPV":    {"type": "spin", "default": 1,   "min": 1,   "max": 500,   "value": 1},
    "UseNNUE":    {"type": "check", "default": True, "value": True},
    "SyzygyPath": {"type": "string", "default": "",  "value": ""},
    "BookPath":   {"type": "string", "default": "",  "value": ""},
    "Contempt":   {"type": "spin", "default": 0, "min": -100, "max": 100, "value": 0},
}


class UCIHandler:
    def __init__(self):
        self._engine: ChessEngine | None = None
        self._stop    = threading.Event()

    def _ensure_engine(self):
        if self._engine is None:
            threads = OPTIONS["Threads"]["value"]
            tt_mb   = OPTIONS["Hash"]["value"]
            self._engine = ChessEngine(num_threads=threads, tt_mb=tt_mb)

    def run(self):
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue
            self._handle(line)

    def _handle(self, line: str):
        parts = line.split()
        cmd   = parts[0] if parts else ""

        if cmd == "uci":
            self._uci()
        elif cmd == "isready":
            self._ensure_engine()
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            self._ensure_engine()
            self._engine.smp.shared_tt.clear()
            if self._engine._csearch:
                self._engine._csearch.new_game()
        elif cmd == "setoption":
            self._setoption(parts[1:])
        elif cmd == "position":
            self._ensure_engine()
            self._position(parts[1:])
        elif cmd == "go":
            self._ensure_engine()
            self._go(parts[1:])
        elif cmd == "stop":
            self._stop.set()
            if self._engine:
                self._engine.stop_search()
        elif cmd == "quit":
            sys.exit(0)
        elif cmd == "perft":
            depth = int(parts[1]) if len(parts) > 1 else 5
            self._ensure_engine()
            print(f"Nodes: {self._engine.perft(depth)}", flush=True)

    def _uci(self):
        print(f"id name {ENGINE_NAME} {ENGINE_VERSION}")
        print(f"id author {ENGINE_AUTHOR}")
        for name, opt in OPTIONS.items():
            t = opt["type"]
            v = opt["default"]
            if t == "spin":
                print(f"option name {name} type spin default {v} min {opt['min']} max {opt['max']}")
            elif t == "check":
                print(f"option name {name} type check default {'true' if v else 'false'}")
            elif t == "string":
                print(f"option name {name} type string default {v or '<empty>'}")
        print("uciok", flush=True)

    def _setoption(self, tokens: list):
        # setoption name <N> value <V>
        if "name" not in tokens:
            return
        ni = tokens.index("name")
        vi = tokens.index("value") if "value" in tokens else len(tokens)
        name  = " ".join(tokens[ni + 1:vi])
        value = " ".join(tokens[vi + 1:]) if vi < len(tokens) else ""
        if name in OPTIONS:
            opt = OPTIONS[name]
            if opt["type"] == "spin":
                OPTIONS[name]["value"] = int(value)
            elif opt["type"] == "check":
                OPTIONS[name]["value"] = value.lower() == "true"
            elif opt["type"] == "string":
                OPTIONS[name]["value"] = value

    def _position(self, tokens: list):
        if not tokens:
            return
        if tokens[0] == "startpos":
            fen   = chess.STARTING_FEN
            moves = tokens[2:] if len(tokens) > 1 and tokens[1] == "moves" else []
        elif tokens[0] == "fen":
            fen_parts = []
            i = 1
            while i < len(tokens) and tokens[i] != "moves":
                fen_parts.append(tokens[i])
                i += 1
            fen   = " ".join(fen_parts)
            moves = tokens[i + 1:] if i < len(tokens) and tokens[i] == "moves" else []
        else:
            return
        self._engine.set_position(fen, moves)

    def _go(self, tokens: list):
        self._stop.clear()
        params = {}
        i = 0
        while i < len(tokens):
            key = tokens[i]
            if i + 1 < len(tokens):
                try:
                    params[key] = int(tokens[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
            params[key] = True
            i += 1

        # Time management
        wtime  = params.get("wtime", 0)
        btime  = params.get("btime", 0)
        winc   = params.get("winc", 0)
        binc   = params.get("binc", 0)
        movetime = params.get("movetime", 0)
        depth  = params.get("depth", 64)
        infinite = "infinite" in params

        if infinite:
            time_limit = 3600.0
        elif movetime:
            time_limit = movetime / 1000.0
        else:
            board = self._engine.board
            my_time = wtime if board.turn == chess.WHITE else btime
            my_inc  = winc  if board.turn == chess.WHITE else binc
            moves_to_go = params.get("movestogo", 30)
            time_limit  = max(0.1, (my_time / moves_to_go + my_inc * 0.8) / 1000.0)
            time_limit  = min(time_limit, my_time / 1000.0 * 0.9)

        def _on_depth(depth, seldepth, score, best_move, pv, nodes, elapsed, nps):
            if self._stop.is_set():
                return
            mate = None
            if abs(score) >= 99000 - 500:
                mate_in = (100000 - abs(score) + 1) // 2
                if score < 0:
                    mate_in = -mate_in
                mate = mate_in
            score_str = (f"mate {mate}" if mate is not None else f"cp {score}")
            pv_str    = " ".join(m.uci() if hasattr(m, 'uci') else str(m) for m in pv)
            elapsed_ms = int(elapsed * 1000)
            print(
                f"info depth {depth} seldepth {seldepth} score {score_str} "
                f"nodes {nodes} nps {nps} time {elapsed_ms} pv {pv_str}",
                flush=True,
            )

        def _on_done(move_uci, score, pv_uci):
            print(f"bestmove {move_uci or '0000'}", flush=True)

        self._engine.start_async_search(
            time_limit=time_limit,
            depth_limit=depth,
            on_depth=_on_depth,
            on_done=_on_done,
        )


def run_uci():
    handler = UCIHandler()
    handler.run()
