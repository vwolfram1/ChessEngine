"""
FastAPI web server for NexusChess UI.
WebSocket streams engine analysis in real-time.
"""

import asyncio
import json
import os
import threading
import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ChessEngine

app    = FastAPI(title="NexusChess")
engine = ChessEngine(num_threads=8, tt_mb=512)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/fen")
async def get_fen():
    return {"fen": engine.get_fen()}


@app.get("/api/legal_moves")
async def legal_moves():
    return {"moves": engine.get_legal_moves()}


@app.post("/api/move/{uci}")
async def make_move(uci: str):
    ok  = engine.make_move(uci)
    gover = engine.is_game_over()
    return {"ok": ok, "fen": engine.get_fen(), "game_over": gover}


@app.post("/api/undo")
async def undo():
    ok = engine.undo_move()
    return {"ok": ok, "fen": engine.get_fen()}


@app.post("/api/reset")
async def reset():
    engine.set_position()
    return {"fen": engine.get_fen()}


@app.post("/api/set_position")
async def set_position(body: dict):
    fen   = body.get("fen", chess.STARTING_FEN)
    moves = body.get("moves", [])
    engine.set_position(fen, moves)
    return {"fen": engine.get_fen()}


@app.websocket("/ws/analysis")
async def ws_analysis(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw  = await ws.receive_text()
            data = json.loads(raw)
            cmd  = data.get("cmd")

            if cmd == "analyze":
                time_limit  = float(data.get("time",  10.0))
                depth_limit = int(data.get("depth", 30))

                queue: asyncio.Queue = asyncio.Queue()

                def on_depth(depth, seldepth, score, best_move, pv, nodes, elapsed, nps):
                    mate_in = None
                    if abs(score) >= 99000 - 500:
                        mate_in = (100000 - abs(score) + 1) // 2
                        if score < 0:
                            mate_in = -mate_in
                    pv_uci = [m.uci() for m in pv]
                    payload = {
                        "type": "depth",
                        "depth": depth,
                        "seldepth": seldepth,
                        "score": score,
                        "mate_in": mate_in,
                        "best_move": best_move.uci() if best_move else None,
                        "pv": pv_uci,
                        "nodes": nodes,
                        "nps": nps,
                        "elapsed": round(elapsed, 3),
                    }
                    asyncio.run_coroutine_threadsafe(queue.put(payload), loop)

                def on_done(move_uci, score, pv_uci):
                    payload = {
                        "type": "bestmove",
                        "move": move_uci,
                        "score": score,
                        "pv": pv_uci,
                    }
                    asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                engine.start_async_search(
                    time_limit=time_limit,
                    depth_limit=depth_limit,
                    on_depth=on_depth,
                    on_done=on_done,
                )

                # Stream analysis updates to the client
                while True:
                    item = await asyncio.wait_for(queue.get(), timeout=time_limit + 5)
                    if item is None:
                        break
                    await ws.send_text(json.dumps(item))

            elif cmd == "stop":
                engine.stop_search()

            elif cmd == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except (WebSocketDisconnect, asyncio.TimeoutError):
        engine.stop_search()


def start_server(host: str = "127.0.0.1", port: int = 8765):
    uvicorn.run(app, host=host, port=port, log_level="warning")
