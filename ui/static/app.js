/* =============================================================
   NexusChess — Frontend Logic
   - Chess.js-free: uses server for move validation
   - WebSocket for real-time engine analysis
   - SVG arrow rendering for best move
   ============================================================= */

'use strict';

// ─── State ────────────────────────────────────────────────────
let state = {
  fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  legalMoves: [],
  selected: null,
  flipped: false,
  engineEnabled: true,
  playAs: 'white',
  moveHistory: [],
  evalHistory: [],
  pendingPromo: null,
  lastFrom: null,
  lastTo: null,
  ws: null,
  analysisRunning: false,
};

// Unicode chess pieces
const PIECES = {
  'K':'♔','Q':'♕','R':'♖','B':'♗','N':'♘','P':'♙',
  'k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟',
};
const FILES = ['a','b','c','d','e','f','g','h'];

// ─── Board sizing ──────────────────────────────────────────────
function resizeBoard() {
  const topbar   = 56;
  const vPad     = 40;   // top + bottom padding in .layout
  const hPad     = 48;   // left + right padding in .layout
  const panelW   = 260;  // each side panel width
  const gapW     = 20;   // gap between panels and board section
  const labelW   = 24;   // rank-labels column
  const labelH   = 24;   // file-labels row
  const maxSize  = 640;

  const availH = window.innerHeight - topbar - vPad - labelH;
  const availW = window.innerWidth  - hPad - (panelW + gapW) * 2 - labelW;

  const size = Math.floor(Math.min(availH, availW, maxSize) / 8) * 8;
  document.documentElement.style.setProperty('--board-size', size + 'px');
}
window.addEventListener('resize', resizeBoard);

// ─── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  resizeBoard();
  buildBoard();
  buildLabels();
  connectWS();
  await refreshPosition();
  await fetchLegalMoves();
  renderBoard();
  updateStatus();
});

// ─── WebSocket ─────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  state.ws = new WebSocket(`${proto}://${location.host}/ws/analysis`);

  state.ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === 'depth') {
      updateEnginePanel(msg);
      drawArrow(msg.best_move);
      updateEvalBar(msg.score, msg.mate_in);
      addEvalHistory(msg.score);
      drawEvalGraph();
    } else if (msg.type === 'bestmove') {
      state.analysisRunning = false;
      if (shouldEnginePlay() && msg.move) {
        setTimeout(() => playEngineMove(msg.move), 200);
      }
    }
  };
  state.ws.onclose = () => {
    setTimeout(connectWS, 1000);
  };
  state.ws.onerror = () => {};
}

function wsSend(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
  }
}

// ─── Board building ───────────────────────────────────────────
function buildBoard() {
  const grid = document.getElementById('boardGrid');
  grid.innerHTML = '';
  // Inject SVG arrow layer
  const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('id','arrowLayer');
  svg.setAttribute('viewBox','0 0 8 8');
  svg.setAttribute('preserveAspectRatio','none');
  svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:5';
  grid.appendChild(svg);

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const sq = document.createElement('div');
      sq.classList.add('sq', (row + col) % 2 === 0 ? 'light' : 'dark');
      sq.dataset.row = row;
      sq.dataset.col = col;
      sq.addEventListener('click', onSquareClick);
      grid.appendChild(sq);
    }
  }
}

function buildLabels() {
  const rankEl = document.getElementById('rankLabels');
  const fileEl = document.getElementById('fileLabels');
  rankEl.innerHTML = '';
  fileEl.innerHTML = '';
  for (let r = 1; r <= 8; r++) {
    const s = document.createElement('span');
    s.textContent = state.flipped ? r : 9 - r;
    rankEl.appendChild(s);
  }
  for (let f = 0; f < 8; f++) {
    const s = document.createElement('span');
    s.textContent = state.flipped ? FILES[7 - f] : FILES[f];
    fileEl.appendChild(s);
  }
}

// ─── FEN parsing & rendering ──────────────────────────────────
function parseFEN(fen) {
  const parts = fen.split(' ');
  const rows  = parts[0].split('/');
  const board = [];
  for (let r = 0; r < 8; r++) {
    const row = [];
    for (const ch of rows[r]) {
      if ('12345678'.includes(ch)) {
        for (let i = 0; i < parseInt(ch); i++) row.push(null);
      } else {
        row.push(ch);
      }
    }
    board.push(row);
  }
  return { board, turn: parts[1], castling: parts[2], ep: parts[3] };
}

function squareToIndex(sq) {
  const f = FILES.indexOf(sq[0]);
  const r = parseInt(sq[1]) - 1;
  return { row: 7 - r, col: f };
}

function indexToSquare(row, col) {
  const f = state.flipped ? 7 - col : col;
  const r = state.flipped ? row     : 7 - row;
  return FILES[f] + (r + 1);
}

function renderBoard() {
  const grid  = document.getElementById('boardGrid');
  const squares = grid.querySelectorAll('.sq');
  const parsed  = parseFEN(state.fen);

  // Determine king in check
  let checkKingSq = null;
  if (state.fen.includes(' w ') || state.fen.includes(' b ')) {
    // Simple check detection: look for 'K' or 'k' and mark if in check
    // We let the server handle legality; highlight by turn
  }

  squares.forEach(sq => {
    const row = parseInt(sq.dataset.row);
    const col = parseInt(sq.dataset.col);

    // Reset classes
    sq.className = `sq ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;

    const boardRow  = state.flipped ? 7 - row : row;
    const boardCol  = state.flipped ? 7 - col : col;
    const piece     = parsed.board[boardRow][boardCol];
    const squareName = indexToSquare(row, col);

    // Last move highlights
    if (squareName === state.lastFrom) sq.classList.add('last-from');
    if (squareName === state.lastTo)   sq.classList.add('last-to');

    // Selection
    if (state.selected === squareName) sq.classList.add('selected');

    // Legal move hints
    if (state.selected) {
      const isLegal = state.legalMoves.some(m => m.startsWith(state.selected) && m.slice(2,4) === squareName);
      if (isLegal) {
        sq.classList.add(piece ? 'legal-capture' : 'legal-dot');
      }
    }

    // Piece
    let pieceEl = sq.querySelector('.piece');
    if (!pieceEl) {
      pieceEl = document.createElement('div');
      sq.appendChild(pieceEl);
    }
    const isWhitePiece = piece && piece === piece.toUpperCase();
    pieceEl.className = 'piece' + (piece ? (isWhitePiece ? ' white-piece' : ' black-piece') : '');
    pieceEl.textContent = piece ? (PIECES[piece] || '') : '';
  });
}

// ─── Square clicks ────────────────────────────────────────────
async function onSquareClick(evt) {
  const sq     = evt.currentTarget;
  const target = indexToSquare(parseInt(sq.dataset.row), parseInt(sq.dataset.col));
  const parsed = parseFEN(state.fen);

  if (state.pendingPromo) return;

  // If engine is playing for this side, ignore clicks
  if (shouldEnginePlay()) return;

  if (!state.selected) {
    // Select piece
    const boardRow = state.flipped ? 7 - parseInt(sq.dataset.row) : parseInt(sq.dataset.row);
    const boardCol = state.flipped ? 7 - parseInt(sq.dataset.col) : parseInt(sq.dataset.col);
    const piece    = parsed.board[boardRow][boardCol];
    if (!piece) return;
    const isWhite = piece === piece.toUpperCase();
    const myTurn  = (parsed.turn === 'w' && isWhite) || (parsed.turn === 'b' && !isWhite);
    if (!myTurn) return;
    state.selected = target;
    renderBoard();
  } else {
    if (state.selected === target) {
      state.selected = null;
      renderBoard();
      return;
    }

    // Check if valid move
    const matching = state.legalMoves.filter(
      m => m.startsWith(state.selected) && m.slice(2, 4) === target
    );

    if (matching.length === 0) {
      // Try selecting new piece
      const boardRow = state.flipped ? 7 - parseInt(sq.dataset.row) : parseInt(sq.dataset.row);
      const boardCol = state.flipped ? 7 - parseInt(sq.dataset.col) : parseInt(sq.dataset.col);
      const piece    = parsed.board[boardRow][boardCol];
      if (piece) {
        const isWhite = piece === piece.toUpperCase();
        const myTurn  = (parsed.turn === 'w' && isWhite) || (parsed.turn === 'b' && !isWhite);
        if (myTurn) { state.selected = target; renderBoard(); return; }
      }
      state.selected = null;
      renderBoard();
      return;
    }

    // Promotion?
    if (matching.length > 1 || (matching.length === 1 && matching[0].length === 5)) {
      // Check if it's a pawn reaching last rank
      const isPawnPromo = matching.some(m => m.length === 5);
      if (isPawnPromo) {
        state.pendingPromo = { from: state.selected, to: target };
        showPromoDialog(parsed.turn);
        return;
      }
    }

    await applyMove(matching[0]);
  }
}

async function applyMove(uci) {
  state.selected = null;
  clearArrow();
  const res  = await fetch(`/api/move/${uci}`, { method: 'POST' });
  const data = await res.json();
  if (!data.ok) return;

  state.lastFrom = uci.slice(0, 2);
  state.lastTo   = uci.slice(2, 4);
  state.fen      = data.fen;

  // Add to move history
  state.moveHistory.push(uci);
  updateMoveList();

  await fetchLegalMoves();
  renderBoard();
  updateStatus();

  if (data.game_over.over) {
    showGameOver(data.game_over);
    return;
  }

  // Trigger engine analysis
  if (state.engineEnabled) {
    startAnalysis();
  }
}

async function playEngineMove(uci) {
  await applyMove(uci);
}

// ─── Promotion dialog ─────────────────────────────────────────
function showPromoDialog(turn) {
  const pieces  = turn === 'w'
    ? [['q','♕'],['r','♖'],['b','♗'],['n','♘']]
    : [['q','♛'],['r','♜'],['b','♝'],['n','♞']];
  const choices = document.getElementById('promoChoices');
  choices.innerHTML = '';
  for (const [pt, glyph] of pieces) {
    const btn = document.createElement('div');
    btn.className   = 'promo-btn';
    btn.textContent = glyph;
    btn.onclick     = () => choosePromo(pt);
    choices.appendChild(btn);
  }
  document.getElementById('promoModal').style.display = 'flex';
}

async function choosePromo(pt) {
  document.getElementById('promoModal').style.display = 'none';
  if (!state.pendingPromo) return;
  const uci = state.pendingPromo.from + state.pendingPromo.to + pt;
  state.pendingPromo = null;
  await applyMove(uci);
}

// ─── Engine panel ─────────────────────────────────────────────
function startAnalysis() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.analysisRunning = true;
  wsSend({
    cmd:   'analyze',
    time:  parseFloat(document.getElementById('thinkTime').value),
    depth: parseInt(document.getElementById('maxDepth').value),
  });
}

function updateEnginePanel(msg) {
  const pv   = (msg.pv || []).slice(0, 5).join(' ');
  const nps  = msg.nps > 1000 ? `${(msg.nps/1000).toFixed(1)}k` : msg.nps;
  let scoreStr;
  if (msg.mate_in !== null && msg.mate_in !== undefined) {
    scoreStr = `M${Math.abs(msg.mate_in)}`;
  } else {
    scoreStr = (msg.score / 100).toFixed(2);
  }
  document.getElementById('enginePV').textContent  = pv || '—';
  document.getElementById('engineStats').textContent =
    `depth ${msg.depth}/${msg.seldepth}  •  ${msg.nodes.toLocaleString()} nodes  •  ${nps} nps`;
  document.getElementById('evalDepth').textContent = `depth ${msg.depth}`;
}

function updateEvalBar(score, mate_in) {
  const bar = document.getElementById('evalBar');
  const val = document.getElementById('evalValue');

  let cp = score;
  if (mate_in !== null && mate_in !== undefined) {
    cp = mate_in > 0 ? 9999 : -9999;
    val.textContent = mate_in > 0 ? `M${Math.abs(mate_in)}` : `-M${Math.abs(mate_in)}`;
  } else {
    val.textContent = (cp / 100).toFixed(2);
  }

  // Convert centipawns to bar percentage (logistic)
  const winPct = 50 + 50 * Math.tanh(cp / 400);
  bar.style.width = `${Math.max(4, Math.min(96, winPct))}%`;

  const wLbl = document.getElementById('evalLabelW');
  const bLbl = document.getElementById('evalLabelB');
  wLbl.textContent = cp >= 0 ? `+${(cp/100).toFixed(1)}` : '';
  bLbl.textContent = cp <  0 ? `+${(-cp/100).toFixed(1)}` : '';
}

function addEvalHistory(score) {
  state.evalHistory.push(score);
  if (state.evalHistory.length > 100) state.evalHistory.shift();
}

// ─── Evaluation graph ─────────────────────────────────────────
function drawEvalGraph() {
  const canvas = document.getElementById('evalGraph');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const hist = state.evalHistory;
  ctx.clearRect(0, 0, W, H);

  if (hist.length < 2) return;

  // Background
  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  ctx.fillRect(0, 0, W, H);

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth   = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0,   'rgba(167,139,250,0.5)');
  grad.addColorStop(0.5, 'rgba(167,139,250,0.1)');
  grad.addColorStop(1,   'rgba(96,165,250,0.05)');

  const pts = hist.map((v, i) => {
    const x = (i / (hist.length - 1)) * W;
    const y = H / 2 - Math.tanh(v / 400) * (H / 2 - 6);
    return [x, y];
  });

  // Fill
  ctx.beginPath();
  ctx.moveTo(pts[0][0], H / 2);
  pts.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.lineTo(pts[pts.length - 1][0], H / 2);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth   = 2;
  ctx.lineJoin    = 'round';
  pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
  ctx.stroke();

  // Current dot
  const [lx, ly] = pts[pts.length - 1];
  ctx.beginPath();
  ctx.arc(lx, ly, 3, 0, Math.PI * 2);
  ctx.fillStyle = '#a78bfa';
  ctx.fill();
}

// ─── SVG arrow ────────────────────────────────────────────────
function drawArrow(uci) {
  const svg = document.getElementById('arrowLayer');
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!uci || uci.length < 4) return;

  const from = uci.slice(0, 2);
  const to   = uci.slice(2, 4);

  const [fc, fr] = squareSVGPos(from);
  const [tc, tr] = squareSVGPos(to);

  // Arrowhead marker
  const defs   = document.createElementNS('http://www.w3.org/2000/svg','defs');
  const marker = document.createElementNS('http://www.w3.org/2000/svg','marker');
  marker.setAttribute('id','arrowHead');
  marker.setAttribute('markerWidth','4');
  marker.setAttribute('markerHeight','4');
  marker.setAttribute('refX','2');
  marker.setAttribute('refY','2');
  marker.setAttribute('orient','auto');
  const poly = document.createElementNS('http://www.w3.org/2000/svg','polygon');
  poly.setAttribute('points','0 0, 4 2, 0 4');
  poly.setAttribute('fill','rgba(255,200,50,0.85)');
  marker.appendChild(poly);
  defs.appendChild(marker);
  svg.appendChild(defs);

  const line = document.createElementNS('http://www.w3.org/2000/svg','line');
  line.setAttribute('x1', fc); line.setAttribute('y1', fr);
  line.setAttribute('x2', tc); line.setAttribute('y2', tr);
  line.setAttribute('stroke','rgba(255,200,50,0.85)');
  line.setAttribute('stroke-width','0.18');
  line.setAttribute('marker-end','url(#arrowHead)');
  svg.appendChild(line);
}

function squareSVGPos(sq) {
  const f = FILES.indexOf(sq[0]);
  const r = parseInt(sq[1]) - 1;
  const c = state.flipped ? 7 - f : f;
  const row = state.flipped ? r : 7 - r;
  return [c + 0.5, row + 0.5];
}

function clearArrow() {
  const svg = document.getElementById('arrowLayer');
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

// ─── Move list ────────────────────────────────────────────────
function updateMoveList() {
  const el   = document.getElementById('movesList');
  el.innerHTML = '';
  const hist = state.moveHistory;
  for (let i = 0; i < hist.length; i += 2) {
    const pair  = document.createElement('div');
    pair.className = 'move-pair';
    const num   = document.createElement('span');
    num.className = 'move-number';
    num.textContent = `${Math.floor(i/2)+1}.`;
    pair.appendChild(num);

    for (let j = 0; j < 2 && i + j < hist.length; j++) {
      const mv    = document.createElement('span');
      mv.className = 'move-san' + (i + j === hist.length - 1 ? ' current' : '');
      mv.textContent = hist[i + j];
      mv.onclick  = () => goToMove(i + j);
      pair.appendChild(mv);
    }
    el.appendChild(pair);
  }
  el.scrollTop = el.scrollHeight;
}

async function goToMove(idx) {
  // Replay from start to idx
  const moves = state.moveHistory.slice(0, idx + 1);
  await fetch('/api/set_position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', moves }),
  });
  await refreshPosition();
  await fetchLegalMoves();
  if (moves.length > 0) {
    const last = moves[moves.length - 1];
    state.lastFrom = last.slice(0, 2);
    state.lastTo   = last.slice(2, 4);
  }
  renderBoard();
  updateStatus();
}

// ─── Status ───────────────────────────────────────────────────
function updateStatus() {
  const parsed = parseFEN(state.fen);
  const ti     = document.getElementById('turnIndicator');
  const dot    = document.createElement('div');
  dot.className = `turn-dot ${parsed.turn === 'w' ? 'white' : 'black'}`;
  ti.innerHTML = '';
  ti.appendChild(dot);
  const txt = document.createTextNode(parsed.turn === 'w' ? "White to move" : "Black to move");
  ti.appendChild(txt);
}

function showGameOver(go) {
  const el  = document.getElementById('gameStatus');
  const msg = go.reason === 'checkmate'
    ? `${go.winner === 'white' ? '♔ White' : '♚ Black'} wins by checkmate`
    : go.reason.replace(/_/g, ' ');
  el.textContent = msg;
}

// ─── Controls ─────────────────────────────────────────────────
async function newGame() {
  state.moveHistory = [];
  state.evalHistory = [];
  state.selected    = null;
  state.lastFrom    = null;
  state.lastTo      = null;
  clearArrow();
  document.getElementById('gameStatus').textContent = '';
  document.getElementById('enginePV').textContent   = 'Waiting...';
  document.getElementById('engineStats').textContent = '';
  updateEvalBar(0, null);
  drawEvalGraph();
  updateMoveList();
  await fetch('/api/reset', { method: 'POST' });
  await refreshPosition();
  await fetchLegalMoves();
  renderBoard();
  updateStatus();
  if (state.engineEnabled && shouldEnginePlay()) {
    startAnalysis();
  }
}

async function undoMove() {
  await fetch('/api/undo', { method: 'POST' });
  if (state.moveHistory.length) state.moveHistory.pop();
  if (state.evalHistory.length) state.evalHistory.pop();
  state.selected = null;
  clearArrow();
  await refreshPosition();
  await fetchLegalMoves();
  renderBoard();
  updateStatus();
  updateMoveList();
  drawEvalGraph();
}

function flipBoard() {
  state.flipped = !state.flipped;
  buildBoard();
  buildLabels();
  renderBoard();
}

function toggleEngine() {
  state.engineEnabled = document.getElementById('engineToggle').checked;
  if (state.engineEnabled) startAnalysis();
}

function setPlayAs() {
  state.playAs = document.getElementById('playAs').value;
}

function shouldEnginePlay() {
  const parsed = parseFEN(state.fen);
  const turn   = parsed.turn;
  if (state.playAs === 'both')  return false;
  if (state.playAs === 'white') return turn === 'b';
  if (state.playAs === 'black') return turn === 'w';
  return false;
}

// ─── API helpers ──────────────────────────────────────────────
async function refreshPosition() {
  const res  = await fetch('/api/fen');
  const data = await res.json();
  state.fen  = data.fen;
}

async function fetchLegalMoves() {
  const res  = await fetch('/api/legal_moves');
  const data = await res.json();
  state.legalMoves = data.moves;
}
