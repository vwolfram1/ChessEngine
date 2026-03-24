"""Test the engine as a UCI subprocess."""
import subprocess, time, sys, threading

PYTHON = r'c:/Users/vinni/OneDrive/Documents/Projects/.venv/Scripts/python.exe'

proc = subprocess.Popen(
    [PYTHON, 'main.py', '--uci'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True, bufsize=1,
    cwd=r'c:/Users/vinni/OneDrive/Documents/Projects/ChessEngine',
)

lines = []
def read_out():
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        print(f'[OUT] {line}')

def read_err():
    for line in proc.stderr:
        print(f'[ERR] {line.rstrip()}')

threading.Thread(target=read_out, daemon=True).start()
threading.Thread(target=read_err, daemon=True).start()

def send(cmd):
    print(f'[IN]  {cmd}')
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()

send('uci')
time.sleep(0.3)
send('isready')
time.sleep(4)          # wait for NNUE load
send('ucinewgame')
send('position startpos')
send('go movetime 3000')
time.sleep(6)          # wait for search to finish + bestmove
send('quit')
proc.wait(timeout=5)
print('--- done ---')
print('Lines received:', lines[-5:])
