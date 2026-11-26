"""Bounded local command runner. Logs stay next to this script. No automatic retries."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--timeout", type=int, required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command and a.command[0] == "--" else a.command
root = Path(__file__).resolve().parent
env = os.environ.copy()
for key in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME", "PIP_CACHE_DIR"):
    env[key] = str(root / "cache" / key.lower())
env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
for key in ("TEMP", "TMP"):
    env[key] = str(root.parents[2] / "tmp/cpu_smoke_tmp_20261029")
    Path(env[key]).mkdir(parents=True, exist_ok=True)
started = time.monotonic()
receipt = {"command": command, "working_dir": str(root), "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "timeout_seconds": a.timeout, "auto_retry": False, "scope": "local bounded CPU functional checks"}
with (root / (a.name + ".log")).open("w", encoding="utf-8") as log:
    proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt["pid"] = proc.pid
    print(f"START {a.name} pid={proc.pid} timeout={a.timeout}s", flush=True)
    next_notice = started + 30
    while proc.poll() is None:
        now = time.monotonic()
        if now - started >= a.timeout:
            print(f"HARD_TIMEOUT {a.name}: killing direct child pid={proc.pid}", flush=True)
            proc.kill()
            receipt["hard_timeout"] = True
            break
        if now >= next_notice:
            size = (root / (a.name + ".log")).stat().st_size
            print(f"ALIVE {a.name} pid={proc.pid} elapsed={now-started:.0f}s log_bytes={size}", flush=True)
            next_notice = now + 30
        time.sleep(.5)
    receipt["exit_code"] = proc.wait()
receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
receipt["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
(root / (a.name + "_receipt.json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(receipt, ensure_ascii=True), flush=True)
sys.exit(0 if receipt["exit_code"] == 0 else 1)
