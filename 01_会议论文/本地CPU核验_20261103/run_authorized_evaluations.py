"""Run each authorized checkpoint once, sequentially; no automatic retries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "execution_ledger.json"
LOGS = ROOT / "logs"
RUNS = ("N0", "N4", "S1")
TIMEOUT_SECONDS = 1800

def now():
    return datetime.now(timezone.utc).isoformat()

def save(value):
    temporary = LEDGER.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEDGER)

def main():
    if LEDGER.exists():
        raise FileExistsError("An execution ledger already exists; do not start a duplicate queue")
    authorization = json.loads((ROOT / "authorization.json").read_text(encoding="utf-8"))
    if authorization["status"] != "AUTHORIZED_LOCAL_CPU_EVALUATION":
        raise ValueError("Expected explicit local evaluation authorization record")
    for run_id, filename in (("N0", "N0.json"), ("N4", "N4_utf8.json"), ("S1", "S1_utf8.json")):
        check = json.loads((ROOT / "compatibility" / filename).read_text(encoding="utf-8"))
        if check["status"] != "STRICT_LOAD_AND_CONSTRUCTION_OK_NO_FORWARD":
            raise ValueError(f"Strict loading preflight not passed: {run_id}")
        if (ROOT / "results" / run_id).exists():
            raise FileExistsError(f"Results already exist for {run_id}; do not repeat")
    LOGS.mkdir(exist_ok=True)
    state = {"status": "RUNNING", "supervisor_pid": os.getpid(), "started_utc": now(),
             "run_order": list(RUNS), "timeout_seconds_per_run": TIMEOUT_SECONDS,
             "auto_retry": False, "runs": []}
    save(state)
    for run_id in RUNS:
        command = [sys.executable, "-I", str(ROOT / "runtime_entry.py"), str(ROOT / "evaluate_cpu.py"),
                   "--run-id", run_id, "--threads", "8", "--confirm-local-cpu-evaluation"]
        entry = {"run_id": run_id, "command": command, "started_utc": now(), "status": "STARTING"}
        state["runs"].append(entry)
        with (LOGS / (run_id + ".log")).open("xb") as log:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            entry.update({"pid": process.pid, "status": "RUNNING"})
            save(state)
            print(f"Started {run_id}; PID {process.pid}", flush=True)
            started = time.monotonic()
            last_print = started
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= TIMEOUT_SECONDS:
                    print(f"Hard timeout for {run_id}; terminating this process and stopping the queue", flush=True)
                    process.terminate()
                    process.wait(timeout=30)
                    entry["status"] = "HARD_TIMEOUT"
                    break
                if time.monotonic() - last_print >= 30:
                    receipt_path = ROOT / "results" / run_id / "receipt.json"
                    if receipt_path.exists():
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        print(f"{run_id}: {receipt['status']}, {receipt['completed_images']}/548", flush=True)
                    last_print = time.monotonic()
                time.sleep(1)
            entry.update({"exit_code": process.returncode, "ended_utc": now()})
        receipt_path = ROOT / "results" / run_id / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
        if process.returncode != 0 or receipt.get("status") != "COMPLETED" or receipt.get("completed_images") != 548:
            entry["status"] = entry["status"] if entry["status"] == "HARD_TIMEOUT" else "FAILED"
            state.update({"status": "STOPPED_ON_FAILURE", "ended_utc": now()})
            save(state)
            print(f"Stopped at {run_id}: exit {process.returncode}; see its preserved log", flush=True)
            return 1
        entry["status"] = "COMPLETED"
        save(state)
        print(f"Completed {run_id}: 548/548", flush=True)
    state.update({"status": "COMPLETED", "ended_utc": now()})
    save(state)
    print("Completed all three authorized single-pass CPU evaluations", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
