"""Linux one-shot ROI12 supervisor; wait for old work, never stop or restart it.

launch_lock.json contract (root-relative paths):
  code_sha256: {"code/pipeline.py": SHA256, "code/supervise.py": SHA256, ...}
    Must cover every code/**/*.py file.
  bindings_sha256: SHA256 of --bindings
  required_receipts: {"preflight/receipt.json": SHA256, ...}  # nonempty
  receipt_statuses: {"preflight/receipt.json": "PASSED" or ["PASSED", ...]}
    Optional; default allowed statuses are PASS/PASSED/VERIFIED/READY/COMPLETE/COMPLETED.
  pipeline_args: []  # appended after pipeline.py --root ROOT; cannot override root.

No elapsed-time deadline grants permission. Two consecutive idle observations
at least60 seconds apart, then another fresh idle check, precede the only launch.
Existing pipeline-start/process/completion receipts forbid any restart.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT_NAME = "sod_roi12_20261009_v1"
DEFAULT_OK = {"PASS", "PASSED", "VERIFIED", "READY", "COMPLETE", "COMPLETED"}
ONCE_MARKERS = ("supervisor_launch_attempt.json", "pipeline_started.json", "pipeline_process.json",
                "pipeline_process_receipt.json", "pipeline_done.json")
ENVIRONMENT = {
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1", "SOD_ROI12_NO_AMP": "1",
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "Expected root-relative POSIX path")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe relative path")
    path = (root / rel).resolve()
    require(root in path.parents, "Path escapes new experiment root")
    return path


def atomic_json(path, value):
    require(not path.is_symlink(), "Refuse symlinked output")
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class Status:
    def __init__(self, root):
        self.root, self.last_signature = root, None

    def write(self, state, **detail):
        value = {"utc": utc(), "state": state, "supervisor_pid": os.getpid(), **detail}
        atomic_json(safe_path(self.root, "supervise_status.json"), value)
        # Memory counters/timestamps change every poll; only substantive state changes log.
        observation = detail.get("observation", {})
        signature = (state, tuple(observation.get("reasons", [])),
                     tuple(x["pid"] for x in observation.get("blocking_processes", [])),
                     tuple((x["gpu_uuid"], x["pid"]) for x in observation.get("compute_apps", [])),
                     detail.get("exit_code"), detail.get("error"))
        if signature != self.last_signature:
            event = {"utc": value["utc"], "state": state,
                     "reasons": observation.get("reasons", []),
                     "blocking_pids": [x["pid"] for x in observation.get("blocking_processes", [])],
                     "compute_pids": [x["pid"] for x in observation.get("compute_apps", [])]}
            for key in ("exit_code", "error", "pipeline_pid"):
                if key in detail:
                    event[key] = detail[key]
            path = safe_path(self.root, "supervise_events.jsonl")
            require(not path.is_symlink(), "Refuse symlinked event log")
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)
            self.last_signature = signature


def validate_lock(root, bindings, original_sha=None):
    path = safe_path(root, "launch_lock.json")
    digest = sha(path)
    require(original_sha is None or digest == original_sha, "Launch lock changed while waiting")
    lock = json.loads(path.read_text(encoding="utf-8"))
    code = lock.get("code_sha256")
    require(isinstance(code, dict) and code, "Missing immutable code SHA map")
    actual = {p.relative_to(root).as_posix() for p in (root / "code").rglob("*.py")}
    require(actual <= set(code), "Code SHA map omits a Python file")
    require({"code/supervise.py", "code/pipeline.py"} <= set(code), "Entrypoints are not bound")
    for rel, expected in code.items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid code SHA")
        require(sha(safe_path(root, rel)) == expected, f"Code SHA mismatch: {rel}")
    require(sha(bindings) == lock.get("bindings_sha256"), "Bindings SHA mismatch")
    receipts = lock.get("required_receipts")
    require(isinstance(receipts, dict) and receipts, "No required preflight receipts")
    statuses = lock.get("receipt_statuses", {})
    require(isinstance(statuses, dict), "Invalid receipt-status contract")
    verified = {}
    for rel, expected in receipts.items():
        receipt_path = safe_path(root, rel)
        require(sha(receipt_path) == expected, f"Receipt SHA mismatch: {rel}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        allowed = statuses.get(rel, list(DEFAULT_OK))
        allowed = [allowed] if isinstance(allowed, str) else allowed
        require(isinstance(allowed, list) and allowed and all(isinstance(v, str) for v in allowed),
                f"Invalid allowed receipt statuses: {rel}")
        require(receipt.get("status") in allowed, f"Preflight not accepted: {rel}")
        verified[rel] = receipt["status"]
    args = lock.get("pipeline_args")
    require(isinstance(args, list) and all(isinstance(x, str) for x in args), "pipeline_args must be a string list")
    require(not any(x == "--root" or x.startswith("--root=") for x in args), "Cannot override pipeline root")
    return lock, digest, verified


def visible_blockers(root):
    blockers, errors = [], []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            if not raw:
                continue
            argv = [x.decode(errors="replace") for x in raw.split(b"\0") if x]
            try:
                cwd = os.readlink(entry / "cwd")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                cwd = ""
            joined = " ".join(argv) + " " + cwd
            if str(root) in joined:
                continue
            python = any(Path(x).name.startswith("python") for x in argv[:2])
            p3 = "p3_readout_screen24" in joined
            prior_python = python and "/experiments/" in joined and any(
                word in joined.lower() for word in ("train", "supervis", "experiment.py", "pipeline.py"))
            if p3 or prior_python:
                blockers.append({"pid": int(entry.name), "kind": "prior_P3" if p3 else "other_experiment_python"})
        except (FileNotFoundError, ProcessLookupError):
            continue  # process finished during the scan
        except PermissionError:
            # Visible cmdlines could not be audited; remain fail-closed.
            errors.append(int(entry.name))
    return sorted(blockers, key=lambda x: x["pid"]), errors


def nvidia_query(fields, kind):
    command = ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    require(result.returncode == 0, f"nvidia-smi {kind} query failed (exit {result.returncode})")
    return list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))


def observe(root):
    result = {"utc": utc(), "idle": False, "reasons": [], "blocking_processes": [], "compute_apps": []}
    try:
        blockers, unreadable = visible_blockers(root)
        result["blocking_processes"] = blockers
        if blockers:
            result["reasons"].append("prior_experiment_process_live")
        if unreadable:
            result["reasons"].append("proc_cmdlines_unreadable")
            result["unreadable_process_count"] = len(unreadable)
        devices = nvidia_query("index,uuid", "gpu")
        device0 = [row[1].strip() for row in devices if len(row) == 2 and row[0].strip() == "0"]
        require(len(device0) == 1, "Visible GPU index0 could not be identified")
        result["gpu0_uuid"] = device0[0]
        for row in nvidia_query("gpu_uuid,pid,used_gpu_memory", "compute-apps"):
            require(len(row) == 3, "Malformed GPU compute-app row")
            pid = int(row[1].strip())
            result["compute_apps"].append({"gpu_uuid": row[0].strip(), "pid": pid,
                "used_memory_mib": row[2].strip(), "pid_visible_in_namespace": Path(f"/proc/{pid}").exists()})
        if result["compute_apps"]:
            # An unmapped host PID is still GPU occupancy, never proof of an idle card.
            result["reasons"].append("gpu_compute_application_present")
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        result["reasons"].append("observation_failed")
        result["error_type"] = type(exc).__name__
    result["idle"] = not result["reasons"]
    return result


def existing_once_marker(root):
    return next((name for name in ONCE_MARKERS if (root / name).exists()), None)


def supervise(root, bindings, poll_seconds):
    status = Status(root)
    marker = existing_once_marker(root)
    if marker:
        status.write("REFUSED_RESTART", error=f"Existing one-shot receipt: {marker}")
        return 2
    lock, lock_sha, receipts = validate_lock(root, bindings)
    first_idle = None
    while True:
        require(not existing_once_marker(root), "A pipeline receipt appeared while waiting; no restart")
        observation = observe(root)
        now = time.monotonic()
        if not observation["idle"]:
            first_idle = None
            status.write("WAITING_BUSY_OR_UNVERIFIED", observation=observation)
        elif first_idle is None:
            first_idle = now
            status.write("WAITING_SECOND_IDLE", observation=observation, minimum_idle_seconds=60)
        elif now - first_idle >= 60:
            lock, _, receipts = validate_lock(root, bindings, lock_sha)
            fresh = observe(root)
            if not fresh["idle"]:
                first_idle = None
                status.write("WAITING_BUSY_OR_UNVERIFIED", observation=fresh)
            else:
                require(not existing_once_marker(root), "Existing pipeline receipt forbids launch")
                argv = [sys.executable, "-B", str(safe_path(root, "code/pipeline.py")), "--root", str(root)]
                argv.extend(lock["pipeline_args"])
                record = {"status": "START_INTENT", "created_utc": utc(), "supervisor_pid": os.getpid(),
                    "launch_lock_sha256": lock_sha, "bindings_sha256": sha(bindings),
                    "verified_receipt_statuses": receipts, "argv": argv, "environment_policy": ENVIRONMENT,
                    "AMP_policy": "No AMP requested; pipeline must enforce and verify FP32/no-autocast",
                    "idle_seconds": now - first_idle, "last_idle_observation": fresh}
                started = safe_path(root, "supervisor_launch_attempt.json")
                # Commit the no-retry marker BEFORE attempting Popen, including spawn failures.
                with started.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
                    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
                env = os.environ.copy()
                env.update(ENVIRONMENT)
                env["SOD_ROI12_BINDINGS_PATH"] = str(bindings)
                log_path = safe_path(root, "pipeline_console.log")
                require(not log_path.is_symlink(), "Refuse symlinked console log")
                try:
                    with log_path.open("ab", buffering=0) as console:
                        child = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
                            stdout=console, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
                        record.update(status="RUNNING", pid=child.pid, started_utc=utc())
                        atomic_json(started, record)
                        status.write("PIPELINE_RUNNING", pipeline_pid=child.pid, environment_policy=ENVIRONMENT)
                        while child.poll() is None:
                            time.sleep(poll_seconds)
                        code = child.returncode
                    record.update(status="FINISHED", exit_code=code, finished_utc=utc())
                    atomic_json(started, record)
                    atomic_json(safe_path(root, "supervisor_completion.json"), record)
                    status.write("PIPELINE_FINISHED", pipeline_pid=child.pid, exit_code=code)
                    return code
                except Exception as exc:
                    record.update(status="SPAWN_OR_SUPERVISION_FAILED_NO_RETRY", error_type=type(exc).__name__, utc=utc())
                    atomic_json(started, record)
                    raise
        time.sleep(poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    require(sys.platform.startswith("linux"), "This supervisor requires Linux flock and /proc")
    import fcntl
    root = Path(args.root).resolve(strict=True)
    require(root.name == ROOT_NAME and root.parent.name == "experiments", "Unexpected experiment root")
    bindings = Path(args.bindings).resolve(strict=True)
    require(root in bindings.parents, "Bindings must be inside the new experiment root")
    require(args.poll_seconds >= 60, "Idle observations must be at least60 seconds apart")
    lock_path = safe_path(root, "supervisor.flock")
    require(not lock_path.is_symlink(), "Refuse symlinked supervisor lock")
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"state": "DUPLICATE_SUPERVISOR_REFUSED"}), flush=True)
            return 2
        try:
            return supervise(root, bindings, args.poll_seconds)
        except KeyboardInterrupt:
            Status(root).write("SUPERVISOR_INTERRUPTED_NO_RESTART", error="No old task or child was killed")
            return 130
        except Exception as exc:
            Status(root).write("SUPERVISOR_FAILED_NO_RETRY", error=str(exc))
            return 1
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
