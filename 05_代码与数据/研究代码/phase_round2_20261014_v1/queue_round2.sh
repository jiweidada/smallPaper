#!/usr/bin/env bash
# Append-only queue: failures stop the queue; existing outputs are never removed.
set -euo pipefail
N="$1"; PY="$2"; shift 2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$N/code"
for job in "$@"; do
  case "$job" in
    c_mean_full96:[012]|rank16_full96:[012]|rank16_mean_full96:[012]|c_full96:2) ;;
    *) echo "REFUSE: job outside the frozen round-two queue: $job"; exit 2 ;;
  esac
  arm="${job%%:*}"; seed="${job##*:}"; id="${arm}_seed${seed}"
  out="$N/outputs/$id"; log="$N/logs/$id.log"; state="$N/jobs/$id.json"
  if [ -e "$out" ] || [ -e "$log" ] || [ -e "$state" ]; then
    echo "REFUSE: existing job artifacts: $id"; exit 2
  fi
  # A separate acceptance receipt, created after audit and GPU preflight, is required.
  test -f "$N/artifacts/round2_20261014_v1/TRAINING_GATE_ACCEPTED.json"
  "$PY" - "$state" "$id" <<'PY'
import json,os,pathlib,sys,time
with pathlib.Path(sys.argv[1]).open('x') as f:
 json.dump({'job':sys.argv[2], 'state':'running','start':time.time(),'queue_pid':os.getppid()},f)
PY
  set +e
  "$PY" -u train.py -c "configs/experiments/phase/${arm}.yml" --seed "$seed" --use-amp \
    -t "$N/artifacts/common_init_seed0.pth" --output-dir "$out" > "$log" 2>&1
  rc=$?
  set -e
  "$PY" - "$state" "$rc" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); rc=int(sys.argv[2])
d.update(state='complete' if rc==0 else 'failed',returncode=rc,end=time.time())
p.write_text(json.dumps(d))
PY
  if [ "$rc" -ne 0 ]; then echo "STOP: $id failed with returncode=$rc"; exit "$rc"; fi
done
