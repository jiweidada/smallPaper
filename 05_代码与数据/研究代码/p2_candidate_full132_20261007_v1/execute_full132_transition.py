"""Control-plane worker, stored outside the frozen training code directory."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parent
    gate=json.loads((root/'gate_24_to_132.json').read_text())
    assert gate['allowed'] and gate['completed_epochs']==24
    assert gate['delta_APs_pp']>=.8 and gate['delta_AP_pp']>=-.2
    assert all(v['exit_code']==0 for v in gate['screen_exit_codes'].values())
    screen=Path('/24085403014/DEIM/experiments/p2_candidate_screen24_20261007_v1/p2_quota60')
    rows=[json.loads(s) for s in (screen/'run/log.txt').read_text().splitlines() if s.strip()]
    assert len(rows)==24 and rows[-1]['epoch']==23
    assert rows[-1]['test_coco_eval_bbox']==gate['P2_final_EMA_metrics']
    assert json.loads((screen/'exit.json').read_text())['exit_code']==0
    assert json.loads((root/'full132_cpu_checks.json').read_text())['passed']
    assert not (root/'p2_seed2/launch.json').exists()
    status=root/'transition_status.json'
    def write(phase,**extra):
        status.write_text(json.dumps({'phase':phase,'remote_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-43038ae9-d619-20e2-0216-822fd1c50fed',
             OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    write('AMP_PREFLIGHT')
    command=[sys.executable,'-u','preflight_full132_amp.py',str(root),
             '/24085403014/DEIM/experiments/enc_nms_screen24_20261006_v1/init/common_init_seed0.pth']
    with (root/'full132_amp_preflight_attempt1.log').open('w') as log:
        result=subprocess.run(command,cwd=root/'code',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    if result.returncode:
        write('PREFLIGHT_FAILED_NO_TRAINING',exit_code=result.returncode)
        sys.exit(result.returncode)
    write('LAUNCHING')
    with (root/'supervisor_full132.log').open('wb') as log:
        child=subprocess.Popen([sys.executable,'-u','supervise_full132.py',str(root)],cwd=root/'code',env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    write('SUPERVISOR_STARTED',supervisor_PID=child.pid,supervisor_start_ticks=ticks)
    print(json.dumps({'supervisor_PID':child.pid,'supervisor_start_ticks':ticks}),flush=True)


if __name__=='__main__':main()
