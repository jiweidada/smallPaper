"""Read a specific frozen job identity and live lower-bound counters; never launch work."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess
ROOT=Path(__file__).resolve().parent
def rd(p):return json.loads(p.read_text()) if p.exists() else None
def process(expected,script):
    if expected is None:return {'identity_available':False,'live':False,'terminal_inferred':False}
    p=Path('/proc')/str(expected['pid'])
    if not p.exists():return {'identity_available':True,'live':False,'process_path_missing':True,'expected':expected}
    try:
        s=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        match=int(s[21])==expected['start_ticks'] and str(ROOT/script) in cmd
        return {'identity_available':True,'live':match and s[2] not in ['Z','X'],'identity_matches':match,
                'pid':expected['pid'],'start_ticks':int(s[21]),'state':s[2],'cmdline':cmd}
    except FileNotFoundError:return {'identity_available':True,'live':False,'process_vanished_during_read':True}
def tail_json(p):
    if not p.exists():return None
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-65536));lines=f.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):continue
    return None
def main():
    launch=rd(ROOT/'supervisor_launch.json');child=rd(ROOT/'training_child.json')
    x={'utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only snapshot; counters from separate live streams are observed lower bounds',
       'supervisor':process(launch,'supervise_job.py'),'child':process(child,'train_pilot.py'),
       'progress':rd(ROOT/'training_progress.json'),'latest_forward_entry':tail_json(ROOT/'forward_events.jsonl'),
       'latest_optimizer_event':tail_json(ROOT/'optimizer_events.jsonl'),'latest_training_event':tail_json(ROOT/'training_events.jsonl'),
       'latest_epoch':rd(ROOT/'latest_epoch.json'),'training_result':rd(ROOT/'training_result.json'),
       'supervisor_result':rd(ROOT/'supervisor_result.json'),'supervisor_failure':rd(ROOT/'supervisor_failure.json'),
       'parameter_change_check':rd(ROOT/'initial_to_first_update_parameter_changes.json'),
       'GPU':subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),
       'compute_apps':subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}
    print(json.dumps(x,ensure_ascii=False))
if __name__=='__main__':main()
