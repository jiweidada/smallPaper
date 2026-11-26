"""Read-only progress/identity observer. Does not import or execute a model."""
from pathlib import Path
import argparse,hashlib,json,math,time

ROOT=Path('/root/uav_a48_20261101_v1')

def read(path):
    return json.loads(path.read_text()) if path.is_file() else None

def identity(path):
    expected=read(path)
    if expected is None:return None
    pid=expected['pid'];proc=Path('/proc')/str(pid)
    result={'pid':pid,'expected_start_ticks':expected['start_ticks'],'present':proc.exists()}
    if proc.exists():
        try:
            stat=(proc/'stat').read_text().split(') ',1)[1].split()
            cmd=(proc/'cmdline').read_bytes().decode(errors='replace')
            result.update(state=stat[0],same_start_ticks=int(stat[19])==expected['start_ticks'],
                          owned_command=str(ROOT) in cmd)
        except FileNotFoundError:result['present']=False
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-freeze',action='store_true');args=parser.parse_args()
    out={'observed_server_unix':time.time(),'observation_only':True,
         'supervisor':read(ROOT/'supervisor_progress.json'),'supervisor_identity':identity(ROOT/'supervisor_identity.json'),
         'arms':{},'three_arm_result':read(ROOT/'three_arm_result.json')}
    for arm in ['B','M','Q']:
        folder=ROOT/('arm_'+arm)
        history=read(folder/'completed_epochs.json') or {}
        progress=read(folder/'progress.json')
        result=read(folder/'result.json')
        stderr=folder/'stderr.log'
        row={'progress':progress,'identity':identity(folder/'child_identity.json'),'result':result,
             'completed_epochs':history.get('completed_epochs',0),
             'evaluations':{str(e['epoch']):e['EMA_validation'] for e in history.get('events',[]) if 'EMA_validation' in e},
             'stderr_tail':stderr.read_text()[-2000:] if stderr.exists() else None}
        if progress:
            row['reported_loss_grad_finite']=all(math.isfinite(progress[k]) for k in ['loss','grad_norm'] if k in progress)
        out['arms'][arm]=row
    keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_sha256']
    out['common_epoch_checks']={}
    for arm in ['M','Q']:
        rows=[]
        for epoch in range(1,out['arms'][arm]['completed_epochs']+1):
            a=read(ROOT/'arm_B'/('epoch%02d_input_flow.json'%epoch))
            b=read(ROOT/('arm_'+arm)/('epoch%02d_input_flow.json'%epoch))
            if a and b:rows.append({'epoch':epoch,'all_pass':all(a[k]==b[k] for k in keys)})
        out['common_epoch_checks'][arm+'_vs_B']=rows
    capacity=read(ROOT/'capacity_result.json') or {}
    out['observed_counters_lower_bounds']={
        'model_forwards':capacity.get('full_model_forwards',0)+sum((r['result'] or r['progress'] or {}).get('full_model_forwards',0) for r in out['arms'].values()),
        'formal_optimizer_updates':sum((r['result'] or r['progress'] or {}).get('optimizer_updates',0) for r in out['arms'].values()),
        'capacity_optimizer_updates':capacity.get('optimizer_updates',0),
        'formal_epoch_sum':sum(r['completed_epochs'] for r in out['arms'].values())}
    if args.verify_freeze:
        protocol=read(ROOT/'protocol_FROZEN.json')
        checks={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['files_sha256'].items()}
        out['frozen_file_checks']={'count':len(checks),'all_pass':all(checks.values()),'failures':[f for f,v in checks.items() if not v]}
        out['protocol_sha256']=hashlib.sha256((ROOT/'protocol_FROZEN.json').read_bytes()).hexdigest()
    print(json.dumps(out,allow_nan=False))
