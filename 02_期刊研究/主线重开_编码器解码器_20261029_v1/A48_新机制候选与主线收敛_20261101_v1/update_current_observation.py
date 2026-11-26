"""Update a mutable progress pointer while retaining immutable SSH receipts."""
from pathlib import Path
import argparse,json

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt');args=p.parse_args()
    source=ROOT/args.receipt
    raw=json.loads(source.read_text(encoding='utf-8'))
    assert raw['status']=='EXECUTED' and raw['exit']==0
    observed=json.loads(raw['stdout'])
    assert observed['observation_only']
    live=[]
    for arm,row in observed['arms'].items():
        ident=row['identity'] or {}
        if ident.get('present') and ident.get('same_start_ticks') and ident.get('owned_command'):
            live.append({'arm':arm,'pid':ident['pid']})
    pairs={}
    for left,right in [('M','B'),('Q','B'),('Q','M')]:
        lm=observed['arms'][left]['evaluations'];rm=observed['arms'][right]['evaluations']
        pairs[left+'_minus_'+right]={e:{k:lm[e][k]-rm[e][k] for k in ['AP','APs','APs75']}
                                    for e in sorted(set(lm)&set(rm),key=int)}
    pointer={'goal_status':'active','mainline_confirmed':False,'mutable_reference_only':True,
       'require_fresh_live_poll_before_action':True,'last_observation':str(source),
       'last_observed_server_unix':observed['observed_server_unix'],
       'remote_root':'/root/uav_a48_20261101_v1','bridge_exec_session_if_alive':85299,
       'observer_command':'/opt/conda/envs/mamba/bin/python /root/uav_a48_20261101_v1/observe_A48.py',
       'owned_live_arms':live,'snapshot':observed,'same_epoch_deltas_pp':pairs,
       'next_action':'Keep B/M/Q each12 fixed. Review matched endpoint results; do not restart/retune or complete broad goal on a negative alone.'}
    (ROOT/'CURRENT_A48.json').write_text(json.dumps(pointer,indent=2,ensure_ascii=False),encoding='utf-8')
    compact={a:{'status':(r['result'] or r['progress'] or {}).get('status','NOT_STARTED'),
                'completed_epochs':r['completed_epochs'],
                'latest_progress':{k:v for k,v in (r['progress'] or {}).items() if k in ['status','epoch','microbatch','optimizer_updates','loss','grad_norm','mixture']},
                'evaluations':{e:{k:m[k] for k in ['AP','APs','APs75']} for e,m in r['evaluations'].items()}}
             for a,r in observed['arms'].items()}
    print(json.dumps({'classification':'verified_wait' if live else 'inspect_terminal_or_unresolved',
                     'live':live,'arms':compact,'same_epoch_deltas_pp':pairs,
                     'counters':observed['observed_counters_lower_bounds']}))
