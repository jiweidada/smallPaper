"""Compare completed log epochs to existing same-seed baseline logs, without models."""
import argparse,json,sys
from pathlib import Path
from verify_terminal_local import ROOT,CORE,local_protocol,read,sha,write
p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
o=read(a.observation);seed=o['seed'];assert seed in (1,2) and not o['read_errors']
protocol=local_protocol();ev=protocol['reused_local_evidence'][str(seed)]
old=ROOT/ev['packet_relative_path'];report=ROOT/ev['report_relative_path']
assert sha(report)==ev['report_sha256'] and sha(old/'manifest.json')==ev['manifest_sha256']
assert read(report)['identity']['seed']==seed
item=read(old/'manifest.json')['files']['reference/baseline_log.txt'];path=old/'reference/baseline_log.txt'
assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
baseline=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x]
assert [r['epoch'] for r in baseline]==list(range(24))
assert o['protocol_sha256']==sha(CORE/'training_protocol.json')
candidate=o['metrics'];assert [r['epoch'] for r in candidate]==list(range(len(candidate))) and len(candidate)<=12
rows=[]
for r in candidate:
    e=r['epoch'];b=baseline[e]['test_coco_eval_bbox'];c=r['test_coco_eval_bbox'];assert len(b)==len(c)==12
    rows.append(dict(epoch_one_based=e+1,baseline_AP=100*b[0],candidate_AP=100*c[0],delta_AP_pp=100*(c[0]-b[0]),
        baseline_APs=100*b[3],candidate_APs=100*c[3],delta_APs_pp=100*(c[3]-b[3])))
result=dict(status='PROVISIONAL_SAME_EPOCH_NATIVE_LOG_COMPARISON',seed=seed,rows=rows,observed_remote_UTC=o['observed_remote_UTC'],
    historical_log_sha256=sha(path),observation_sha256=sha(a.observation),completed_epochs=len(candidate),
    fixed12_saved_EMA_verification_passed=False,new_model_forwards=0,new_training_commands=0,
    interpretation='Same seed/epoch log comparison only. Final decision requires fixed12 saved-EMA verification; early positive values are not an efficacy conclusion.')
assert 'torch' not in sys.modules
write(a.output,result);print(json.dumps(dict(seed=seed,latest=rows[-1] if rows else None,fixed12_verified=False)))
