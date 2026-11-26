"""Run only after the owned supervisor and both arms are terminal. No model import."""
from pathlib import Path
import hashlib,json,time
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()
def own_process(identity):
    p=Path('/proc',str(identity['pid']))
    if not p.exists():return {'owned_live':False,'pid':identity['pid'],'reason':'missing'}
    raw=(p/'stat').read_text().split(') ',1)[1].split()
    match=int(raw[19])==int(identity['start_ticks'])
    return {'owned_live':match and raw[0] not in ('Z','X'),'pid':identity['pid'],'identity_matches':match,'state':raw[0]}


def main():
    results=load(ROOT/'three_arm_result.json');assert results['status']=='COMPLETED'
    supervisor=load(ROOT/'supervisor_progress.json');assert supervisor['status']=='COMPLETED'
    processes={'supervisor':own_process(load(ROOT/'supervisor_identity.json'))}
    for arm in ['M','Q']:
        folder=ROOT/('arm_'+arm);r=load(folder/'result.json')
        assert r['status']=='COMPLETED' and r['epochs']==12 and r['optimizer_updates']==9708 and r['full_model_forwards']==10159
        assert load(folder/'exit.json')['exit_code']==0
        processes[arm]=own_process(load(folder/'child_identity.json'))
    assert not any(x['owned_live'] for x in processes.values()),processes
    protocol=ROOT/'protocol_FROZEN_v1.json'
    assert sha(protocol)=='a301ec5fb1f6a5c4675ef129f11e8359fa46c5d49640f636ccbeb343703cb8e8'
    plan=load(protocol)
    local={n:sha(ROOT/n)==h for n,h in plan['files_sha256'].items()}
    external={p:sha(Path(p))==h for p,h in plan['external_sha256'].items()}
    assert all(local.values()) and all(external.values())
    result={'status':'PASS','phase':'A51_TERMINAL_ONLY','observed_server_unix':time.time(),
       'protocol_sha256':sha(protocol),'owned_processes':processes,'owned_jobs_live':False,
       'local_frozen_files':local,'external_frozen_inputs':external,
       'new_detector_forwards':0,'new_optimizer_updates':0,
       'scope':'Terminal process identity and all protocol-named frozen input hashes; no model/evaluator execution'}
    p=ROOT/'terminal_frozen_input_audit.json';assert not p.exists();p.write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'PASS','local_files':len(local),'external_inputs':len(external),'owned_jobs_live':False,'new_model_ops':0}))


if __name__=='__main__':main()
