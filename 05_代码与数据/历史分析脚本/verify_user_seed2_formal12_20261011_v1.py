"""Persist exact metadata and run the frozen original-user formal terminal reader."""
from pathlib import Path
import argparse,base64,hashlib,json,subprocess,sys
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def main(prepare=False,arm=None):
    v=read(ROOT/'outputs/user_seed2_formal_collection12_20261011_v1_inputs.json');raw=v['payload'].encode()
    assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256'];captured=json.loads(raw)
    assert set(captured['jobs'])=={'p2_skip','backbone_silu','backbone_attention'}
    for a,j in captured['jobs'].items():
        assert j['candidate_reload_v1/verification.json']['status']=='PASS_SAVED_CANDIDATE_RELOAD' and j['candidate_reload_v1/verification.json']['all548_stage_arrays_exact'] and j['candidate_reload_v1/verification.json']['native_GT_hits_exact'] and j['candidate_reload_v1/verification.json']['stats_max_abs_fraction_difference']==0
        assert j['candidate_reload_v1/completion.json']['exit_code']==0 and j['post_training/completion.json']['status']=='STEM_FIXED12_RELOAD_AND_COLLECTION_FINISHED'
        assert 'post_training/failure.json' not in j
    root=ROOT/'outputs/user_seed2_formal_collection12_20261011_v1'
    if prepare:
        assert set(captured['metadata'])=={'terminal_'+a+'_seed2_v1_metadata.json' for a in captured['jobs']}
        for n,item in captured['metadata'].items():
            b=base64.b64decode(item['base64']);assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
            with (root/'downloads'/n).open('xb') as f:f.write(b)
        with (root/'remote_capture.json').open('xb') as f:f.write(raw)
        write(root/'remote_transport.json',v['transport']);print('PASS_THREE_USER_BYTE_EXACT_FORMAL_METADATA');return
    assert arm in captured['jobs'];archive=root/'downloads'/('terminal_'+arm+'_seed2_v1.zip');metadata=archive.with_name(archive.stem+'_metadata.json');meta=read(metadata)
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    reader=ROOT/'英文小论文/code/user_backbone_formal_verify12_20261011_v1/verify.py';out=root/('seed2_'+arm);out.mkdir(exist_ok=False)
    cmd=[sys.executable,'-X','utf8','-B',str(reader),'--seed','2','--arm',arm,'--archive',str(archive),'--metadata',str(metadata),'--destination',str(out/'extracted'),'--report',str(out/'verification.json')]
    write(out/'reader_dispatch.json',dict(command=cmd,reader_sha256=sha(reader),protocol_sha256=sha(reader.parent/'protocol.json'),new_model_forwards=0,new_training_epochs=0))
    process=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8');(out/'reader.log').write_text(process.stdout,encoding='utf8')
    write(out/'reader_exit.json',dict(exit_code=process.returncode,reader_sha256=sha(reader),new_training_epochs=0,new_model_forwards=0));assert process.returncode==0,process.stdout
    proof=read(out/'verification.json');assert proof['status']=='PASS_USER_FORMAL_FIXED12_PRIMARY_AND_SAVED_EMA_EXACT' and proof['independent_saved_EMA_replay_verified'] and proof['formal_terminal_reader_passed']
    assert proof['native_gate_passed_this_seed']==(arm=='p2_skip')
    print(json.dumps(dict(arm=arm,status=proof['status'],delta=proof['delta_AP_metrics_pp'],gate=proof['native_gate_passed_this_seed'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm');a=p.parse_args();main(a.prepare,a.arm)
