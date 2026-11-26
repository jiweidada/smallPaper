"""Consume six completed savedEMA terminal archives with their frozen exact readers."""
from pathlib import Path
import argparse,base64,hashlib,json,subprocess,sys
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def main(prepare=False,arm=None):
    v=read(ROOT/'outputs/stem_fixed12_seed2_formal_collection12_20261011_v1_inputs.json');raw=v['payload'].encode()
    assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256']
    captured=json.loads(raw);assert captured['status']=='SIX_EXISTING_FIXED12_FORMAL_COLLECTION_METADATA_RECEIVED' and len(captured['jobs'])==6
    root=ROOT/'outputs/stem_fixed12_seed2_formal_collection12_20261011_v1'
    if prepare:
        for name,item in captured['jobs'].items():
            b=base64.b64decode(item['metadata']['base64']);assert len(b)==item['metadata']['bytes'] and hashlib.sha256(b).hexdigest()==item['metadata']['sha256']
            with (root/'downloads'/('terminal_'+name+'_seed2_v1_metadata.json')).open('xb') as f:f.write(b)
        with (root/'remote_metadata_capture.json').open('xb') as f:f.write(raw)
        write(root/'remote_metadata_transport.json',v['transport']);print('PASS_SIX_BYTE_EXACT_REMOTE_METADATA_FILES');return
    assert arm in captured['jobs']
    names={'original_stem':'original_stem12_monitor_20261011_v1','native_stem_scratch':'native_stem_scratch12_monitor_20261011_v2'}
    reader=ROOT/'英文小论文/code'/names.get(arm,'stem_component12_monitor_20261011_v1')/'verify_terminal_local.py'
    archive=root/'downloads'/('terminal_'+arm+'_seed2_v1.zip');metadata=archive.with_name(archive.stem+'_metadata.json');meta=read(metadata)
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    out=root/('seed2_'+arm);out.mkdir(exist_ok=False)
    cmd=[sys.executable,'-X','utf8','-B',str(reader),'--archive',str(archive),'--metadata',str(metadata),'--destination',str(out/'extracted'),'--report',str(out/'verification.json')]
    write(out/'reader_dispatch.json',dict(command=cmd,reader_sha256=sha(reader),reader_protocol_sha256=sha(reader.parent/'local_reader_protocol.json'),new_training_epochs=0,new_model_forwards=0))
    process=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8');(out/'reader.log').write_text(process.stdout,encoding='utf8')
    write(out/'reader_exit.json',dict(exit_code=process.returncode,reader_sha256=sha(reader),new_training_epochs=0,new_model_forwards=0))
    assert process.returncode==0,process.stdout
    proof=read(out/'verification.json');assert proof['identity']['arm']==arm and proof['identity']['seed']==2 and proof['all548_stage_arrays_reloaded_exact']
    assert proof['verification_script_sha256']==sha(reader) and not proof['native_gate_passed_this_seed']
    print(json.dumps(dict(arm=arm,status=proof['status'],delta=proof['delta_AP_metrics_pp'],saved548_EMA_arrays_exact=True)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm');a=p.parse_args();main(a.prepare,a.arm)
