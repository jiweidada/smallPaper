"""Snapshot existing first-epoch audit files; no model imports or training writes."""
from pathlib import Path
import sys,json,hashlib,zipfile,datetime,base64,textwrap


def main(seed):
    prefix={1:'/25085400120',2:'/24085403014'}[seed];version={1:4,2:3}[seed]
    root=Path(prefix)/f'DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}'
    screen=root/f'seed{seed}_fine';run=screen/'run';bundle=root/f'prepared_seed{seed}'
    read=lambda p:json.loads(p.read_text())
    rows=[json.loads(s) for s in (run/'training_audit_epochs.jsonl').read_text().splitlines()]
    epoch0=[r for r in rows if r['epoch_zero_based']==0];assert len(epoch0)==1
    epoch0=epoch0[0];assert epoch0['attempts_total']==1617 and epoch0['EMA_updates']==1617
    prefixes=('encoder.geometry_field.','decoder.enc_bbox_head.geometry_proposal.',
              'decoder.decoder.geometry_layers.','decoder.decoder.geometry_attention.')
    changed=epoch0['added_changed_elements_from_initial']
    assert all(any(v>0 for n,v in changed.items() if n.startswith(p)) for p in prefixes)
    receipt=read(bundle/'receipt.json');reference=Path(receipt['paths']['baseline_first100'])
    actual=[json.loads(s) for s in (run/'first100_training_inputs.jsonl').read_text().splitlines()]
    expected=[json.loads(s) for s in reference.read_text().splitlines()]
    assert len(actual)==100 and actual==expected
    files={}
    for p in (root/'code').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:files['code/'+p.relative_to(root/'code').as_posix()]=p
    for n in ('receipt.json','plan.json','resolved.yml','image_manifest.json','baseline_cache_check.json'):
        p=bundle/n
        if p.exists():files['prepared/'+n]=p
    for n in ('controller_start.json','candidate_start.json','baseline_completion.json','baseline_reuse.json'):
        p=screen/n
        if p.exists():files['screen/'+n]=p
    for n in ('runtime_start.json','args.json','first100_training_inputs.jsonl','first100_parity.json',
              'first8_optimizer_steps.json','training_audit_epochs.jsonl','log.txt','training_audit_live.json'):
        p=run/n
        if p.exists():files['run/'+n]=p
    files['reference/baseline_first100.jsonl']=reference
    files['reference/baseline_log.txt']=reference.with_name('log.txt')
    files['snapshot/candidate.log']=screen/'candidate.log'
    archive=root/f'early_audit_seed{seed}.zip';assert not archive.exists()
    captured={};manifest={'status':'LIVE_FIRST_EPOCH_ENGINEERING_EVIDENCE_NOT_FIXED12',
        'seed':seed,'runtime_version':version,'captured_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_root':str(root),'new_model_forwards':0,'new_evaluator_calls':0,'new_optimizer_updates':0,
        'notes':'Live logs are captured bytes at collection time, not terminal artifacts. Step8 raw parameter tensors remain on server and belong in final collection.',
        'files':{}}
    for name,p in files.items():
        b=p.read_bytes();captured[name]=b
        manifest['files'][name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'source':str(p)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,b in captured.items():z.writestr(name,b)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    meta={'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
          'bytes':archive.stat().st_size,'files':len(captured)+1,'seed':seed,'new_training_updates':0}
    with (root/f'early_audit_seed{seed}_metadata.json').open('x') as f:json.dump(meta,f,indent=2)
    print('\nB64_BEGIN\n'+'\n'.join(textwrap.wrap(base64.b64encode(json.dumps(meta).encode()).decode(),76))+'\nB64_END')


if __name__=='__main__':main(int(sys.argv[1]))
