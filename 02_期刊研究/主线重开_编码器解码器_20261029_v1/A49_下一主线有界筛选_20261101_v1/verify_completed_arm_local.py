from pathlib import Path,PurePosixPath
import argparse,collections,hashlib,json,sys,zipfile

ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','M','Q'],required=True);a=parser.parse_args().arm
    ready=json.loads((ROOT/(a+'_bundle_ready.json')).read_text())
    bundle=ROOT/(a+'_completed_bundle_v1.zip')
    assert bundle.stat().st_size==ready['bytes'] and sha(bundle)==ready['sha256']
    dest=ROOT/('completed_'+a+'_v1');dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(bundle) as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        assert len(manifest['files'])==ready['payload_files']
        assert set(z.namelist())==set(manifest['files'])|{'bundle_manifest.json'}
        for name,expected in manifest['files'].items():
            rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
            path=(dest/Path(*rel.parts)).resolve();assert path.is_relative_to(dest.resolve())
            path.parent.mkdir(exist_ok=True,parents=True)
            h=hashlib.sha256();count=0
            with z.open(name) as src,path.open('xb') as out:
                for data in iter(lambda:src.read(1<<20),b''):out.write(data);h.update(data);count+=len(data)
            assert h.hexdigest()==expected['sha256'] and count==expected['bytes'],name
    save(dest/'bundle_manifest.json',manifest)
    folder=dest/('arm_'+a)
    result=json.loads((folder/'result.json').read_text())
    assert result['status']=='COMPLETED' and result['epochs']==12 and result['optimizer_updates']==9708
    assert json.loads((folder/'exit.json').read_text())['exit_code']==0
    assert result['protocol_sha256']=='d3a971a21d1c339052c8a7ef41fcab345882edc8893c4762c66dbbc5bc10f8f3'
    assert result['native_inference_no_mask_reads'] is True
    assert result['extra_mask_core_reads']==(38832 if a=='Q' else 0)
    metrics=json.loads((folder/'epoch12_EMA_metrics.json').read_text())
    assert result['endpoint_metrics']==metrics==ready['endpoint']
    sys.path.insert(0,str(ROOT.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
    import torch
    torch.set_num_threads(4)
    initial_path=ROOT.parent/'A43_ECDet三组蒸馏最小验证_20261031_v1/common_init_10class.pth'
    assert sha(initial_path)==result['initialization_sha256']
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)['model']
    ckpt=torch.load(folder/'epoch12.pth',map_location='cpu',weights_only=True)
    assert ckpt['epoch']==12 and ckpt['arm']==a and ckpt['optimizer_updates']==9708
    assert ckpt['protocol_sha256']==result['protocol_sha256']
    states={'raw':ckpt['model'],'EMA':ckpt['ema']['module']}
    checks={}
    for label,state in states.items():
        assert set(state)==set(initial)
        changed=0;structural=[]
        for name,tensor in state.items():
            assert tensor.shape==initial[name].shape and tensor.dtype==initial[name].dtype,name
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                assert name.endswith('.anchors') and torch.equal(tensor,initial[name]),name
                structural.append(name)
            changed+=int(not torch.equal(tensor,initial[name]))
        bn={}
        for site in ['encoder.fpn_blocks.0.cv4.norm','encoder.fpn_blocks.1.cv4.norm','encoder.pan_blocks.0.cv4.norm','encoder.pan_blocks.1.cv4.norm']:
            assert int(state[site+'.num_batches_tracked'])==9708
            bn[site]={s:float((state[site+'.'+s]-initial[site+'.'+s]).abs().max()) for s in ['weight','bias','running_mean','running_var']}
        assert changed>0 and all(v['weight']>0 or v['bias']>0 for v in bn.values())
        checks[label]={'changed_state_tensors':changed,'unchanged_structural_nonfinite_buffers':structural,'four_BN_sites':bn}
    steps=collections.Counter(int(v['step']) for v in ckpt['optimizer']['state'].values() if 'step' in v)
    assert max(steps)==9708 and min(steps)>0
    verified={'status':'PASS','arm':a,'epochs':12,'optimizer_updates':9708,'full_model_forwards':result['full_model_forwards'],
      'extra_mask_core_reads':result['extra_mask_core_reads'],'native_inference_no_mask_reads':True,
      'payload_files_verified':len(manifest['files']),'bundle_sha256':ready['sha256'],'endpoint_metrics':metrics,
      'checkpoint_state_checks':checks,'optimizer_step_histogram':dict(steps),'model_forward_operations_added':0,
      'scope':'Archive/checkpoint integrity verified on CPU; not a rerun of model predictions or efficacy proof.'}
    save(ROOT/(a+'_completed_local_verification.json'),verified)
    print(json.dumps({k:v for k,v in verified.items() if k not in ['checkpoint_state_checks']}))
