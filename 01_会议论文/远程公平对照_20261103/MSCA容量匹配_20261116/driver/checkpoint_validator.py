"""CPU-only saved-payload inspection; no model construction or inference."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys

def tensor_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(json.dumps([key,str(value.dtype),list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,type=Path)
    p.add_argument('--protocol',required=True,type=Path)
    p.add_argument('--torch-site',type=Path)
    a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('fresh_payload_protocol',here/'backup_protocol.py')
    protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    assert protocol.file_sha(a.protocol)==protocol.PROTOCOL
    stage=a.stage.resolve(strict=True)
    manifest_data=(stage/'manifest.json').read_bytes()
    raw=protocol.decode(manifest_data)
    manifest=protocol.parse_manifest(manifest_data,raw['run_id'],raw['epoch'])
    protocol.check_metric_files(manifest,lambda name:protocol.local_metadata(stage/name))
    for item in manifest['files']:
        path=stage/item['name']
        assert path.stat().st_size==item['bytes'] and protocol.file_sha(path)==item['sha256']
    dll_handles=[]
    if a.torch_site is not None:
        site=a.torch_site.resolve(strict=True)
        sys.path.insert(0,str(site))
        if os.name=='nt':
            for folder in [site/'torch/lib',Path(sys.base_prefix)/'DLLs']:
                if folder.is_dir():dll_handles.append(os.add_dll_directory(str(folder)))
    import torch
    torch.set_num_threads(4)
    checked=[]
    if manifest['epoch']==-1:
        initial=torch.load(stage/'initial.pt',map_location='cpu',weights_only=True)
        pre=protocol.decode((stage/'preflight.json').read_bytes())
        assert tensor_hash(initial)==pre['initial_tensor_sha256']==pre['initial_ema_tensor_sha256']
        assert pre['optimizer_updates']==pre['initial_ema_updates']==pre['full_validation_passes']==0
        checked.append({'role':'initial','tensor_sha256':pre['initial_tensor_sha256']})
    else:
        epoch=manifest['epoch']
        row=protocol.decode((stage/'epoch.metrics.json').read_bytes())
        candidates=[('next_epoch.pt',epoch,manifest['next_ema_tensor_sha256'],'next'),
                    ('evaluated_epoch.pt',epoch,row['ema_tensor_sha256'],'evaluated')]
        for ref in (manifest['best'],manifest['stage_one']):
            name=f"selected/epoch{ref['epoch']:04d}/best.pt"
            selected_row=protocol.decode((stage/f"selected/epoch{ref['epoch']:04d}/best.metrics.json").read_bytes())
            candidates.append((name,ref['epoch'],selected_row['ema_tensor_sha256'],'selected'))
        if epoch==319:
            candidates.append(('terminal_epoch0319.pt',319,row['ema_tensor_sha256'],'terminal'))
        seen={}
        for name,index,expected_ema,role in candidates:
            path=stage/name;file_sha=protocol.file_sha(path)
            if file_sha in seen:
                assert seen[file_sha]['epoch']==index and seen[file_sha]['ema_tensor_sha256']==expected_ema
                checked.append({'role':role,'file':name,'file_sha256':file_sha,'same_payload_as':seen[file_sha]['file']})
                continue
            value=torch.load(path,map_location='cpu',weights_only=True)
            assert value['arm']==protocol.ARM and value['protocol_sha256']==protocol.PROTOCOL
            assert value['configuration_sha256']==protocol.CONFIG and value['actual_loop_epoch']==index
            assert value['optimizer_step_attempts']==(index+1)*1617
            assert all(key in value for key in ['model','ema','optimizer','criterion','postprocessor','lr_warmup_scheduler','rng_state','ema_decay_attribute'])
            assert all(key in value['optimizer'] for key in ['state','param_groups'])
            assert len(value['optimizer']['param_groups'])==4
            rng=value['rng_state']
            assert set(rng)=={'python','numpy','torch_cpu','torch_cuda_all','train_loader','train_sampler','validation_loader'}
            assert isinstance(rng['python'],tuple) and len(rng['python'])==3
            assert rng['numpy'][0]=='MT19937' and len(rng['numpy'][1])==624 and 0<=rng['numpy'][2]<=624
            for key in ['torch_cpu','train_loader','train_sampler','validation_loader']:
                assert torch.is_tensor(rng[key]) and rng[key].dtype==torch.uint8 and rng[key].ndim==1 and rng[key].numel()>0
            assert len(rng['torch_cuda_all'])==1 and all(torch.is_tensor(v) and v.dtype==torch.uint8 for v in rng['torch_cuda_all'])
            actual_ema=tensor_hash(value['ema']['module'])
            assert actual_ema==expected_ema
            if role=='next':
                assert value['resume_format']=='fresh_pair_complete_post_policy_epoch_v1'
                assert value['next_epoch']==index+1
                assert value['best']==manifest['best'] and value['stage_one']==manifest['stage_one']
                assert value['latest_evaluated_row']['metrics']==row['metrics']
            entry={'role':role,'file':name,'file_sha256':file_sha,'epoch':index,'ema_tensor_sha256':actual_ema,'complete_state_and_rng_fields_verified':True}
            seen[file_sha]=entry;checked.append(entry)
            del value
    report={'status':'CHECKPOINT_PAYLOADS_VERIFIED_CPU_ONLY','manifest_sha256':protocol.digest(manifest_data),
            'protocol_sha256':protocol.PROTOCOL,'validator_sha256':protocol.file_sha(Path(__file__)),
            'arm':protocol.ARM,'epoch':manifest['epoch'],'checked_payloads':checked,'model_forwards':0,'evaluator_runs':0,'optimizer_updates':0}
    out=stage/'payload_validation.json'
    assert not out.exists(),'Do not overwrite an existing payload validation receipt'
    protocol.write_atomic(out,protocol.encode(report))
    print(json.dumps({'status':report['status'],'epoch':manifest['epoch'],'checked_roles':len(checked)}),flush=True)

if __name__=='__main__':main()
