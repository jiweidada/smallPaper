"""Derive the candidate config from the actual running formal seed2 baseline."""
import copy
import json
from pathlib import Path
import sys
import yaml
from screen_runtime import canonical_cfg, file_sha, source_manifest


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for k in sorted(set(a)|set(b)):
            p=f'{path}.{k}' if path else k
            out.extend([p] if k not in a or k not in b else differences(a[k],b[k],p))
        return out
    return [] if a==b else [path]


def main():
    root=Path(sys.argv[1]); data='/24085403014/DEIM/dataset_visdrone'
    baseline=json.loads((root/'baseline132_args_snapshot.json').read_text())['yaml_cfg']
    baseline.pop('__include__',None)
    cfg=copy.deepcopy(baseline)
    for k in ('config','tuning','resume','test_only','mode','print_method','print_rank','local_rank'):
        cfg.pop(k,None)
    assert cfg['epoches']==132 and cfg['flat_epoch']==64 and cfg['no_aug_epoch']==12
    assert cfg['train_dataloader']['dataset']['transforms']['policy']['epoch']==[4,64,120]
    assert cfg['train_dataloader']['collate_fn']['mixup_epochs']==[4,64]
    assert cfg['train_dataloader']['collate_fn']['stop_epoch']==120
    assert cfg['seed']==2 and cfg['train_dataloader']['total_batch_size']==cfg['val_dataloader']['total_batch_size']==4
    assert cfg['evaluator']['using_tiny_metrice'] is False and cfg['evaluator']['max_dets']==100
    assert cfg['HGNetv2']['pretrained'] is False
    cfg['output_dir']=str(root/'p2_seed2/run')
    cfg['model']='P2CandidateDEIM'
    cfg['P2CandidateDEIM']=dict(cfg['DEIM'],c2_channels=64,p2_budget=60,anchor_wh_px=8.2)
    cfg['profiling_rng_mode']='native_reference_replay'
    diff=differences(canonical_cfg(baseline),canonical_cfg(cfg))
    assert diff==['P2CandidateDEIM','model','profiling_rng_mode'],diff
    (root/'p2_full132.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (root/'baseline132_reference.yml').write_text(yaml.safe_dump(baseline,sort_keys=False))
    sources=source_manifest(root/'code')
    receipt={'epoches':132,'seed':2,'flat_epoch':64,'no_aug_epoch':12,
             'policy':[4,64,120],'mixup_epochs':[4,64],'stop_epoch':120,
             'config_differences':diff,'native_profile_RNG_replayed':True,
             'full_config_sha256':file_sha(root/'p2_full132.yml'),
             'source_digest':__import__('hashlib').sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
             'source_manifest':sources,'baseline_args_sha256':file_sha(root/'baseline132_args_snapshot.json')}
    (root/'full132_configuration_alignment.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_manifest'}),flush=True)


if __name__=='__main__': main()
