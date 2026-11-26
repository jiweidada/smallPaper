"""Prepare a new fixed12 FDR candidate from verified historical data/recipe."""
import argparse
import ast
from pathlib import Path
import sys
from runtime import HERE,COMMON_INIT_SHA,read,write,sha,implementation_hashes,resolved_config,verify_source,verify_bundle


def prepare(prefix,seed):
    from evaluator_binding import fingerprint
    protocol=read(HERE/'training_protocol.json')
    assert seed in protocol['execution_allowlist']['support_cost']
    root=Path(prefix)/'DEIM/experiments/support_matching12_20261011_v1'
    assert HERE.resolve()==(root/'code').resolve()
    old=Path(prefix)/'DEIM/experiments/fine_residual_screen24_20261009_v2'
    legacy=old/f'bundles/baseline_seed{seed}'
    original=read(legacy/'receipt.json'); old_plan=read(legacy/'plan.json')
    assert original['identity']['arm']=='baseline' and original['identity']['seed']==seed
    assert sha(legacy/'plan.json')==original['identity']['plan_sha256']
    assert sha(legacy/'resolved.yml')==original['resolved_sha256']
    assert all(sha(old/'code'/n)==h for n,h in original['identity']['implementation'].items())
    assert all(sha(HERE/n)==original['identity']['implementation'][n] for n in ('geometry.py','residual.py'))
    screen=root/f'seed{seed}_support_cost'; bundle=screen/'bundle'
    assert not screen.exists(), 'No existing configuration may be restarted or overwritten'
    paths={k:Path(v) for k,v in original['paths'].items()}
    paths.update(experiment_root=root,screen_root=screen,output=screen/'run',legacy_bundle=legacy,
        baseline_checkpoint=Path(original['paths']['output'])/'checkpoint0011.pth',
        baseline_first100=Path(original['paths']['output'])/'first100_training_inputs.jsonl',
        baseline_cached=Path(protocol['baseline_reuse'][str(seed)]['source']),
        image_manifest=bundle/'image_manifest.json',image_verification=bundle/'image_verification.json')
    reference=protocol['references'][str(seed)]
    for key in ('train_ann','val_ann','common_init','baseline_checkpoint','baseline_first100'):
        assert sha(paths[key])==reference[key+'_sha256'],key
    assert sha(paths['common_init'])==COMMON_INIT_SHA
    for name,digest in protocol['baseline_reuse'][str(seed)]['pinned_files'].items():
        assert sha(paths['baseline_cached']/name)==digest,name
    verify_source(paths['native_root'])
    evaluator=fingerprint(paths['evaluator_root']); assert evaluator==reference['evaluator_binding']
    manifest_source=legacy/'image_manifest.json'
    if not manifest_source.exists():
        manifest_source=Path(prefix)/f'DEIM/experiments/scale_protection12_20261010_v1/image_manifest_seed{seed}.json'
    assert sha(manifest_source)==reference['image_manifest_sha256']
    manifest=read(manifest_source); checked=count_bytes=0
    for split,count in (('train',6471),('val',548)):
        entries=manifest[split]; assert len(entries)==count
        images=paths[split+'_images']
        assert {p.name for p in images.glob('*.jpg')}==set(entries)
        for name,item in entries.items():
            path=images/name
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
            checked+=1; count_bytes+=item['bytes']
    assert checked==7019
    summary=read(paths['baseline_cached']/'endpoint_epoch12/summary.json')
    assert summary['identity']['seed']==seed and summary['identity']['arm']=='baseline'
    assert summary['epoch_zero_based']==11 and summary['evaluated_weights']=='EMA'
    for key in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','evaluator_binding'):
        assert summary['identity'][key]==reference[key],key
    plan=dict(schema=protocol['schema'],arm='support_cost',seed=seed,matching_coefficient=2.0,
        historical_resolved_config=read(legacy/'resolved.yml'),model_recipe=old_plan['model_recipe'],
        legacy_receipt_sha256=sha(legacy/'receipt.json'),legacy_plan_sha256=sha(legacy/'plan.json'),
        historical_baseline_stats=summary['native_stats'],max_epochs=12,recipe_epochs=132,
        fixed_endpoint='native_epoch12_EMA',extra_parameters=0,
        scope='Only ordinary matching costs change; prediction grid, native augmentation/loss weights/schedule remain; no new baseline forward.')
    cfg=resolved_config(plan,paths)
    for file in HERE.glob('*.py'):
        ast.parse(file.read_text(encoding='utf8'))
    screen.mkdir(); bundle.mkdir()
    paths['image_manifest'].write_bytes(manifest_source.read_bytes())
    write(paths['image_verification'],dict(status='ALL_CURRENT_IMAGES_MATCH_FROZEN_DATASET',files_verified=checked,
        bytes_verified=count_bytes,image_manifest_sha256=sha(paths['image_manifest']),
        new_model_forwards=0,new_training_updates=0))
    write(bundle/'plan.json',plan); write(bundle/'resolved.yml',cfg)
    identity=dict(schema=plan['schema'],arm=plan['arm'],seed=seed,matching_coefficient=2.0,
        plan_sha256=sha(bundle/'plan.json'),implementation=implementation_hashes(),
        protocol_sha256=sha(HERE/'training_protocol.json'),common_init_sha256=COMMON_INIT_SHA,
        evaluator_binding=evaluator,val_annotation_sha256=sha(paths['val_ann']),
        train_annotation_sha256=sha(paths['train_ann']),image_manifest_sha256=sha(paths['image_manifest']),
        image_verification_sha256=sha(paths['image_verification']),native_parameter_count=10231997,
        added_parameter_count=0,baseline_training_epochs_added=0,baseline_model_forwards_added=0)
    for key in ('train_ann','val_ann','baseline_checkpoint','baseline_first100'):
        identity[key+'_sha256']=sha(paths[key])
    receipt=dict(status='PREPARED_NOT_TRAINED',identity=identity,
        paths={key:str(value) for key,value in paths.items()},resolved_sha256=sha(bundle/'resolved.yml'))
    write(bundle/'receipt.json',receipt)
    assert 'torch' not in sys.modules
    verify_bundle(bundle)
    print(__import__('json').dumps(dict(status=receipt['status'],seed=seed,bundle=str(bundle),
        current_images_verified=7019,model_forwards=0,optimizer_updates=0,new_training_epochs=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    a=p.parse_args();prepare(a.prefix,a.seed)
