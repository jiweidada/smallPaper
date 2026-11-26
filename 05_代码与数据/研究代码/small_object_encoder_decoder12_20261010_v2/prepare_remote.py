"""Bind current remote resources/files without Torch, optimizer or training."""
from pathlib import Path
import argparse,hashlib,json,time
from runtime import HERE,COMMON_INIT_SHA,read,write,sha,implementation_hashes,verify_source,resolved_config
def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=[1,2],required=True);a=p.parse_args()
    assert a.prefix=={1:'/25085400120',2:'/24085403014'}[a.seed]
    prefix=Path(a.prefix);experiment=HERE.parent;plan_path=HERE/f'configs/fine_seed{a.seed}.json';plan=read(plan_path)
    assert plan['seed']==a.seed and plan['variant']=='fine' and plan['screen_protocol_sha256']==sha(HERE/'protocol.json')
    source=prefix/'DEIM/experiments/fine_residual_screen24_20261009_v2/native';verify_source(source)
    old=prefix/f'DEIM/experiments/fine_residual_screen24_20261009_v2/runs/baseline_seed{a.seed}'
    original=read(old/'args.json')['yaml_cfg'];expected=plan['native_baseline_config'];allowed={'ResidualGeometryDEIM','model','output_dir','use_swanlab','__include__','config'}
    assert all(original.get(k)==expected.get(k) for k in set(original)|set(expected) if k not in allowed)
    assert original['ResidualGeometryDEIM']['arm']=='baseline'
    paths={'native_root':source,'common_init':Path(plan['common_init']),'evaluator_root':Path(plan['evaluator_root']),'baseline_checkpoint':old/'checkpoint0011.pth','baseline_first100':old/'first100_training_inputs.jsonl','screen_root':experiment/f'seed{a.seed}_fine'}
    for split in ('train','val'):
        ds=expected[split+'_dataloader']['dataset'];paths[split+'_images']=Path(ds['img_folder']);paths[split+'_ann']=Path(ds['ann_file'])
    paths['output']=paths['screen_root']/'run'
    assert sha(paths['common_init'])==COMMON_INIT_SHA and sha(paths['baseline_checkpoint'])==plan['baseline_checkpoint_sha256']
    assert sha(paths['baseline_first100'])==plan['baseline_first100_sha256']
    assert sha(paths['train_ann'])=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
    assert sha(paths['val_ann'])=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    row=[json.loads(x) for x in (old/'log.txt').read_text().splitlines() if json.loads(x)['epoch']==11]
    assert len(row)==1 and row[0]['test_coco_eval_bbox']==plan['historical_baseline_stats']
    from evaluator_binding import fingerprint
    evaluator=fingerprint(paths['evaluator_root']);assert evaluator==plan['evaluator_binding']
    bundle=experiment/f'prepared_seed{a.seed}';assert not bundle.exists() and not paths['screen_root'].exists();bundle.mkdir();paths['screen_root'].mkdir()
    started=time.monotonic();images={}
    for split,count in (('train',6471),('val',548)):
        data=read(paths[split+'_ann']);assert len(data['images'])==count
        images[split]={}
        for i,image in enumerate(sorted(data['images'],key=lambda x:x['id'])):
            path=paths[split+'_images']/image['file_name'];assert path.is_file()
            images[split][image['file_name']]={'bytes':path.stat().st_size,'sha256':sha(path)}
            if (i+1)%1000==0:print(json.dumps({'stage':'image_identity','split':split,'files':i+1}),flush=True)
    write(bundle/'image_manifest.json',images)
    identity={'schema':plan['schema'],'arm':plan['variant'],'seed':a.seed,'plan_sha256':sha(plan_path),'protocol_sha256':sha(HERE/'protocol.json'),'implementation':implementation_hashes(),'common_init_sha256':COMMON_INIT_SHA,'train_ann_sha256':sha(paths['train_ann']),'val_ann_sha256':sha(paths['val_ann']),'common_init_sha256':sha(paths['common_init']),'baseline_checkpoint_sha256':sha(paths['baseline_checkpoint']),'baseline_first100_sha256':sha(paths['baseline_first100']),'val_annotation_sha256':sha(paths['val_ann']),'train_annotation_sha256':sha(paths['train_ann']),'image_manifest_sha256':sha(bundle/'image_manifest.json'),'evaluator_binding':evaluator,'native_source_files':468,'preparation_code_sha256':sha(__file__)}
    config=resolved_config(plan,paths);write(bundle/'resolved.yml',config);(bundle/'plan.json').write_bytes(plan_path.read_bytes())
    write(bundle/'receipt.json',{'status':'PREPARED_NOT_TRAINED','identity':identity,'paths':{k:str(v) for k,v in paths.items()},'resolved_sha256':sha(bundle/'resolved.yml'),'image_files':{'train':6471,'val':548},'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'status':'PREPARED','bundle':str(bundle),'image_files':7019,'optimizer_steps':0,'epochs_added':0}),flush=True)
if __name__=='__main__':main()
