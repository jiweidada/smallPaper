"""Freeze the two prospectively specified controls after verified fine gates.

Creates a new deployment; never trains, changes old runs or calls a model.
"""
from pathlib import Path
import ast,datetime,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'英文小论文/code/small_object_encoder_decoder12_20261010_v4'
NEW=OLD.with_name('small_object_encoder_decoder12_20261010_v5')
OUT=ROOT/'outputs/small_object_encoder_decoder12_20261010_v5'
ANALYSIS=ROOT/'outputs/small_object_encoder_decoder12_analysis_20261010_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))

def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)

def replace_once(p,old,new):
    text=p.read_text(encoding='utf8');assert text.count(old)==1,(p.name,old)
    p.write_text(text.replace(old,new),encoding='utf8')

PROMOTION_CODE='''"""Read-only fine-gate binding. Does not import Torch or evaluate a model."""
from pathlib import Path
from runtime import HERE,read,sha

def verify_fine_evidence(seed):
    protocol=read(HERE/'protocol.json');folder=HERE/'promotion';index=read(folder/'index.json')
    assert sha(folder/'index.json')==protocol['fine_promotion_sha256']
    assert index['status']=='BOTH_FINE_FIXED12_RELOAD_AND_LOCAL_VERIFICATION_PASS'
    assert index['scientific_plan_sha256']==protocol['scientific_plan_sha256']
    for n in (1,2):
        entry=index['seeds'][str(n)];report=read(folder/f'seed{n}_verification_report.json')
        assert sha(folder/f'seed{n}_verification_report.json')==entry['local_verification_sha256']
        assert report['status']=='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
        assert report['seed']==n and report['files_verified']==2202
        assert report['archive_sha256']==entry['terminal_archive']['sha256']
        assert report['all548_saved_weight_stage_arrays_exact'] and report['stage_geometry_TP_conflicts']==0
        base=report['baseline_overall_percent'];fine=report['candidate_overall_percent']
        assert fine['APs']>base['APs'] and fine['AP']>=base['AP'] and report['native_gate_passed_this_seed']
        assert report['cost']['candidate_parameters']==13806625 and report['cost']['added_parameters']==3574628
        count=report['final_training_counts'];assert count['attempts_total']==count['EMA_updates']==19404
        assert 0<count['actual_optimizer_steps_total']<=19404
        done=read(folder/f'seed{n}_training_completion.json');replay=read(folder/f'seed{n}_reload_verification.json')
        assert sha(folder/f'seed{n}_training_completion.json')==entry['training_completion_sha256']
        assert sha(folder/f'seed{n}_reload_verification.json')==entry['reload_verification_sha256']
        assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12)) and done['native_gate_passed']
        assert done['identity']['arm']=='fine' and done['identity']['seed']==n
        assert done['identity']['implementation']['small_object_model.py']==protocol['prototype_model_sha256']['small_object_model.py']
        assert done['identity']['implementation']['geometry_memory.py']==protocol['prototype_model_sha256']['geometry_memory.py']
        assert replay['status']=='PASS_SAVED_CANDIDATE_RELOAD' and replay['all548_stage_arrays_exact']
        assert replay['checkpoint_sha256']==done['last_checkpoint_sha256']==entry['checkpoint_sha256']
        assert replay['optimizer_updates']==replay['epochs_added']==replay['baseline_model_forwards']==0
        assert replay['candidate_model_forwards']==137 and replay['images']==548
    own=index['seeds'][str(seed)];root=Path(own['remote_root']);screen=root/f'seed{seed}_fine'
    assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    assert read(screen/'candidate_completion.json')['exit_code']==0
    assert sha(screen/'run/training_completion.json')==own['training_completion_sha256']
    assert sha(screen/'run/last.pth')==own['checkpoint_sha256']
    assert sha(screen/'candidate_reload_v1/verification.json')==own['reload_verification_sha256']
    assert read(screen/'candidate_reload_v1/completion.json')['exit_code']==0
    metadata=read(root/f'terminal_evidence_seed{seed}_metadata.json')
    assert metadata==own['terminal_archive'] and sha(root/f'terminal_evidence_seed{seed}.zip')==metadata['sha256']
    for name,pk,tk in [('controller_start.json','pid','start_ticks'),('candidate_start.json','child_pid','child_start_ticks')]:
        record=read(screen/name);proc=Path(f"/proc/{record[pk]}/stat")
        assert not proc.exists() or proc.read_text().split()[21]!=str(record[tk])
    return index

def verify_promotion(receipt,plan,paths):
    assert plan['variant']=='degraded' and plan['seed'] in (1,2)
    protocol=read(HERE/'protocol.json');assert protocol['execution_allowlist']=={'fine':[],'degraded':[1,2]}
    index=verify_fine_evidence(plan['seed']);proof=read(paths['screen_root']/'dual_seed_fine_promotion.json')
    assert proof['both_same_seed_native_gates_passed'] and proof['protocol_sha256']==sha(HERE/'protocol.json')
    assert proof['fine_promotion_sha256']==protocol['fine_promotion_sha256']==receipt['identity']['fine_promotion_sha256']
    identity=receipt['identity'];fine=read(HERE/'promotion'/f"seed{plan['seed']}_training_completion.json")['identity']
    for key in ('common_init_sha256','train_ann_sha256','val_ann_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','image_manifest_sha256','evaluator_binding'):
        assert identity[key]==fine[key],key
    return index
'''

PREPARE_CODE='''"""Prepare one never-trained degraded control using verified fine evidence."""
from pathlib import Path
import argparse,json,time
from runtime import HERE,COMMON_INIT_SHA,read,write,sha,implementation_hashes,verify_source,resolved_config
from promotion import verify_fine_evidence

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args()
    assert a.prefix=={1:'/25085400120',2:'/24085403014'}[a.seed]
    prefix=Path(a.prefix);experiment=HERE.parent;protocol=read(HERE/'protocol.json')
    assert protocol['execution_allowlist']=={'fine':[],'degraded':[1,2]}
    index=verify_fine_evidence(a.seed)
    for version in (1,2,3,4):
        prior=prefix/f'DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}/seed{a.seed}_degraded'
        assert not (prior/'candidate_start.json').exists() and not (prior/'run').exists(),'Do not reset a prior control budget'
    plan_path=HERE/f'configs/degraded_seed{a.seed}.json';plan=read(plan_path)
    assert plan['seed']==a.seed and plan['variant']=='degraded' and plan['screen_protocol_sha256']==sha(HERE/'protocol.json')
    source=prefix/'DEIM/experiments/fine_residual_screen24_20261009_v2/native';verify_source(source)
    old=prefix/f'DEIM/experiments/fine_residual_screen24_20261009_v2/runs/baseline_seed{a.seed}'
    original=read(old/'args.json')['yaml_cfg'];expected=plan['native_baseline_config'];allowed={'ResidualGeometryDEIM','model','output_dir','use_swanlab','__include__','config'}
    assert all(original.get(k)==expected.get(k) for k in set(original)|set(expected) if k not in allowed)
    assert original['ResidualGeometryDEIM']['arm']=='baseline'
    paths={'native_root':source,'common_init':Path(plan['common_init']),'evaluator_root':Path(plan['evaluator_root']),'baseline_checkpoint':old/'checkpoint0011.pth','baseline_first100':old/'first100_training_inputs.jsonl','screen_root':experiment/f'seed{a.seed}_degraded'}
    for split in ('train','val'):
        ds=expected[split+'_dataloader']['dataset'];paths[split+'_images']=Path(ds['img_folder']);paths[split+'_ann']=Path(ds['ann_file'])
    paths['output']=paths['screen_root']/'run'
    assert sha(paths['common_init'])==COMMON_INIT_SHA and sha(paths['baseline_checkpoint'])==plan['baseline_checkpoint_sha256']
    assert sha(paths['baseline_first100'])==plan['baseline_first100_sha256']
    assert sha(paths['train_ann'])=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
    assert sha(paths['val_ann'])=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    rows=[json.loads(s) for s in (old/'log.txt').read_text().splitlines() if json.loads(s)['epoch']==11]
    assert len(rows)==1 and rows[0]['test_coco_eval_bbox']==plan['historical_baseline_stats']
    from evaluator_binding import fingerprint
    evaluator=fingerprint(paths['evaluator_root']);assert evaluator==plan['evaluator_binding']
    bundle=experiment/f'prepared_seed{a.seed}';assert not bundle.exists() and not paths['screen_root'].exists()
    bundle.mkdir();paths['screen_root'].mkdir();started=time.monotonic();images={}
    for split,count in (('train',6471),('val',548)):
        data=read(paths[split+'_ann']);assert len(data['images'])==count;images[split]={}
        for i,image in enumerate(sorted(data['images'],key=lambda x:x['id'])):
            path=paths[split+'_images']/image['file_name'];assert path.is_file()
            images[split][image['file_name']]={'bytes':path.stat().st_size,'sha256':sha(path)}
            if (i+1)%1000==0:print(json.dumps({'stage':'image_identity','split':split,'files':i+1}),flush=True)
    write(bundle/'image_manifest.json',images)
    assert sha(bundle/'image_manifest.json')=='405381d365b72f6a1b700a5c8820056fa129fab8e7f558765f5168266f6b53f5'
    identity={'schema':plan['schema'],'arm':'degraded','seed':a.seed,'plan_sha256':sha(plan_path),'protocol_sha256':sha(HERE/'protocol.json'),'implementation':implementation_hashes(),'common_init_sha256':sha(paths['common_init']),'train_ann_sha256':sha(paths['train_ann']),'val_ann_sha256':sha(paths['val_ann']),'baseline_checkpoint_sha256':sha(paths['baseline_checkpoint']),'baseline_first100_sha256':sha(paths['baseline_first100']),'val_annotation_sha256':sha(paths['val_ann']),'train_annotation_sha256':sha(paths['train_ann']),'image_manifest_sha256':sha(bundle/'image_manifest.json'),'evaluator_binding':evaluator,'native_source_files':468,'preparation_code_sha256':sha(__file__),'fine_promotion_sha256':protocol['fine_promotion_sha256']}
    config=resolved_config(plan,paths);write(bundle/'resolved.yml',config);(bundle/'plan.json').write_bytes(plan_path.read_bytes())
    write(bundle/'receipt.json',{'status':'PREPARED_NOT_TRAINED','identity':identity,'paths':{k:str(v) for k,v in paths.items()},'resolved_sha256':sha(bundle/'resolved.yml'),'image_files':{'train':6471,'val':548},'elapsed_seconds':time.monotonic()-started})
    write(paths['screen_root']/'dual_seed_fine_promotion.json',{'both_same_seed_native_gates_passed':True,'both_saved_weight_reload_and_local_verification_passed':True,'protocol_sha256':sha(HERE/'protocol.json'),'fine_promotion_sha256':protocol['fine_promotion_sha256'],'automatic_retry':False})
    from reuse_baseline import verify_cached_baseline
    _,proof=verify_cached_baseline(read(bundle/'receipt.json'),plan,paths);write(bundle/'baseline_cache_check.json',proof)
    from runtime import verify_bundle
    verify_bundle(bundle)
    print(json.dumps({'status':'PREPARED','seed':a.seed,'arm':'degraded','bundle':str(bundle),'image_files':7019,'optimizer_steps':0,'epochs_added':0,'new_baseline_forwards':0}),flush=True)
if __name__=='__main__':main()
'''

def main():
    original=read(OLD/'protocol.json');plan_path=ROOT/'outputs/small_object_encoder_decoder_20261010_v2/prospective_screen_plan.json'
    assert sha(plan_path)==original['scientific_plan_sha256']
    assert not NEW.exists() and not OUT.exists()
    index={'status':'BOTH_FINE_FIXED12_RELOAD_AND_LOCAL_VERIFICATION_PASS','created_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scientific_plan_sha256':sha(plan_path),'seeds':{},'new_baseline_training':False,'new_baseline_forwards':0,'reader_implementation_sha256':sha(ANALYSIS/'../../英文小论文/code/small_object_encoder_decoder12_analysis_20261010_v1/verify_terminal_collection.py')}
    payload={};cache={};human=['两组细粒度模型固定第12轮结果','使用既有同seed BASE12，均为共同初始化的原132轮配方前12轮；不是成熟132轮模型的性能对比。','Seed\t指标\t基线\t改进\t差值（百分点）']
    for seed,version,prefix in ((1,4,'/25085400120'),(2,3,'/24085403014')):
        collection=ANALYSIS/f'terminal_collection/seed{seed}';x=collection/'extracted'
        r=read(collection/'verification_report.json');m=read(collection/f'terminal_evidence_seed{seed}_metadata.json')
        assert r['status']=='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD' and r['native_gate_passed_this_seed']
        assert r['files_verified']==2202 and r['archive_sha256']==m['sha256']==sha(collection/f'terminal_evidence_seed{seed}.zip')
        assert m['status']=='FIXED12_AND_RELOAD_COMPLETE'
        done=read(x/'run/training_completion.json');replay=read(x/'candidate_reload_v1/verification.json')
        assert sha(x/'run/last.pth')==done['last_checkpoint_sha256']==replay['checkpoint_sha256']
        assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
        for key in ('small_object_model.py','geometry_memory.py'):assert sha(OLD/key)==done['identity']['implementation'][key]==original['prototype_model_sha256'][key]
        for metric in ('AP','APs','AP75','small_AP75'):
            human.append(f"{seed}\t{metric}\t{r['baseline_overall_percent'][metric]:.6f}\t{r['candidate_overall_percent'][metric]:.6f}\t{r['delta_AP_metrics_pp'][metric]:+.6f}")
        st=read(ANALYSIS/f'paired_seed{seed}/report.json');p=st['paired_changes']['COCO_small']
        human.append(f"Seed{seed} COCO小目标原生全面积匹配：IoU0.50 新增{p['new_native_TP'][0]}、损失{p['lost_native_TP'][0]}；IoU0.75 新增{p['new_native_TP'][5]}、损失{p['lost_native_TP'][5]}。")
        index['seeds'][str(seed)]={'seed':seed,'remote_root':f'{prefix}/DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}','terminal_archive':m,'local_verification_sha256':sha(collection/'verification_report.json'),'training_completion_sha256':sha(x/'run/training_completion.json'),'reload_verification_sha256':sha(x/'candidate_reload_v1/verification.json'),'checkpoint_sha256':done['last_checkpoint_sha256'],'delta_AP_metrics_pp':r['delta_AP_metrics_pp']}
        for filename,path in ((f'seed{seed}_verification_report.json',collection/'verification_report.json'),(f'seed{seed}_training_completion.json',x/'run/training_completion.json'),(f'seed{seed}_reload_verification.json',x/'candidate_reload_v1/verification.json')):payload[filename]=path.read_bytes()
        base=x/'baseline_eval';cache[str(seed)]={'source':read(base/'baseline.yml')['output_dir'],'pinned_files':{n:sha(base/n) for n in ('completion.json','historical_comparison.json','baseline.yml','endpoint_epoch12/manifest.json','stages_epoch12/manifest.json')},'new_baseline_forwards':0,'new_baseline_epochs':0}
    human.extend(['','两seed均通过预设APs>BASE12且AP≥BASE12门槛，保存权重重载及4404文件本地核验通过。','参数+34.9358%，收益仍小。Seed1小目标IoU0.50 TP净减少111，IoU0.75净增加174；APs不是GT微平均检出率。','这些结果不能证明细粒度场的独立作用，也不能证明流形结构、分类定位冲突已解决、稳定性或首创。','下一步仅执行已预设同容量degraded seed1/2，各最多12轮；原模型代码、初始化及配方相同，P2仅经2×平均池化并双线性恢复。','不重新训练或GPU评估基线、不追加第三seed、不改终点、不延长任一配置。'])
    NEW.mkdir();OUT.mkdir();(NEW/'promotion').mkdir();(NEW/'configs').mkdir()
    for p in OLD.iterdir():
        if p.is_file() and p.name not in ('protocol.json','deployment_manifest.json'):(NEW/p.name).write_bytes(p.read_bytes())
    (NEW/'promotion.py').write_text(PROMOTION_CODE,encoding='utf8');(NEW/'prepare_remote.py').write_text(PREPARE_CODE,encoding='utf8')
    for name,data in payload.items():(NEW/'promotion'/name).write_bytes(data)
    write(NEW/'promotion/index.json',index)
    replace_once(NEW/'runtime.py',"IMPLEMENTATION=('reuse_baseline.py',","IMPLEMENTATION=('promotion.py','reuse_baseline.py',")
    replace_once(NEW/'runtime.py','    return receipt,plan,paths','    from promotion import verify_promotion\n    verify_promotion(receipt,plan,paths)\n    return receipt,plan,paths')
    replace_once(NEW/'reuse_baseline.py',"spec = read(HERE / 'protocol.json')['baseline_reuse']","spec = read(HERE / 'protocol.json')['baseline_reuse'][str(plan['seed'])]")
    replace_once(NEW/'experiment.py',"assert plan['seed']==1 and plan['variant']=='fine','Only the user-authorized corrected seed1 is allowed'","assert plan['seed'] in (1,2) and plan['variant']=='degraded','This deployment permits only the two prospectively specified controls'")
    replace_once(NEW/'experiment.py','        def load_tuning_state(self,path):self.initialization_receipt=load_common(self.model,path)',"        def load_tuning_state(self,path):\n            self.initialization_receipt=load_common(self.model,path)\n            assert self.initialization_receipt['variant']=='degraded' and self.initialization_receipt['parameters']==13806625 and self.initialization_receipt['extra_parameters']==3574628\n            assert self.model.encoder.geometry_field.degraded is True")
    protocol=dict(original);protocol.update(schema='small_object_encoder_decoder12_20261010_v5',status='FROZEN_TWO_DEGRADED_CONTROLS_AFTER_DUAL_FINE_VERIFIED_GATES',frozen_local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessor_protocol_sha256=sha(OLD/'protocol.json'),execution_allowlist={'fine':[],'degraded':[1,2]},stage1='Both fine seeds completed exact12; native gates,saved-weight reload,full local artifacts verified.',stage2='Only two predefined degraded samecapacity controls,each exactly native132-prefix12; no further configurations or automatic queue.',baseline_reuse=cache,fine_promotion_sha256=sha(NEW/'promotion/index.json'),audit_correction=dict(original['audit_correction'],actual_full_epoch_training_integration_verified=True))
    protocol['historical_repaired_seed1_authority']=protocol.pop('repaired_seed1_authority')
    write(NEW/'protocol.json',protocol);write(OUT/'protocol.json',protocol)
    for seed in (1,2):
        plan=read(OLD/f'configs/degraded_seed{seed}.json');plan.update(schema=protocol['schema'],screen_protocol_sha256=sha(NEW/'protocol.json'));write(NEW/f'configs/degraded_seed{seed}.json',plan)
    unchanged=('small_object_model.py','geometry_memory.py','training_audit.py','results.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','baseline_eval.py','supervise.py')
    assert all(sha(NEW/n)==sha(OLD/n) for n in unchanged)
    for p in NEW.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=p.name)
    files={p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()};write(NEW/'deployment_manifest.json',files)
    archive=OUT/'deploy_bundle.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in list(files)+['deployment_manifest.json']:z.write(NEW/n,n)
    meta={'status':'FROZEN_NOT_DEPLOYED','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'protocol_sha256':sha(NEW/'protocol.json'),'files':len(files)+1,'unchanged_files':list(unchanged),'promotion_sha256':sha(NEW/'promotion/index.json'),'new_optimizer_updates':0,'new_model_forwards':0,'new_baseline_forwards':0,'max_epochs_per_control':12}
    write(OUT/'deployment_local.json',meta);write(OUT/'fine_stage1_promotion.json',index)
    (OUT/'两组固定12轮结果与对照计划.txt').write_text('\n'.join(human)+'\n',encoding='utf8')
    print(json.dumps(meta,ensure_ascii=False))

if __name__=='__main__':main()
