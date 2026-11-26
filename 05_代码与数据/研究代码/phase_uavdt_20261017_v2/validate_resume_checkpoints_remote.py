"""Read-only CPU validation of complete, finite, architecture-compatible resume states."""
import contextlib,datetime,hashlib,io,json,math,os,sys,zipfile
from pathlib import Path
root=Path(CONFIG['root'])
def write_record(path,value):
    if not path:return
    path=Path(path);assert path.is_relative_to(root/'recovery_20261019_v2')
    temp=path.with_suffix('.next');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)
def progress(phase,**extra):
    write_record(CONFIG.get('validation_progress_path'),dict(phase=phase,remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
def fatal(kind,value,traceback):
    write_record(CONFIG.get('validation_result_path'),{'account':CONFIG['account'],'all_passed':False,'fatal':kind.__name__+': '+str(value)})
    sys.__excepthook__(kind,value,traceback)
sys.excepthook=fatal
progress('importing_torch',pid=os.getpid())
import torch
torch.set_num_threads(2)
plan=json.loads((root/'plan.json').read_text())
sys.path.insert(0,str(root/'code'))
captured=io.StringIO()
progress('importing_frozen_model_code',pid=os.getpid())
with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
    from uavdt_protocol import install_native_adapter
    install_native_adapter()
    from engine.core import YAMLConfig

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

EXPECTED_SENTINELS={}
def finite_tree(value,location='state'):
    tensors=0;elements=0
    if torch.is_tensor(value):
        tensors=1;elements=value.numel()
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                expected=EXPECTED_SENTINELS.get(location)
                assert expected is not None,'Non-finite tensor at '+location
                assert value.shape==expected.shape and not torch.isnan(value).any().item()
                assert torch.equal(torch.isposinf(value),torch.isposinf(expected))
                assert torch.equal(torch.isneginf(value),torch.isneginf(expected))
    elif isinstance(value,dict):
        for k,v in value.items():
            a,b=finite_tree(v,location+'.'+str(k));tensors+=a;elements+=b
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):
            a,b=finite_tree(v,location+'.'+str(i));tensors+=a;elements+=b
    elif isinstance(value,float):assert math.isfinite(value),'Non-finite scalar at '+location
    return tensors,elements

result={'account':CONFIG['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'mode':'CPU validation only; no model forward, optimizer step or GPU use','jobs':[]}
for arm in plan['jobs']:
    job=f'uavdt_{arm}_full96_seed{plan["seed"]}';out=root/'outputs'/job;cp=out/'last.pth'
    item={'job':job,'arm':arm,'checkpoint':str(cp),'passed':False}
    try:
        progress('checkpoint_crc',job=job,pid=os.getpid())
        before=cp.stat();assert before.st_size>100000000
        with zipfile.ZipFile(cp) as z:
            assert z.testzip() is None;item['zip_members']=len(z.infolist())
        checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
        assert {'model','ema','optimizer','scaler','lr_warmup_scheduler','last_epoch','best_stat'}.issubset(checkpoint)
        assert 0<=checkpoint['last_epoch']<95
        rows=[json.loads(s) for s in (out/'log.txt').read_text().splitlines() if s.strip()]
        rows=[r for r in rows if 'test_uavdt_eval_bbox' in r]
        assert [r['epoch'] for r in rows]==list(range(checkpoint['last_epoch']+1)),'Log/checkpoint epoch mismatch'
        args=json.loads((out/'args.json').read_text())
        assert args['seed']==plan['seed'] and args['epoches']==96 and args['use_amp'] is True
        assert Path(args['output_dir']).resolve()==out.resolve()
        declared_config=args.get('config') or args.get('yaml_cfg',{}).get('config')
        assert declared_config and Path(declared_config).resolve()==Path(plan['configs'][arm]).resolve(),'Config binding missing or mismatched'
        assert checkpoint['ema']['updates']>0 and checkpoint['ema']['module']
        assert checkpoint['optimizer']['state'] and checkpoint['optimizer']['param_groups']
        assert checkpoint['scaler']['scale']>0 and checkpoint['scaler']['growth_interval']>0
        progress('strict_model_compatibility',job=job,pid=os.getpid())
        with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
            cfg=YAMLConfig(plan['configs'][arm]);assert cfg.yaml_cfg['num_classes']==1
            assert cfg.yaml_cfg['epoches']==96
            backbone=cfg.yaml_cfg['DEIM']['backbone']
            assert cfg.yaml_cfg[backbone].get('pretrained') is False,'Validation must never fetch pretrained weights'
            model=cfg.model.cpu()
            EXPECTED_SENTINELS.clear()
            for name,tensor in model.named_buffers():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                    assert name.endswith('anchors') and not torch.isnan(tensor).any().item(),'Unexpected non-finite frozen-model buffer'
                    EXPECTED_SENTINELS['state.model.'+name]=tensor.clone()
                    EXPECTED_SENTINELS['state.ema.module.'+name]=tensor.clone()
            model.load_state_dict(checkpoint['model'],strict=True)
            model.load_state_dict(checkpoint['ema']['module'],strict=True)
        progress('finite_state_validation',job=job,pid=os.getpid())
        tensors,elements=finite_tree(checkpoint)
        for parameter_id,state in checkpoint['optimizer']['state'].items():
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                assert state['exp_avg'].shape==state['exp_avg_sq'].shape
                assert (state['exp_avg_sq']>=0).all().item()
        after=cp.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        item.update(passed=True,checkpoint_sha256=sha(cp),checkpoint_bytes=before.st_size,
            last_epoch=checkpoint['last_epoch'],logged_epochs=len(rows),
            zip_crc_passed=True,all_learned_and_optimizer_state_finite=True,model_and_ema_strict_load_passed=True,
            architecture_defined_anchor_sentinels_validated=list(EXPECTED_SENTINELS),
            tensors_checked=tensors,tensor_elements_checked=elements,
            optimizer_state_entries=len(checkpoint['optimizer']['state']),
            ema_updates=checkpoint['ema']['updates'],scaler_scale=checkpoint['scaler']['scale'],
            expected_output_directory=str(out),backbone=backbone,
            latest_complete_AP70=rows[-1]['test_uavdt_eval_bbox'][0])
        del model,checkpoint,cfg
    except Exception as exc:item.update(error=type(exc).__name__+': '+str(exc))
    result['jobs'].append(item)
    progress('job_validated',job=job,passed=item['passed'],pid=os.getpid())
result['all_passed']=len(result['jobs'])==3 and all(j['passed'] for j in result['jobs'])
result['captured_initialization_log_tail']=captured.getvalue()[-1800:]
write_record(CONFIG.get('validation_result_path'),result)
progress('complete',all_passed=result['all_passed'],pid=os.getpid())
print(json.dumps(result),flush=True)
