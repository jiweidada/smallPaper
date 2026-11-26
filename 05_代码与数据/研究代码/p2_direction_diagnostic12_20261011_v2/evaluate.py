"""Two previously unattempted direction controls; preserve failed replay and compare controls directly."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def main(prefix,seed):
    assert seed==1 and prefix=='/25085400120'
    protocol=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in protocol['remote_sources']}==protocol['remote_sources']
    user=Path(prefix)/'DEIM/experiments/user_backbone12_20261011_v1';code=user/'code'
    assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==protocol['user28_sources']
    screen=user/f'seed{seed}_p2_skip';run=screen/'run';cpu=Path(prefix)/'DEIM/experiments/p2_subspace_diagnostic12_20261011_v1/seed1_CPU'
    preparation=read(cpu/'report.json');completed=read(cpu/'completion.json')
    assert preparation['status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS' and completed['exit_code']==0
    assert preparation['seed']==seed and preparation['source_sha256']==protocol['cpu_prepare_source_sha256'] and preparation['protocol_sha256']==protocol['cpu_protocol_sha256']
    assert completed['report_sha256']==sha(cpu/'report.json') and completed['weights_npz_sha256']==sha(cpu/'weights.npz')
    replay=read(screen/'candidate_reload_v1/verification.json')
    from reference_contract import validate
    validate(preparation,completed,replay,protocol)
    assert sha(screen/'candidate_reload_v1/verification.json')==protocol['failed_replay_verification_sha256']
    assert sha(cpu/'report.json')==protocol['CPU_report_sha256'] and sha(cpu/'weights.npz')==protocol['CPU_weights_sha256']
    assert not (Path(prefix)/'DEIM/experiments/p2_subspace_diagnostic12_20261011_v1/seed1_GPU').exists(), 'These controls must still be unattempted'
    assert replay['verification_script_sha256']==sha(code/'verify_candidate_reload.py') and replay['closeout_protocol_sha256']==sha(code/'closeout_protocol.json')
    assert replay['identity']==preparation['identity'] and replay['checkpoint_sha256']==preparation['checkpoint_sha256']==sha(run/'last.pth')
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==protocol['GPU_uuid'][str(seed)]
    sys.path.insert(0,str(code));from runtime import verify_bundle,activate,register_model,resolved_config
    receipt,plan,paths=verify_bundle(screen/'bundle');assert receipt['identity']==preparation['identity'] and plan['arm']=='p2_skip' and plan['max_epochs']==12
    from results import load_endpoint,capture
    from stage_trace import StageTrace,TracedLoader,verify_trace
    from evaluator_binding import activate as bind_evaluator
    parent,parent_hits,_=load_endpoint(run/'endpoint_epoch12');assert parent['identity']==receipt['identity'] and parent['native_stats']==replay['original_stats']
    reloaded,_,_=load_endpoint(screen/'candidate_reload_v1/endpoint_epoch12')
    assert reloaded['identity']==receipt['identity'] and reloaded['native_stats']==replay['reloaded_stats']
    assert sha(screen/'candidate_reload_v1/endpoint_epoch12/manifest.json')==protocol['reloaded_reference_endpoint_manifest_sha256']
    assert sha(run/'endpoint_epoch12/manifest.json')==read(run/'training_completion.json')['endpoint_manifest_sha256']
    assert sha(run/'stages_epoch12/manifest.json')==read(run/'training_completion.json')['stage_manifest_sha256']
    verify_trace(run/'stages_epoch12',receipt['identity'])
    activate(paths['native_root']);bind_evaluator(paths['evaluator_root'],receipt['identity']['evaluator_binding']);register_model()
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver.det_engine import evaluate
    root=HERE.parent/f'seed{seed}_GPU';root.mkdir(exist_ok=False)
    started=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=seed,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_sha256=preparation['checkpoint_sha256'],source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),automatic_retry=False,new_training_epochs=0,new_BASE_forwards=0)
    write(root/'start.json',started)
    try:
        with np.load(cpu/'weights.npz',allow_pickle=False) as z:weights={n:z[n].copy() for n in z.files}
        assert set(weights)=={'original',*protocol['interventions']}
        assert hashlib.sha256(weights['original'].tobytes()).hexdigest()==preparation['parent_restore_weight_SHA256']
        assert [r['mode'] for r in preparation['cases']]==protocol['interventions']
        for case in preparation['cases']:
            assert case['numerical_rank']==16 and case['relative_Frobenius_norm_error']<1e-6
            assert weights[case['mode']].shape==(256,64,1,1) and weights[case['mode']].dtype==np.float32
            assert hashlib.sha256(weights[case['mode']].tobytes()).hexdigest()==case['weight_SHA256']
        results=[];total_calls=0
        for mode in protocol['interventions']:
            dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=seed)
            out=root/mode;out.mkdir(exist_ok=False)
            config=resolved_config(plan,dict(paths,output=out));write(out/'resolved.yml',config)
            cfg=YAMLConfig(str(out/'resolved.yml'));model=cfg.model.cuda()
            payload=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
            assert payload['last_epoch']==11 and payload['user_backbone12_identity']==receipt['identity']
            expected=payload['ema']['module'];model.load_state_dict(expected,strict=True)
            loaded=model.state_dict();assert set(loaded)==set(expected) and all(torch.equal(v.detach().cpu(),expected[n].cpu()) for n,v in loaded.items())
            assert len(loaded)==801 and sum(p.numel() for p in model.parameters())==10249437
            assert np.array_equal(model.backbone.p2_skip.restore.weight.detach().cpu().numpy(),weights['original'])
            with torch.no_grad():model.backbone.p2_skip.restore.weight.copy_(torch.from_numpy(weights[mode]).to(model.backbone.p2_skip.restore.weight))
            changed=[n for n,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),expected[n].cpu())]
            assert changed==['backbone.p2_skip.restore.weight']
            identity=copy.deepcopy(receipt['identity']);identity.update(schema='P2_unattempted_direction_control_v2',runtime_intervention=dict(mode=mode,rank=16,normalization='parent_restore_Frobenius_norm',parent_checkpoint_sha256=preparation['checkpoint_sha256'],parent_training_identity_sha256=hashlib.sha256(json.dumps(receipt['identity'],sort_keys=True).encode()).hexdigest(),projected_restore_weight_sha256=hashlib.sha256(weights[mode].tobytes()).hexdigest(),CPU_preparation_report_sha256=sha(cpu/'report.json')))
            del payload,expected,loaded
            model.requires_grad_(False);model.eval();criterion=cfg.criterion.cuda();criterion.set_model(model);criterion.set_epoch(11)
            post,loader,evaluator=cfg.postprocessor.cuda(),cfg.val_dataloader,cfg.evaluator
            calls=0
            def observe(m,args,output):
                nonlocal calls
                assert not m.training and not torch.is_grad_enabled()
                assert output['pred_boxes'].shape[1:]==(300,4) and output['pred_logits'].shape[1:]==(300,10)
                assert torch.isfinite(output['pred_boxes']).all() and torch.isfinite(output['pred_logits']).all()
                calls+=1
            handle=model.register_forward_hook(observe);trace=StageTrace(model,identity,paths['val_ann'])
            try:
                with torch.no_grad():stats,obj=evaluate(model,criterion,post,TracedLoader(loader,trace),evaluator,torch.device('cuda:0'),test_only=False,output_dir=out,yolo_metrice=config['yolo_metrice'])
                assert obj is evaluator and calls==len(loader)==137
                trace.finish(out/'stages_epoch12');capture(evaluator.coco_eval['bbox'],stats,identity,paths['val_ann'],out/'endpoint_epoch12',epoch=11,environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0)))
                summary,_,_=load_endpoint(out/'endpoint_epoch12');verify_trace(out/'stages_epoch12',identity)
                assert summary['identity']==identity and summary['evaluated_weights']=='EMA' and summary['epoch_zero_based']==11
                record=dict(mode=mode,identity=identity,native_stats=summary['native_stats'],parent_native_stats=parent['native_stats'],delta_AP_points=100*(summary['native_stats'][0]-parent['native_stats'][0]),delta_APs_points=100*(summary['native_stats'][3]-parent['native_stats'][3]),candidate_model_forwards=calls,changed_state_tensors=changed,parameters=10249437,endpoint_manifest_sha256=sha(out/'endpoint_epoch12/manifest.json'),stage_manifest_sha256=sha(out/'stages_epoch12/manifest.json'))
                record.update(overall_percent=summary['overall_percent'],categories_percent=summary['categories_percent'],descriptive_reloaded_reference_delta_points=dict(AP=100*(summary['native_stats'][0]-reloaded['native_stats'][0]),APs=100*(summary['native_stats'][3]-reloaded['native_stats'][3])),reference_replay_failed_preserved=True,parent_delta_is_not_formal_efficacy=True)
                write(out/'report.json',record);results.append(record);total_calls+=calls
            finally:
                handle.remove();trace.close();dist_utils.cleanup()
            del model,criterion,post,loader,evaluator,cfg,trace,handle,obj;torch.cuda.empty_cache()
        assert total_calls==274 and sha(run/'last.pth')==preparation['checkpoint_sha256']
        assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==protocol['user28_sources']
        report=dict(status='TWO_PREVIOUSLY_UNATTEMPTED_P2_DIRECTION_CONTROLS_NATIVE_EVALUATED',seed=seed,results=results,parent_checkpoint_sha256=preparation['checkpoint_sha256'],parent_replay_verification_sha256=sha(screen/'candidate_reload_v1/verification.json'),CPU_preparation_report_sha256=sha(cpu/'report.json'),source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),candidate_model_forwards=274,new_native_evaluator_runs=2,new_BASE_forwards=0,new_training_epochs=0,new_optimizer_updates=0,checkpoint_unchanged=True,original28_training_sources_unchanged=True,root_proven=False,novelty_proven=False,training_efficacy_proven=False,scope='Direct SVD16-minus-DC16 direction contrast with all non-restore checkpoint tensors exact. Both controls were previously unattempted. Rank and restore-weight norm are matched; operator approximation error and feature energy are not matched. Failed original parent replay remains failed; original/reloaded references are descriptive only. No parent/BASE forwards, training efficacy or unique root claim.')
        report.update(primary_contrast=dict(name='svd16_norm_minus_dc16_norm',delta_points={k:results[1]['overall_percent'][k]-results[0]['overall_percent'][k] for k in results[0]['overall_percent']}),parent_reference_replay_status=replay['status'],reference_replay_failed_preserved=True,old_failed_replay_retried=False,parent_model_forwards_added=0)
        write(root/'report.json',report);write(root/'completion.json',dict(status=report['status'],exit_code=0,receipt=started,report_sha256=sha(root/'report.json'),automatic_retry=False))
        print(json.dumps(dict(status=report['status'],seed=seed,results=[dict(mode=r['mode'],delta_AP=r['delta_AP_points'],delta_APs=r['delta_APs_points']) for r in results])),flush=True)
    except BaseException:
        write(root/'failure.json',dict(receipt=started,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();main(a.prefix,a.seed)
