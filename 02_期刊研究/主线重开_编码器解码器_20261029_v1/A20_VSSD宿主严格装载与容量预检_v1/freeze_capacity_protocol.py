"""Freeze the bounded protocol and its ledger before any GPU model call."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;R=O.parent
def rd(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (O/'capacity_protocol_FROZEN.json').exists(),'Already frozen; inspect, never recreate'
    previous=rd(R/'49_A19方法与宿主资产核验回执.json')
    for x in previous['files']:assert sha(R/x['path'])==x['sha256'],x['path']
    strict=rd(O/'strict_host_load.json')
    assert strict['classification_strict_load'] and strict['mapped_backbone_strict_load'] and strict['complete_model_forwards']==0
    assert strict['host_state_digest']=='8e27a3a06df1f52d91c4190c1f903e4a27566e9d8aaa7b4ce95d30e5f3c76229'
    inputs=rd(O/'capacity_inputs/input_manifest.json')
    assert [x['valid_gt_count'] for x in inputs['cases']]==[902,512]
    ast.parse((O/'a20_capacity.py').read_text(encoding='utf-8'))
    st=rd(R/'15_当前状态与结果校验.json')
    assert st['model_forwards']==st['cap']==252 and st['training_updates']==0 and not st['mainline_validated']
    snap=R/'history_before_A20_freeze_v1';snap.mkdir(exist_ok=False)
    for n in ['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json','49_A19方法与宿主资产核验回执.json']:shutil.copy2(R/n,snap/n)
    names=['a20_host.py','a20_capacity.py','bundle_manifest.json','strict_host_load.json','capacity_inputs/input_manifest.json']
    for c in inputs['cases']:names.extend(['capacity_inputs/'+c['image_file'],'capacity_inputs/'+c['annotation_file']])
    protocol={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_ANY_NEW_FORWARD',
       'purpose':'Specified host resource/integration capacity only; not novelty, effectiveness or causal evidence',
       'host':'VSSD-Tiny ICCV2025 + fresh native Dome-S encoder/decoder, native losses and DN',
       'input_size':[800,800],'channels':[96,192,384,768],'strides':[4,8,16,32],
       'num_classes':12,'valid_gt_classes':list(range(1,11)),'queries':[250,500],
       'expected_host_state_digest':strict['host_state_digest'],'host_seed':20261030,'runtime_seed':42301,
       'stages':[{'id':'eval_fp32_b1','batch':1,'complete_forwards':1,'optimizer_updates':0},
                 {'id':'train_amp_b1','batch':1,'complete_forwards':1,'optimizer_updates':1},
                 {'id':'train_amp_b2','batch':2,'complete_forwards':1,'optimizer_updates':1}],
       'caps':{'new_complete_detector_forwards':3,'new_suffix_forwards':0,'disposable_optimizer_updates':2,'formal_epochs':0},
       'counting':'Model prehook logs every complete-call entry including failures; no warmup, profiling forward, replay or checkpoint recomputation',
       'optimizer':{'type':'AdamW','lr':1e-5,'betas':[0.9,0.999],'weight_decay':0.000065,'foreach':False,
                    'grouping':'all trainable parameters, capacity-only; this does not claim full author training recipe'},
       'precision':{'eval':'FP32 TF32 disabled','train':'autocast CUDA FP16; native criterion outside autocast as author engine',
                    'grad_scaler_init':128.0,'growth_interval':2000,'max_grad_norm':0.1},
       'source_observation':'Before native torch.nan_to_num loss sanitation, temporary observer asserts finite tensor input; finite arithmetic unchanged',
       'runtime':'OMP/MKL4 threads; cuDNN benchmarkFalse; minimum20GiB free before host allocation',
       'success':'All3 calls complete with exact4 feature shapes, finite outputs/rawlosses/gradients; both actual optimizer updates; peak memory recorded including optimizer state',
       'failure':'Stop at first mismatch, OOM, nonfinite, skipped AMP update; retain traceback/events; no retries or resolution/query/model changes',
       'interpretation_limits':['No convergence, AP, stable throughput, full training duration, confirmed failure cause or novelty claim',
                                'High GT training cases selected by annotations only; no test or validation accuracy use',
                                'No capacity weight retained; no automatic formal training/old queue'],
       'files':[{'path':n,'sha256':sha(O/n),'bytes':(O/n).stat().st_size} for n in names]}
    wr(O/'capacity_protocol_FROZEN.json',protocol)
    with zipfile.ZipFile(O/'capacity_run_control.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for n in ['a20_capacity.py','capacity_protocol_FROZEN.json']:z.write(O/n,n)
    wr(O/'capacity_run_control_receipt.json',{'sha256':sha(O/'capacity_run_control.zip'),'bytes':(O/'capacity_run_control.zip').stat().st_size})
    hist=st.setdefault('authoritative_phase_history',[]);hist.append(st['latest_authoritative_phase'])
    st['cap']=255;st['recorded_utc']=protocol['utc'];st['goal_status']='ACTIVE'
    st['current_goal_turn']='progress_A20_strict_native_host_loaded_capacity_protocol_frozen'
    st['latest_authoritative_phase']={'utc':protocol['utc'],'phase':'A20','classification':'progress','status':'CAPACITY_PROTOCOL_FROZEN_NOT_YET_EXECUTED',
      'full_model_forwards':252,'cumulative_authorized_protocol_cap':255,'new_full_cap':3,'new_optimizer_update_cap':2,
      'actual_new_full_forwards':0,'actual_capacity_optimizer_updates':0,'mainline_validated':False,
      'protocol':str(O/'capacity_protocol_FROZEN.json'),'protocol_sha256':sha(O/'capacity_protocol_FROZEN.json')}
    wr(R/'15_当前状态与结果校验.json',st)
    print(json.dumps({'status':'FROZEN_BEFORE_NEW_CALLS','capacity_control':rd(O/'capacity_run_control_receipt.json'),
       'previous_complete_calls':252,'new_forward_cap':3,'new_capacity_update_cap':2,'canonical_cap':st['cap']}))
if __name__=='__main__':main()
