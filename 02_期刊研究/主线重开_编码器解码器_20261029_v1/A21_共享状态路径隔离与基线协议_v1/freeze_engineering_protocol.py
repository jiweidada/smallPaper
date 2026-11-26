"""Freeze five no-training instrumentation calls before execution."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;R=O.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):return json.loads(p.read_text(encoding='utf-8'))
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (O/'engineering_protocol_FROZEN.json').exists(),'Frozen already: inspect, never rerun'
    last=rd(R/'51_A20严格装载与容量实测核验回执.json')
    for row in last['files']:assert sha(R/row['path'])==row['sha256'],row['path']
    for n in ['a21_engineering_check.py','state_path_bridge.py']:ast.parse((O/n).read_text(encoding='utf-8'))
    st=rd(R/'15_当前状态与结果校验.json')
    assert st['model_forwards']==st['cap']==255 and st['training_updates']==2
    assert st['formal_detector_training_updates']==0 and not st['mainline_validated']
    snap=R/'history_before_A21_freeze_v1';snap.mkdir(exist_ok=False)
    for n in ['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json','51_A20严格装载与容量实测核验回执.json']:shutil.copy2(R/n,snap/n)
    p={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_ANY_A21_FORWARD',
       'purpose':'Engineering validity of native state/mean mediator intervention; no trained-host or scientific effect claim',
       'host':'Same fresh VSSD-Tiny + native Dome-S diagnostic host verified A20; no temporary capacity weights reused',
       'expected_initial_state_digest':'8e27a3a06df1f52d91c4190c1f903e4a27566e9d8aaa7b4ce95d30e5f3c76229',
       'operator':'backbone.native.layers.0.blocks.0.attn.non_casual_linear_attn; normalized noncausal branch',
       'mediators':['S=(K_rope^T/sqrtL)@(V/sqrtL)','M=weighted unrotated K_pre.mean(token axis)'],
       'replacement':'replace S and/or M; denominator recomputed using original Q_pre and donated M; original Q,V,D,u-derived gate and local source inputs unchanged',
       'no_mean_replacement_uses_original_denominator_tensor':True,
       'query_control':'min_num_select=max_num_select=500 during evaluation only; native protected first500 selection; all5 calls assert actual500 queries',
       'input':'A20 first fixed training capacity image 0000059_01886_d_0000114.jpg, resize800; no val or GT-outcome analysis',
       'engineering_perturbation':'Fixed RGB inversion in [400:408,400:408] on resized float tensor; not an object/neighbor intervention or scientific treatment',
       'stages':['native_original','noop_original','identity_state_and_mean','altered_input_capture','original_with_altered_mediators'],
       'new_complete_forward_cap':5,'optimizer_update_cap':0,'formal_epoch_cap':0,'new_suffix_calls':0,
       'checks':['Native/noop and native/identity complete logits+boxes bitwise equal',
                 'Original module inputs and x/dt/A/B/C/D arguments identical across original/hybrid',
                 'Original Q_pre/Q/V/D unchanged in hybrid; donated S/M exact; nontrivial intervention active',
                 'Only one AST observer insertion; removing it recovers exact original function AST',
                 'Native author files unchanged; no silently ignored mismatches'],
       'runtime':'FP32 eval, inference_mode, TF32False, benchmarkFalse, deterministicAlgorithmsTrue, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL4, minimum16GiB free',
       'failure':'Stop at first mismatch/nonfinite/OOM; preserve calls and evidence; no automatic retries/tolerance or location changes',
       'interpretation':'No detector accuracy, harmful-interaction estimate, trained-model evidence, novelty or validated mainline',
       'existing_user_authorization_continues':True,
       'files':[{'path':n,'sha256':sha(O/n),'bytes':(O/n).stat().st_size} for n in ['a21_engineering_check.py','state_path_bridge.py']]}
    wr(O/'engineering_protocol_FROZEN.json',p)
    with zipfile.ZipFile(O/'engineering_control.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for n in ['a21_engineering_check.py','state_path_bridge.py','engineering_protocol_FROZEN.json']:z.write(O/n,n)
    wr(O/'engineering_control_receipt.json',{'sha256':sha(O/'engineering_control.zip'),'bytes':(O/'engineering_control.zip').stat().st_size})
    st.setdefault('authoritative_phase_history',[]).append(st['latest_authoritative_phase'])
    st['cap']=260;st['recorded_utc']=p['utc'];st['current_goal_turn']='progress_A21_native_state_mediation_instrumentation_protocol_frozen'
    st['latest_authoritative_phase']={'utc':p['utc'],'phase':'A21','classification':'progress','status':'FROZEN_NOT_EXECUTED',
       'full_model_forwards':255,'cumulative_authorized_protocol_cap':260,'new_full_cap':5,
       'actual_new_full_forwards':0,'optimizer_update_cap':0,'mainline_validated':False,'protocol_sha256':sha(O/'engineering_protocol_FROZEN.json')}
    wr(R/'15_当前状态与结果校验.json',st)
    print(json.dumps({'status':'A21_FROZEN','new_forwards_cap':5,'new_update_cap':0,'control':rd(O/'engineering_control_receipt.json')}))
if __name__=='__main__':main()
