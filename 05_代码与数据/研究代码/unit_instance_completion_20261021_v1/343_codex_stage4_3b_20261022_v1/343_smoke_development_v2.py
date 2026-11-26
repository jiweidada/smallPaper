"""New-label CPU smoke only: same first32 windows, 20 steps/arm/seed, bitwise resume."""
import importlib,json,time,copy,random,traceback
C=importlib.import_module('343_context');G=C.load_geometry()
import torch,numpy as np
T=importlib.import_module('train342');D=importlib.import_module('343_development')
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
COUNTS={'new_cache_loads':0,'head_forwards':0,'native_loss_adapter_calls':0,'optimizer_steps':0,
    'checkpoint_saves':0,'checkpoint_loads':0,'global_model_loads':0,'global_forwards':0}
original=T.compute_losses_340
def compute(*args,**kwargs):
    COUNTS['head_forwards']+=1
    return original(*args,**kwargs)
T.compute_losses_340=compute
UL=importlib.import_module('uic_losses');native_original=UL.compute_losses
def native(*args,**kwargs):
    COUNTS['native_loss_adapter_calls']+=1
    return native_original(*args,**kwargs)
UL.compute_losses=native

def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def rng(seed):
    random.seed(2026102100+seed);np.random.seed(2026102100+seed);torch.manual_seed(2026102100+seed)

def optimizer(model):
    o=T.build_optimizer(model);assert {id(p) for g in o.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    def hook(*args):COUNTS['optimizer_steps']+=1
    o.register_step_post_hook(hook);return o

def smoke(arm,seed,batches):
    rng(seed);m=T.make_common_init(arm,seed);m.train();initial=copy.deepcopy(m.state_dict());o=optimizer(m);logs=[];start=time.perf_counter()
    for step in range(20):
        for gi,g in enumerate(o.param_groups):g['lr']=T.lr_at_kappa(step,1218) if arm=='G' and gi==2 else T.lr_at(step,1218)
        o.zero_grad(set_to_none=True);loss,out=compute(arm,m,batches[step%2])
        assert all(bool(torch.isfinite(x)) for x in loss.values())
        loss['loss_total'].backward();assert all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
        row={'step':step,'losses':{k:float(v.detach()) for k,v in loss.items()},'lr_groups':[g['lr'] for g in o.param_groups]}
        if arm=='G':
            row['kappa_before']=m.class_null.detach().tolist();row['kappa_gradient_before_clip']=m.class_null.grad.detach().tolist()
        row['gradient_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.))
        if arm=='G':row['kappa_gradient_after_clip']=m.class_null.grad.detach().tolist()
        o.step()
        if arm=='G':
            row['kappa_after']=m.class_null.detach().tolist();row['kappa_update']=(np.array(row['kappa_after'])-np.array(row['kappa_before'])).tolist()
        logs.append(row)
    changed=[k for k,v in m.state_dict().items() if not torch.equal(initial[k],v)];assert changed
    result={'steps':20,'loss_finite':True,'gradients_finite':True,'changed_state_tensor_count':len(changed),
        'parameter_count':sum(p.numel() for p in m.parameters()),'log':logs,'cpu_seconds':time.perf_counter()-start}
    if arm=='G':
        grads=np.array([r['kappa_gradient_before_clip'] for r in logs]);result['per_class_nonzero_gradient_steps']=(grads!=0).sum(0).tolist()
        result['class_null_final']=m.class_null.detach().tolist();loss,_=compute(arm,m,batches[0]);routing={}
        for k in ('loss_det','loss_mass','loss_assoc'):
            grad=torch.autograd.grad(loss[k],m.class_null,allow_unused=True,retain_graph=True)[0];routing[k]=None if grad is None else grad.detach().tolist()
        assert routing['loss_mass'] is None and routing['loss_assoc'] is None;result['class_null_loss_routing']=routing
    print(f'smoke seed={seed} arm={arm}20; lastloss={logs[-1]["losses"]["loss_total"]:.8f}',flush=True)
    return result

def resume(arm,seed,batches):
    rng(seed);a=T.make_common_init(arm,seed);oa=optimizer(a);T.run_steps(a,oa,batches,0,2,1218)
    state=copy.deepcopy(a.state_dict());opstate=copy.deepcopy(oa.state_dict());rstate=T.capture_rng()
    rng(seed);b=T.make_common_init(arm,seed);ob=optimizer(b);T.run_steps(b,ob,batches,0,1,1218)
    path=C.HERE/f'343_CPU_smoke_resume_seed{seed}_{arm}.pt'
    T.save_checkpoint(path,b,ob,1,0,2026102100+seed,T.capture_rng());COUNTS['checkpoint_saves']+=1
    restored=T.make_common_init(arm,seed);ore=optimizer(restored)
    cursor=T.load_checkpoint(path,restored,ore);COUNTS['checkpoint_loads']+=1
    assert cursor==(1,0,2026102100+seed);assert [g['lr'] for g in ore.param_groups]==[g['lr'] for g in ob.param_groups]
    T.run_steps(restored,ore,batches,1,1,1218)
    assert equal(state,restored.state_dict()) and equal(opstate,ore.state_dict()) and equal(rstate,T.capture_rng())
    return {'model_state_bit_equal':True,'optimizer_bit_equal':True,'rng_bit_equal':True,'cursor':list(cursor),
        'checkpoint_path':str(path),'checkpoint_sha256':C.sha(path),'scope':'continuous2 versus1+save/load+1; same two batches; CPU only'}

def main():
    assert json.loads((C.HERE/'343_result_3b已生成6469条独立验收.json').read_text(encoding='utf-8'))['validated_entries']==6469
    ids,_=C.development_ids();ids=ids[:32];results={};resumes={};comparisons=[]
    old=json.loads((C.S2/'342_result_四臂CPU冒烟.json').read_text(encoding='utf-8'))['arms']
    try:
        for seed in (1,2):
            entries=[torch.load(D.entry_path(seed,i),map_location='cpu',weights_only=True) for i in ids];COUNTS['new_cache_loads']+=32
            assert [e['image_id'] for e in entries]==ids
            batches=[T.collate(entries[:16]),T.collate(entries[16:])];results[str(seed)]={};resumes[str(seed)]={}
            for arm in 'NSGR':
                results[str(seed)][arm]=smoke(arm,seed,batches);resumes[str(seed)][arm]=resume(arm,seed,batches)
                if seed==1:
                    now=results['1'][arm]
                    comparisons.append({'seed':seed,'arm':arm,'old_label_record':str(C.S2/'342_result_四臂CPU冒烟.json'),
                        'stepwise_losses':[{'step':j,'old':old[arm]['log'][j]['losses'],'new':now['log'][j]['losses'],
                            'delta':{k:now['log'][j]['losses'][k]-old[arm]['log'][j]['losses'][k] for k in now['log'][j]['losses']}} for j in range(20)],
                        'old_per_class_nonzero_gradient_steps':old[arm].get('per_class_nonzero_gradient_steps'),
                        'new_per_class_nonzero_gradient_steps':now.get('per_class_nonzero_gradient_steps')})
        C.write('343_result_3b新标签四臂冒烟与续跑.json',{'status':'已验证（有界CPU工程测试）；完整缓存仍因#1466失败','cache_dataset_status':'仅使用已独立验收的两seed各前32窗；未跳过或替换这64窗；不构成6470条完整缓存通过','same_development_ids_per_seed':ids,
            'arms':results,'resume':resumes,'phase2_old_label_comparisons':comparisons,
            'old_seed2_comparison':'未执行；阶段2旧标签只有seed1记录，未补做旧标签新运行。',
            'global_contract':'本次未构造global模型，所有优化参数仅来自head；global零grad/state合同继承阶段2，本次不冒称重测。',
            '已执行':COUNTS,'未执行':['真实训练','完整1218步运行','global前向','GPU','阶段5'],
            '未核':['主控独立重跑','完整训练后G有效性'],'source_sha256':C.sources()})
    except Exception as ex:
        C.write('343_result_3b新标签冒烟_实现失败.json',{'status':'实现失败','exception':repr(ex),'traceback':traceback.format_exc(),
            '已执行':COUNTS,'未执行':['真实训练','阶段5'],'未核':['新标签冒烟和续跑'],'source_sha256':C.sources()});raise

if __name__=='__main__':main()
