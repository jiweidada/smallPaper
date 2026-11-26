"""Extend the CPU criterion check to dense DN and density branches, no detector."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import torch

helper = Path(__file__).with_name('cpu_full_native_criterion_probe.py')
nodes = ast.parse(helper.read_text(encoding='utf-8')).body
prefix = []
for node in nodes:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='results' for t in node.targets):
        break
    prefix.append(node)
env = {'__name__':'a25_setup_only','__file__':str(helper)}
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(helper),'exec'),env)
assert 'results' not in env and not torch.cuda.is_initialized()

def fixture():
    B,Q,DN,C = 2,500,1804,12
    counts = [10,902]
    box = torch.tensor([0.5,0.5,0.05,0.05])
    leaves = []
    def item(n,stage):
        logits = torch.zeros((B,n,C),requires_grad=True)
        boxes = box.repeat(B,n,1).requires_grad_()
        pattern = torch.linspace(-0.5,0.5,33).repeat(4)
        corners = (pattern.repeat(B,n,1)*(stage+1)*0.2+stage*0.5).to(torch.float16).requires_grad_()
        leaves.extend([logits,boxes,corners])
        return {'pred_logits':logits,'pred_boxes':boxes,'pred_corners':corners,'ref_points':box.repeat(B,n,1)}
    main = [item(Q,i) for i in range(3)]
    dn = [item(DN,i) for i in range(3)]
    outputs = dict(main[-1])
    outputs.update(up=torch.tensor([0.5]),reg_scale=torch.tensor([4.0]))
    for group in [main,dn]:
        for row in group:
            row['teacher_corners'] = group[-1]['pred_corners'].detach()
            row['teacher_logits'] = group[-1]['pred_logits'].detach()
    outputs['aux_outputs'] = main[:-1]
    outputs['pre_outputs'] = {k:main[0][k] for k in ['pred_logits','pred_boxes']}
    outputs['enc_aux_outputs'] = [{k:main[0][k] for k in ['pred_logits','pred_boxes']}]
    outputs['enc_meta'] = {'class_agnostic':False}
    outputs['batch_queries_num'] = [250,500]
    outputs['dn_outputs'] = dn
    outputs['dn_pre_outputs'] = {k:dn[0][k] for k in ['pred_logits','pred_boxes']}
    outputs['dn_meta'] = {'dn_num_group':1,'dn_num_split':[DN,Q],
                          'dn_positive_idx':[torch.arange(n) for n in counts]}
    density = torch.ones((B,1,8,8),requires_grad=True)
    reg = torch.tensor([0.2,0.8],requires_grad=True)
    leaves.extend([density,reg])
    gt_density = torch.stack([torch.full((1,8,8),n/64) for n in counts])
    outputs['defe'] = {'min_num_select':250,'max_num_select':500,
                       'reg_value':reg,'defe_feature':density,'gt_density_map':gt_density}
    targets = [{'labels':torch.ones(n,dtype=torch.long),'boxes':box.repeat(n,1)} for n in counts]
    return outputs,targets,leaves

cases = []
for stable in [False,True]:
    criterion = env['build_criterion'](stable)
    outputs,targets,leaves = fixture()
    raw = []
    original = torch.nan_to_num
    def observe(value,*args,**kwargs):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_filename==str(env['source_path']) and 'k' in caller.f_locals:
            raw.append({'key':caller.f_locals['k'],'dtype':str(value.dtype),
                        'finite':bool(torch.isfinite(value).all()),'value_text':str(float(value.detach()))})
        return original(value,*args,**kwargs)
    torch.nan_to_num = observe
    try:
        losses = criterion(outputs,targets,epoch=0)
    finally:
        torch.nan_to_num = original
    bad = [x['key'] for x in raw if not x['finite']]
    assert 'defe_reg_loss' in losses and 'defe_density_loss' in losses
    assert 'loss_ddf_dn_2' in losses
    gradients = None
    if stable:
        assert not bad
        sum(losses.values()).backward()
        present = [x.grad for x in leaves if x.grad is not None]
        gradients = {'present_tensors':len(present),'leaf_tensors':len(leaves),
                     'all_present':len(present)==len(leaves),
                     'all_present_gradients_finite':all(bool(torch.isfinite(g).all()) for g in present)}
        assert gradients['all_present'] and gradients['all_present_gradients_finite']
    else:
        assert bad == ['loss_ddf_dn_2'],bad
    cases.append({'stable_zero_only':stable,'raw_losses':raw,'nonfinite_raw_keys':bad,
                  'returned_all_finite':all(bool(torch.isfinite(v)) for v in losses.values()),
                  'synthetic_gradients':gradients})
assert not torch.cuda.is_initialized()
assert hashlib.sha256(env['source_path'].read_bytes()).hexdigest()==env['source_hash']
print(json.dumps({
    'status':'DENSE_DN_AND_DENSITY_BRANCH_SYNTHETIC_ZERO_FIX_VERIFIED',
    'device':'cpu','CUDA_initialized':False,'detector_forwards':0,'optimizer_updates':0,
    'complete_criterion_calls':2,'native_matcher_calls':env['matcher_calls'][0],
    'GT_counts':[10,902],'queries':[250,500],'padded_main_queries':500,'DN_queries':1804,
    'source_sha256':env['source_hash'],'cases':cases,
    'scope':'CPU synthetic full-criterion interface including auxiliary/pre/encoder/DN/density branches; no image model, actual A22 root-cause replay, AMP GPU kernel validation or AP.',
    'old_training_relaunched':False,
},ensure_ascii=False,allow_nan=False))
