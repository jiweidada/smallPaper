from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from conservative_scale_exchange import ScaleExchange,down,up,channel_map
torch.manual_seed(20261103)
features=[torch.randn(2,12,s,s,dtype=torch.float64) for s in (16,8,4)]
q=ScaleExchange(12,'Q').double();m=ScaleExchange(12,'M').double()
m.load_state_dict(q.state_dict(),strict=True)
assert sum(p.numel() for p in q.parameters())==sum(p.numel() for p in m.parameters())
for module in (q,m):
    out=module(features)
    assert all(torch.equal(a,b) for a,b in zip(features,out))
with torch.no_grad():
    for pair in q.pairs:
        pair.frame.normal_(0,.1);pair.beta.fill_(.7)
        pair.gates.weight.normal_(0,.2);pair.gates.bias.copy_(torch.tensor([.9,-.7],dtype=torch.float64))
m.load_state_dict(q.state_dict(),strict=True)
a0=q.pairs[0].orthogonal_frame();a1=q.pairs[1].orthogonal_frame()
assert max((a.T@a-torch.eye(12,dtype=torch.float64)).abs().max().item() for a in (a0,a1))<1e-12
def total(xs):
    return channel_map(a1@a0,down(down(xs[0])))+channel_map(a1,down(xs[1]))+xs[2]
outq=q(features);outm=m(features)
qerr=(total(outq)-total(features)).abs().max().item()
merr=(total(outm)-total(features)).abs().max().item()
hbefore=features[0]-up(down(features[0]))
herr=(outq[0]-up(down(outq[0]))-hbefore).abs().max().item()
assert qerr<1e-12 and herr<1e-12 and merr>1e-4
loss=sum(v.square().mean() for v in outq)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters())
assert all(p.grad.abs().max()>0 for p in q.parameters())
result={'status':'PASS','scope':'synthetic algebra/gradient check; no detector or AP test',
 'identity_initialization_bitwise':True,'orthogonal_frame_max_error':max((a.T@a-torch.eye(12,dtype=torch.float64)).abs().max().item() for a in (a0,a1)),
 'Q_global_coarse_frame_sum_max_error':qerr,'M_global_coarse_frame_sum_max_drift':merr,
 'Q_finest_block_residual_max_error':herr,'all_new_parameter_gradients_nonzero_finite':True,
 'extra_parameters_at_C192':sum(p.numel() for p in ScaleExchange(192,'Q').parameters()),
 'new_model_forwards':0,'new_optimizer_updates':0,
 'limits':['feature-coordinate identity is not conservation of semantic information or L2 energy',
           'signed alpha permits amplification of differences; no nonexpansiveness guarantee',
           'global sum preserved only with same fixed per-forward channel maps; maps learn across updates',
           'BF16 output casts introduce rounding; exact equalities hold in exact arithmetic, not promised bitwise after training']}
(ROOT/'math_check_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result))
