"""Synthetic diagnostic tests; constructs no model and forwards no real image."""
import argparse,faulthandler,json,pathlib,sys
faulthandler.dump_traceback_later(60,repeat=True)
ROOT=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'extension_v2'),str(ROOT/'probe')]
import torch
from diagnostic_extension_v2 import analyze,tie_diagnostics

def main(seed):
    torch.set_num_threads(1);torch.manual_seed(6201020)
    y=json.loads((ROOT/'baseline_recipes.json').read_text())[str(seed)]
    b,q,c=2,6,10
    target=[{'boxes':torch.tensor([[.2,.2,.1,.1],[.4,.3,.1,.1],[.6,.5,.2,.1],[.8,.7,.1,.1]]),'labels':torch.tensor([0,1,2,3])} for _ in range(b)]
    def layer(queries=q):
        boxes=torch.rand(b,queries,4);boxes[:,:,2:]=boxes[:,:,2:]*.15+.05
        return {'pred_boxes':boxes,'pred_logits':torch.randn(b,queries,c),
                'pred_corners':torch.randn(b,queries,132),'ref_points':torch.full((b,queries,4),.25)}
    def head(v):return {k:v[k].clone() for k in ['pred_boxes','pred_logits']}
    out=layer();out.update(up=torch.tensor([.5]),reg_scale=torch.tensor([4.]))
    out['aux_outputs']=[layer(),layer()]
    for v in out['aux_outputs']:v.update(teacher_corners=out['pred_corners'],teacher_logits=out['pred_logits'])
    out['pre_outputs']=head(out);out['enc_aux_outputs']=[head(out)];out['enc_meta']={'class_agnostic':False}
    out['dn_outputs']=[layer(16) for _ in range(3)];out['dn_pre_outputs']=head(out['dn_outputs'][0])
    out['dn_meta']={'dn_num_group':2,'dn_positive_idx':[torch.arange(8),torch.arange(8)],'dn_num_split':[16,q]}
    report=analyze(out,target,y,check_grad=True)
    assert report['DN_geometry']['present'] and report['DN_geometry']['dn_num_group']==2
    assert len(report['DN_geometry']['branches'])==4
    assert all(g['positive_DN_slots']==2 for v in report['DN_geometry']['branches'].values() for im in v['images'] for g in im['GT'])
    tie=tie_diagnostics([[0,0,2],[0,1,2],[1,0,3]],{0:1,1:0},2)
    assert tie['top_vote_tied_queries']==1 and tie['native_differs_min_GT_queries']==1
    # Exercise the N>Q path with five complete branch matchings capped at Q=6.
    dense=[{'boxes':torch.cat([t['boxes'],t['boxes']]),'labels':torch.cat([t['labels'],t['labels']])} for t in target]
    overflow={k:v for k,v in out.items() if not k.startswith('dn_')}
    dense_report=analyze(overflow,dense,y,check_grad=False)
    assert all(r['overflow'] and not r['capacity_eligible'] and r['solver']['fallback'] for r in dense_report['inputs'])
    assert all(r['changed_queries']['coverage_constrained']==0 for r in dense_report['inputs'])
    result={'state':'CPU_extension_preflight_passed','seed':seed,'DN_additive_sums_match_native':True,
            'DN_positive_slots_per_GT':2,'top_frequency_tie_diagnostic_passed':True,'N_greater_Q_fallback_passed':True,
            'DDF_value_and_leaf_gradient_check_passed':True,'model_constructions':0,'model_forwards':0,'optimizer_steps':0}
    (ROOT/f'extension_preflight_seed{seed}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    faulthandler.cancel_dump_traceback_later()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);main(p.parse_args().seed)
