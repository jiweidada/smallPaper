"""CPU-only construction/state and synthetic-loss tests; no detector forward."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'probe'))
import torch
from probe_runtime import verify_package,build_model,build_criterion,sha
from evaluate_rules import analyze

def main(seed):
    torch.set_num_threads(1);verify_package();plan=json.loads((ROOT/'sample_plan.json').read_text());y=json.loads((ROOT/'baseline_recipes.json').read_text())[str(seed)]
    cp=ROOT/'assets'/f'baseline24_seed{seed}.pth';assert sha(cp)==plan['checkpoints'][str(seed)]['sha256']
    value=torch.load(cp,map_location='cpu',weights_only=False);assert value['last_epoch']==23
    model=build_model(y);model.load_state_dict(value['model'],strict=True);crit=build_criterion(y)
    torch.manual_seed(19261019);b,q,c=2,6,10
    target=[{'boxes':torch.tensor([[.2,.2,.1,.1],[.4,.3,.1,.1],[.6,.5,.2,.1],[.8,.7,.1,.1]]),'labels':torch.tensor([0,1,2,3])} for _ in range(b)]
    def layer():
        boxes=torch.rand(b,q,4);boxes[:,:,2:]=boxes[:,:,2:]*.15+.05
        return {'pred_boxes':boxes,'pred_logits':torch.randn(b,q,c),'pred_corners':torch.randn(b,q,132),'ref_points':torch.full((b,q,4),.25)}
    out=layer();out.update(up=torch.tensor([.5]),reg_scale=torch.tensor([4.]))
    out['aux_outputs']=[layer(),layer()]
    for x in out['aux_outputs']:x.update(teacher_corners=out['pred_corners'],teacher_logits=out['pred_logits'])
    out['pre_outputs']={k:v.clone() for k,v in out.items() if k in ['pred_boxes','pred_logits']}
    out['enc_aux_outputs']=[{k:v.clone() for k,v in out.items() if k in ['pred_boxes','pred_logits']}];out['enc_meta']={'class_agnostic':False}
    report=analyze(out,target,y,check_grad=True)
    assert len(report['branch_names'])==5
    result={'state':'CPU_preflight_passed','seed':seed,'model_state_strict_load':True,'model_state_entries':len(value['model']),'checkpoint_keys':sorted(value),'synthetic_losses_finite':True,'DDF_value_and_leaf_gradient_checks':True,'model_forwards':0,'real_input_forwards':0,'optimizer_steps':0,'torch':torch.__version__,'plan_sha256':sha(ROOT/'sample_plan.json')}
    (ROOT/f'preflight_seed{seed}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);main(p.parse_args().seed)
