"""Focused additional checks: nonzero signal wiring and actual common checkpoint.
Uses a parameter-free stand-in for delta on a throwaway decoder, never edits
baseline weights and never takes an optimizer step.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from integration import HERE, activate_local_snapshot, load_baseline_state, sha
activate_local_snapshot()
from model import P3ReadoutDEIM
from readout import sampling_diagnostics
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
checks=[]
def check(name, ok, **details):
    checks.append(dict(name=name,passed=bool(ok),**details))
    if not ok: raise AssertionError(name)

class KnownDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_p3=None
    def forward(self, semantic, p3, z0):
        self.seen_p3=p3.detach().clone()
        return z0.new_tensor([.1,-.2,.3,-.1]).expand_as(z0)

def run():
    plan=json.loads((HERE/'configs/spatial_seed1.json').read_text(encoding='utf-8'))
    paths=json.loads((HERE/'local_paths.json').read_text(encoding='utf-8'))
    check('actual_common_init_sha',sha(paths['common_init'])==paths['common_init_sha256'])
    ck=torch.load(paths['common_init'],map_location='cpu',weights_only=False)
    state=ck['ema']['module'] if 'ema' in ck else ck['model']
    candidate=P3ReadoutDEIM(plan['model_recipe'],'spatial')
    baseline=P3ReadoutDEIM(plan['model_recipe'],'baseline')
    check('actual_common_init_candidate_strict_native_keys',load_baseline_state(candidate,state)==6)
    check('actual_common_init_baseline_strict',load_baseline_state(baseline,state)==0)
    # Three unequal levels and raw P3 verify slice/order, not an image experiment.
    shapes=[[16,24],[8,12],[4,6]]
    mem=torch.randn(2,384+96+24,256,requires_grad=True)
    baseline.train();candidate.train()
    native=baseline.decoder._get_decoder_input(mem,shapes)
    candidate.decoder.local_readout=KnownDelta()
    output=candidate.decoder._get_decoder_input(mem,shapes)
    delta=mem.new_tensor([.1,-.2,.3,-.1])
    check('nonzero_delta_reaches_reference',torch.equal(output[1],native[1]+delta))
    check('nonzero_delta_reaches_encoder_aux',torch.equal(output[2][0],(native[1]+delta).sigmoid()))
    check('native_content_and_all_scores_unchanged',torch.equal(output[0],native[0]) and torch.equal(output[4],native[4]))
    expected=mem[:,:384].transpose(1,2).reshape(2,256,16,24)
    check('reads_raw_projected_P3_before_mask',torch.equal(candidate.decoder.local_readout.seen_p3,expected))
    check('reference_detached_encoder_supervised',not output[1].requires_grad and output[2][0].requires_grad)
    zeros=torch.zeros(1,1,4)
    d=sampling_diagnostics(zeros,'center',80,80)
    check('center_duplicates_count_eight',d['duplicate_after_clamp_points'].item()==8)
    z=torch.logit(torch.tensor([[[.99,.99,.8,.8]]]))
    d=sampling_diagnostics(z,'spatial',80,80)
    check('border_observer_counts',d['outside_normalized_image_points'].item()==5 and
          d['border_clamped_points'].item()==5,values={k:v.tolist() for k,v in d.items()})
    # Observer must not consume RNG or mutate supplied logits.
    rng=torch.get_rng_state().clone();z_before=z.clone()
    sampling_diagnostics(z,'spatial',80,80)
    check('observer_passive',torch.equal(z,z_before) and torch.equal(rng,torch.get_rng_state()))
    check('no_CUDA',not torch.cuda.is_initialized())

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    result={'status':'RUNNING','checks':checks,'full_detector_forwards':0,
            'decoder_input_calls':2,'optimizer_constructions':0,'optimizer_steps':0,
            'common_init_read_only':True,'known_delta_is_test_double_not_model_parameter_update':True}
    try:run();result['status']='PASS'
    except Exception:result.update(status='FAIL',traceback=traceback.format_exc());raise
    finally:
        with a.report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps({'status':result['status'],'checks':len(checks)}))
