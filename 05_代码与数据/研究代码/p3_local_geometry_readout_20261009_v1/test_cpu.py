"""CPU engineering acceptance only. No optimizer, training loop, or AP sweep."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from integration import (HERE, activate_local_snapshot, build_criterion,
                         load_baseline_state, verify_sources)
activate_local_snapshot()
from model import P3ReadoutDEIM
from readout import P3LocalReadout, sample_locations, sample_nine

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
REPORT = {'kind': 'engineering_acceptance_not_scientific_efficacy', 'checks': [],
          'optimizer_constructions': 0, 'optimizer_steps': 0, 'AP_evaluations': 0,
          'full_detector_forwards': 0, 'model_parameter_updates': 0}


def check(name, condition, **details):
    REPORT['checks'].append(dict(name=name, passed=bool(condition), **details))
    if not condition:
        raise AssertionError(name)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def digest_parameters(model):
    h = hashlib.sha256()
    for name, tensor in model.named_parameters():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def forward(model, *args):
    REPORT['full_detector_forwards'] += 1
    return model(*args)


def run():
    plan = json.loads((HERE/'configs'/'baseline_seed1.json').read_text(encoding='utf-8'))
    recipe = plan['model_recipe']
    rng = torch.get_rng_state().clone()
    spatial = P3LocalReadout('spatial')
    center = P3LocalReadout('center')
    check('new_initialization_preserves_CPU_rng', torch.equal(rng, torch.get_rng_state()))
    check('added_parameter_count_78500', sum(p.numel() for p in spatial.parameters()) == 78500)
    check('matched_control_initialization', equal_tree(spatial.state_dict(), center.state_dict()))
    # Independent hand-computable pixel-center/border test on a non-square map.
    ramp = torch.tensor([[[[0., 1., 2., 3.], [10., 11., 12., 13.]]]])
    points = torch.tensor([[[[.125,.25],[.375,.25],[.625,.25],[.875,.25],
                             [.125,.75],[.875,.75],[0.,0.],[1.,1.],[.5,.5]]]])
    actual = sample_nine(ramp, points).flatten()
    expected = torch.tensor([0.,1.,2.,3.,10.,13.,0.,13.,6.5])
    check('bilinear_xy_align_false_border', torch.equal(actual, expected))
    box = torch.tensor([[[.5,.4,.2,.4]]])
    loc = sample_locations(torch.logit(box), 'spatial')
    expected_loc = torch.tensor([[[[.35,.1],[.5,.1],[.65,.1],
                                  [.35,.4],[.5,.4],[.65,.4],
                                  [.35,.7],[.5,.7],[.65,.7]]]])
    check('nine_positions_row_major', torch.allclose(loc, expected_loc, atol=1e-7))
    check('center_nine_repeats', torch.equal(sample_locations(torch.logit(box), 'center'),
                                            box[...,:2].unsqueeze(2).expand(-1,-1,9,-1)))
    z = torch.zeros(1,2,4,requires_grad=True)
    check('sampling_coordinate_detach', not sample_locations(z, 'spatial').requires_grad)
    check('empty_query_shape', spatial(torch.empty(1,0,256),torch.zeros(1,256,4,4),
                                      torch.empty(1,0,4)).shape == (1,0,4))
    torch.random.default_generator.manual_seed(124)
    before_rng = torch.get_rng_state().clone()
    baseline = P3ReadoutDEIM(recipe, 'baseline')
    after_rng = torch.get_rng_state().clone()
    native_state = deepcopy(baseline.state_dict())
    models = {'baseline': baseline}
    for arm in ('center', 'spatial'):
        torch.set_rng_state(before_rng)
        model = P3ReadoutDEIM(recipe, arm)
        check(arm+'_common_model_rng', torch.equal(after_rng, torch.get_rng_state()))
        check(arm+'_common_native_initialization',
              all(torch.equal(v, model.state_dict()[k]) for k,v in native_state.items()))
        check(arm+'_strict_baseline_load', load_baseline_state(model, native_state) == 6)
        models[arm] = model
    # Full native model at fixed 640. This is synthetic interface verification.
    x = torch.rand(1,3,640,640)
    outputs, times = {}, {}
    for arm, model in models.items():
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            outputs[arm] = forward(model,x)
        times[arm] = time.perf_counter()-start
    check('full_640_eval_center_bitwise', equal_tree(outputs['baseline'],outputs['center']))
    check('full_640_eval_spatial_bitwise', equal_tree(outputs['baseline'],outputs['spatial']))
    REPORT['one_forward_CPU_seconds_not_latency_benchmark'] = times
    # Training branch, native denoising and GO criterion; no optimizer step.
    x = torch.rand(2,3,128,128)
    targets = [dict(labels=torch.tensor([3,5]), boxes=torch.tensor([[.5,.5,.2,.3],[.3,.4,.1,.1]])),
               dict(labels=torch.empty(0,dtype=torch.long), boxes=torch.empty(0,4))]
    training_rng = torch.get_rng_state().clone()
    train_out = {}
    before_parameters = {arm:digest_parameters(m) for arm,m in models.items()}
    for arm, model in models.items():
        model.train()
        torch.set_rng_state(training_rng)
        train_out[arm] = forward(model, x, deepcopy(targets))
    check('native_DN_present', 'dn_outputs' in train_out['spatial'])
    check('train_center_bitwise_including_DN', equal_tree(train_out['baseline'],train_out['center']))
    check('train_spatial_bitwise_including_DN', equal_tree(train_out['baseline'],train_out['spatial']))
    criterion = build_criterion(plan['criterion_recipe'])
    losses = {arm:criterion(out, deepcopy(targets), epoch=0) for arm,out in train_out.items()}
    check('criterion_center_bitwise', equal_tree(losses['baseline'],losses['center']))
    check('criterion_spatial_bitwise', equal_tree(losses['baseline'],losses['spatial']))
    check('all_native_losses_finite', all(torch.isfinite(v) for d in losses.values() for v in d.values()))
    REPORT['native_loss_keys'] = sorted(losses['spatial'])
    for arm in ('center','spatial'):
        branch = models[arm].decoder.local_readout
        loss = losses[arm]
        enc = loss['loss_bbox_enc_0'] + loss['loss_giou_enc_0']
        grad = torch.autograd.grad(enc, branch.output.weight, retain_graph=True)[0]
        check(arm+'_encoder_box_loss_reaches_last_layer', torch.count_nonzero(grad).item() > 0,
              gradient_norm=grad.norm().item())
        other = sum(v for k,v in loss.items() if not k.endswith('_enc_0'))
        no_direct = torch.autograd.grad(other, tuple(branch.parameters()),
                                        retain_graph=True, allow_unused=True)
        check(arm+'_decoder_losses_do_not_cross_detached_reference', all(g is None for g in no_direct))
        sum(loss.values()).backward()
        grads = {n:p.grad.norm().item() if p.grad is not None else None for n,p in branch.named_parameters()}
        check(arm+'_zero_init_first_step_gradients',
              grads['output.weight'] > 0 and grads['output.bias'] > 0 and
              all(grads[n] == 0 for n in ('projection.weight','projection.bias','hidden.weight','hidden.bias')),
              gradients=grads)
    check('no_model_parameter_updates', all(digest_parameters(m)==before_parameters[a] for a,m in models.items()))
    # Round trip and deepcopy exercise the native EMA storage requirement.
    candidate = models['spatial'].eval()
    clone = deepcopy(candidate)
    result = clone.load_state_dict(candidate.state_dict(), strict=True)
    check('candidate_state_dict_strict_and_deepcopy', not result.missing_keys and not result.unexpected_keys)
    bad = dict(native_state); bad.pop(next(iter(bad)))
    try:
        load_baseline_state(models['center'], bad)
    except RuntimeError:
        check('missing_native_key_rejected', True)
    else:
        check('missing_native_key_rejected', False)
    # Native deployment conversion on a copy, original object unchanged.
    clone.deploy()
    native_deployed = deepcopy(models['baseline'].eval()).deploy()
    x = torch.rand(1,3,640,640)
    with torch.no_grad():
        original = forward(candidate,x)
        deployed = forward(clone,x)
        native_before = forward(models['baseline'],x)
        native_after = forward(native_deployed,x)
    REPORT['deployment_difference'] = {
        arm: {k:(left[k]-right[k]).abs().max().item() for k in left}
        for arm,left,right in [('baseline',native_before,native_after),
                               ('spatial',original,deployed)]}
    # Native deploy folds Conv/BN and precomputes FDR with a different arithmetic
    # path. Check identity against the same native conversion, not unfused eval.
    check('undeployed_zero_residual_matches_native', equal_tree(original,native_before))
    check('deployed_zero_residual_matches_native_bitwise', equal_tree(deployed,native_after))
    check('deploy_finite', all(torch.isfinite(v).all() for v in deployed.values()))
    # All-empty GT batch, rectangular/multi-scale shape, native train interface.
    empty = [dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,4)) for _ in range(2)]
    candidate.train()
    out = forward(candidate,torch.rand(2,3,128,160),empty)
    empty_losses = criterion(out,empty,epoch=0)
    check('all_empty_GT_rectangular_finite', all(torch.isfinite(v) for v in empty_losses.values()))
    verify_sources()
    check('shared_sources_unchanged', True)
    check('CUDA_not_initialized', not torch.cuda.is_initialized())
    REPORT.update(status='PASS', torch_version=torch.__version__, threads=4,
                  executed_on='CPU', synthetic_images_only=True,
                  pending=['GPU/AMP', 'actual latency and GPU memory', 'training efficacy',
                           'full native training runner and evaluator integration'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('A previous engineering receipt must not be overwritten')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    try:
        run()
    except Exception:
        REPORT.update(status='FAIL', traceback=traceback.format_exc())
        raise
    finally:
        REPORT['elapsed_seconds'] = time.perf_counter()-started
        with args.report.open('x',encoding='utf-8') as f:
            json.dump(REPORT,f,ensure_ascii=False,indent=2)
        print(json.dumps({'status':REPORT.get('status'), 'checks':len(REPORT['checks']),
                          'report':str(args.report)},ensure_ascii=False))
