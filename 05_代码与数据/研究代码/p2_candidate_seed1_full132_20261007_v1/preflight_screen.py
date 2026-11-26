"""Development checks on the actual native factory, criterion, EMA and AMP."""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
import p2_candidate
from engine.core import YAMLConfig
from engine.misc import dist_utils
from screen_runtime import strict_common_init, tensor_digest, NEW_KEY
import hashlib


def seed():
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--arm', required=True)
    p.add_argument('--init', required=True); p.add_argument('--cpu-only', action='store_true')
    args = p.parse_args(); root = Path(args.root)
    torch.set_num_threads(6)
    report = {'arm': args.arm, 'torch': torch.__version__, 'checks': {}}
    seed(); bcfg = YAMLConfig(str(root/'base.yml')); base = bcfg.model
    base_rng = torch.get_rng_state().clone(); strict_common_init(base, args.init)
    seed(); pcfg = YAMLConfig(str(root/'p2_quota60.yml')); candidate = pcfg.model
    assert torch.equal(base_rng, torch.get_rng_state())
    report['checks']['construction_rng_equal'] = True
    strict_common_init(candidate, args.init)
    bstate, pstate = base.state_dict(), candidate.state_dict()
    assert set(pstate)-set(bstate) == {NEW_KEY}
    assert all(torch.equal(v, pstate[k]) for k,v in bstate.items())
    assert torch.count_nonzero(pstate[NEW_KEY]) == 0
    report['checks'].update(common_weights_equal=True, new_zero_parameters=pstate[NEW_KEY].numel())
    ema = copy.deepcopy(candidate).eval()
    assert ema.decoder._get_decoder_input.__self__ is ema.decoder
    assert ema.decoder.query_recruitment is not candidate.decoder.query_recruitment
    report['checks']['deepcopy_binding'] = True
    shapes = [(8,8),(4,4),(2,2)]
    base.decoder.eval_spatial_size = candidate.decoder.eval_spatial_size = None
    base.decoder.train(); candidate.decoder.train()
    # Use 640 shapes so both pools have enough candidates and use real priors.
    shapes = [(80,80),(40,40),(20,20)]
    seed(); memory = torch.randn(1,8400,256)
    candidate.decoder._p2_context = (torch.randn(1,64,160,160), memory[:,:6400].transpose(1,2).reshape(1,256,80,80))
    with torch.no_grad():
        native = base.decoder._get_decoder_input(memory, shapes)
        new = candidate.decoder._get_decoder_input(memory, shapes)
    # A separate P2 bank cannot perturb the original bank's 240 selected queries.
    assert torch.equal(native[0][:,:240], new[0][:,:240])
    assert torch.equal(native[1][:,:240], new[1][:,:240])
    assert new[0].shape == (1,300,256)
    report['checks']['native_old_top240_equal'] = True
    report['candidate_selection'] = candidate.decoder._p2_last_info
    candidate.decoder._p2_context = None
    from screen_runtime import install_runtime
    from engine.solver.det_solver import DetSolver
    from types import SimpleNamespace
    pcfg.device = 'cpu'
    install_runtime(pcfg, root/f'preflight_guard_{args.arm}')
    fixture = DetSolver(pcfg)
    fixture.model = candidate; fixture.ema = pcfg.ema
    fixture.optimizer = pcfg.optimizer; fixture.scaler = pcfg.scaler
    fixture.last_epoch = -1; fixture.self_lr_scheduler = False
    fixture.train_dataloader = SimpleNamespace(dataset=SimpleNamespace(category2name={},remap_mscoco_category=False))
    saved = fixture.state_dict()
    marker = candidate.decoder.query_recruitment.project.weight.clone()
    wrong = dict(saved); wrong['screen_identity'] = copy.deepcopy(saved['screen_identity'])
    wrong['screen_identity']['config']['P2CandidateDEIM']['p2_budget'] = 59
    try: fixture.load_state_dict(wrong)
    except RuntimeError: pass
    else: raise AssertionError('Wrong quota checkpoint accepted')
    assert fixture.last_epoch == -1 and torch.equal(marker,candidate.decoder.query_recruitment.project.weight)
    fixture.load_state_dict(saved)
    report['checks']['checkpoint_config_rejected_before_mutation'] = True
    report['checks']['native_solver_state_roundtrip'] = True
    del fixture, saved, wrong, marker
    del base, candidate, ema, memory, native, new, bcfg, pcfg
    import gc; gc.collect()
    if not args.cpu_only:
        seed(); cfg = YAMLConfig(str(root/f'{args.arm}.yml'))
        model = cfg.model; report['init_load'] = strict_common_init(model, args.init)
        model.to('cuda:0').train(); criterion = cfg.criterion.to('cuda:0'); criterion.set_model(model)
        optimizer = cfg.optimizer; scaler = cfg.scaler
        ids = [id(v) for g in optimizer.param_groups for v in g['params']]
        assert len(ids) == len(set(ids)) and set(ids) == {id(v) for v in model.parameters() if v.requires_grad}
        report['checks']['optimizer_coverage'] = True
        loader = cfg.train_dataloader; loader.set_epoch(0)
        images, targets = next(iter(loader))
        h = hashlib.sha256(); tensor_digest(images,h); tensor_digest(targets,h)
        report['real_batch'] = {'sha256': h.hexdigest(), 'shape': list(images.shape), 'gt_counts': [len(t['labels']) for t in targets]}
        images = images.to('cuda:0'); targets = [{k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
        ema = cfg.ema.to('cuda:0')
        # Capacity check also covers the largest configured multiscale input.
        from torch.nn.functional import interpolate
        images = interpolate(images, size=(800,800), mode='bilinear', align_corners=False)
        updates = {'count':0}; optimizer.register_step_post_hook(lambda *_: updates.__setitem__('count',updates['count']+1))
        torch.cuda.reset_peak_memory_stats()
        report['amp_steps'] = []
        for i in range(10):
            optimizer.zero_grad(set_to_none=True)
            # Real GT, then one empty sample, then normal GT again.
            step_targets = copy.deepcopy(targets)
            if i == 1:
                for k in ('boxes','labels','area','iscrowd'):
                    if k in step_targets[0]: step_targets[0][k] = step_targets[0][k][:0]
            with torch.autocast('cuda', dtype=torch.float16): outputs = model(images, targets=step_targets)
            with torch.autocast('cuda', enabled=False):
                losses = criterion(outputs, step_targets, epoch=0, step=i, global_step=i, epoch_step=len(loader))
                loss = sum(losses.values())
            assert torch.isfinite(loss)
            assert outputs['pred_logits'].shape == (4,300,10)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad = model.decoder.query_recruitment.project.weight.grad if args.arm != 'base' else None
            grad_norm = float(grad.float().norm()) if grad is not None else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            inf = sum(float(v.item()) for v in scaler._found_inf_per_device(optimizer).values())
            before = updates['count']; scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update(); ema.update(model)
            report['amp_steps'].append({'step':i,'loss':float(loss),'scale':scale,'found_inf':inf,
                                        'updated':updates['count']>before,'p2_gradient_norm':grad_norm,
                                        'dn_split': outputs.get('dn_meta',{}).get('dn_num_split')})
            if i >= 2 and updates['count'] >= 2: break
        assert updates['count'] >= 2, 'No effective AMP optimizer updates in bounded development fixture'
        if args.arm != 'base':
            assert model.decoder.query_recruitment.project.weight.norm() > 0
            assert any(s['updated'] and s['p2_gradient_norm'] > 0 for s in report['amp_steps'])
            report['p2_weight_norm'] = float(model.decoder.query_recruitment.project.weight.norm())
        torch.cuda.synchronize()
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['checks']['real_batch_amp_optimizer_and_dn'] = True
        # Verify model/EMA reload against saved states, including branch.
        criterion.reset_model()
        state = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()}
        path = root/f'preflight_{args.arm}_fixture.pth'; torch.save(state,path)
        loaded = torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(loaded['model'],strict=True); ema.load_state_dict(loaded['ema'])
        report['checks']['model_ema_roundtrip'] = True
    report['passed'] = True
    (root/f'preflight_{args.arm}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
