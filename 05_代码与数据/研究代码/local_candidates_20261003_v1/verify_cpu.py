"""Meaningful CPU checks: geometry, information paths, native gradients and query invariants."""
import argparse
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import bootstrap
from bootstrap import ROOT
from factory import load_config, build_model, build_criterion, verify_snapshot, load_common_init
from geometry import (windows, make_views, align_global_p3, targets_for_window,
                      map_local_prediction, merge_predictions)
from dual_view import P3Interaction, paired_loss
from query_coverage import coverage_indices
from dataset import PairedCocoImages, paired_collate, native_query_loader


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(314)
    results = []
    started = time.time()
    def check(name, condition):
        if not condition: raise AssertionError(name)
        results.append({'name': name, 'passed': True})
    def close(name, a, b, tol=2e-5):
        check(name, torch.allclose(a, b, rtol=0, atol=tol))
    manifest = verify_snapshot()
    check('native leaf files match copied source hashes', len(manifest['files']) == 31)
    check('edge anchored native windows', windows(1300, 900) == [
        (0, 0, 640, 640), (512, 0, 1152, 640), (660, 0, 1300, 640),
        (0, 260, 640, 900), (512, 260, 1152, 900), (660, 260, 1300, 900)])
    image = torch.rand(3, 400, 300)
    full, local, crop = make_views(image, windows(300, 400)[0])
    check('native small image crop padded to640', local.shape == (3, 640, 640))
    close('small image is not resized inside native crop', local[:, :400, :300], image, 0)
    check('padding is black', local[:, 400:].count_nonzero() == 0 and local[:, :, 300:].count_nonzero() == 0)
    p3 = torch.rand(1, 2, 80, 80)
    aligned, valid = align_global_p3(p3, torch.tensor([[0., 0., 1., 1.]]), (80, 80))
    close('identity crop feature centres align', aligned, p3, 1e-5)
    check('identity crop all sites valid', bool(valid.all()))
    yy, xx = torch.meshgrid((torch.arange(20) + .5) / 20, (torch.arange(30) + .5) / 30, indexing='ij')
    ramp = torch.stack((xx, yy))[None]
    aligned, _ = align_global_p3(ramp, torch.tensor([[.25, .2, .75, .8]]), (4, 5))
    expected_x = .25 + (torch.arange(5) + .5) / 5 * .5
    expected_y = .2 + (torch.arange(4) + .5) / 4 * .6
    close('anisotropic global warp x coordinates', aligned[0, 0], expected_x[None].expand(4, -1))
    close('anisotropic global warp y coordinates', aligned[0, 1], expected_y[:, None].expand(-1, 5))
    aligned, valid = align_global_p3(ramp, torch.tensor([[0., 0., 2., 2.]]), (4, 4))
    check('padded sites do not import global context', not valid[0, 0, -1, -1] and aligned[0, :, -1, -1].count_nonzero() == 0)
    target = {'image_id': torch.tensor([0]), 'boxes': torch.tensor([[90., 90., 140., 150.], [0., 0., 10., 10.]]),
              'labels': torch.tensor([2, 1]), 'annotation_ids': torch.tensor([0, 7])}
    local_target = targets_for_window(target, (100, 100, 740, 740), (900, 1300))
    close('clipped crop box normalized correctly', local_target['boxes'], torch.tensor([[20., 25., 40., 50.]]) / 640)
    check('id0 annotation preserved and nonintersecting box excluded', local_target['annotation_ids'].tolist() == [0])
    pred = {'boxes': torch.tensor([[0., 0., 20., 30.], [620., 620., 640., 640.]]),
            'scores': torch.tensor([.9, .8]), 'labels': torch.tensor([1, 1])}
    mapped = map_local_prediction(pred, (100, 100, 740, 740), (400, 300))
    close('prediction maps to original and padded centre removed', mapped['boxes'], torch.tensor([[100., 100., 120., 130.]]))
    merged = merge_predictions([mapped, {**mapped, 'scores': torch.tensor([.7])}])
    check('same-class duplicate suppressed once', len(merged['scores']) == 1 and float(merged['scores'][0]) > .89)
    merged = merge_predictions([mapped, {**mapped, 'labels': torch.tensor([2])}])
    check('different classes survive shared merge', len(merged['scores']) == 2)

    fusion = P3Interaction(4, 2)
    a, b = torch.rand(1, 4, 5, 5), torch.rand(1, 4, 5, 5, requires_grad=True)
    valid = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    close('zero initialization preserves local feature exactly', fusion(a, b, valid), a, 0)
    fusion.restore.weight.data.normal_(0, .1)
    fusion(a, b, valid).sum().backward()
    check('learned interaction has gradient into global feature', b.grad is not None and bool(b.grad.abs().sum() > 0))

    # Construct a concentrated high-score population plus useful dispersed P3 tokens.
    xy = torch.rand(1, 500, 2) * .8 + .1
    xy[:, :240] = .12
    xy[:, 240:256] = torch.tensor([[(c % 4 + .5) / 4, (c // 4 + .5) / 4] for c in range(16)])
    wh = torch.full((1, 500, 2), .05); wh[:, 400:] = .1
    anchors = torch.cat((xy, wh), -1).logit()
    logits = torch.full((1, 500, 10), -10.)
    logits[:, :, 0] = torch.linspace(6., 5.5, 500)
    native = logits.max(-1).values.topk(300, dim=-1).indices
    close('reserve0 exactly reproduces native indices', coverage_indices(logits, anchors, reserve=0), native, 0)
    selected = coverage_indices(logits, anchors)
    check('coverage keeps exactly300 unique queries', selected.shape == (1, 300) and len(selected[0].unique()) == 300)
    check('high-score225 retained', set(native[0, :225].tolist()) <= set(selected[0].tolist()))
    cell = (xy[0] * 4).long(); cell = cell[:, 1] * 4 + cell[:, 0]
    check('coverage selects dispersed P3 sites', len(cell[selected[0]].unique()) >= 12)
    bad = anchors.clone(); bad[:, 300:] = torch.inf
    selected_bad = coverage_indices(logits, bad)
    check('nonfinite anchors never selected', bool((selected_bad < 300).all()))
    weak = logits.clone(); weak[:, 240:, 0] = -100
    selected_weak = coverage_indices(weak, anchors)
    check('weak cells backfill without forcing invalid queries', len(selected_weak.unique()) == 300)
    tied = torch.zeros_like(logits)
    check('coverage ties deterministic', torch.equal(coverage_indices(tied, anchors), coverage_indices(tied, anchors)))
    try:
        coverage_indices(logits, torch.full_like(anchors, torch.inf))
    except ValueError:
        check('insufficient valid tokens rejected', True)
    else: raise AssertionError('invalid anchors accepted')

    # Synthetic images/GT validate the real dataset and native augmentation wiring.
    with tempfile.TemporaryDirectory(prefix='deim_candidate_cpu_') as tmp:
        tmp = Path(tmp)
        for index in range(8):
            array = np.full((700, 900, 3), index * 20 + 30, np.uint8)
            Image.fromarray(array).save(tmp / f'{index}.png')
        ann = {'images': [{'id': i, 'file_name': f'{i}.png', 'width': 900, 'height': 700} for i in range(8)],
               'categories': [{'id': i, 'name': str(i)} for i in range(10)],
               'annotations': [{'id': i, 'image_id': i, 'category_id': i % 10,
                                'bbox': [100, 120, 16, 20], 'area': 320, 'iscrowd': 0} for i in range(8)]}
        (tmp / 'ann.json').write_text(json.dumps(ann), encoding='utf8')
        first = PairedCocoImages(tmp, tmp / 'ann.json'); second = PairedCocoImages(tmp, tmp / 'ann.json')
        first.set_epoch(7); second.set_epoch(7)
        x, t = first[0]; y, u = second[0]
        check('pair arms obtain identical deterministic geometry', all(torch.equal(x[k], y[k]) for k in x))
        check('pair arms obtain identical local labels', torch.equal(t['local']['boxes'], u['local']['boxes']))
        samples, targets = paired_collate([first[0]])
        loader = native_query_loader(tmp, tmp / 'ann.json', workers=0)
        loader.dataset.set_epoch(7); loader.collate_fn.set_epoch(7)
        query_images, query_targets = next(iter(loader))
        check('native query augmentation produces true batch8', query_images.shape == (8, 3, 640, 640) and len(query_targets) == 8)
        # Full actual B0 + HybridEncoder + three-layer DFINE paths at640 on CPU.
        cfg = load_config(ROOT / 'configs/dual_independent.yml')
        torch.manual_seed(123); independent = build_model(cfg).eval()
        cfg_aligned = load_config(ROOT / 'configs/dual_aligned.yml')
        torch.manual_seed(123); aligned_model = build_model(cfg_aligned).eval()
        check('all initial native and new states shared exactly', all(torch.equal(v, aligned_model.state_dict()[k])
              for k, v in independent.state_dict().items()))
        with torch.no_grad():
            p = independent(samples); q = aligned_model(samples)
        for view in ['global', 'local']:
            close(f'zero-fusion {view} logits equal independent', p[view]['pred_logits'], q[view]['pred_logits'], 0)
            close(f'zero-fusion {view} boxes equal independent', p[view]['pred_boxes'], q[view]['pred_boxes'], 0)
        feats = independent.backbone(torch.cat((samples['global'], samples['local'])))
        local_feats, global_feats = [f[1:] for f in feats], [f[:1] for f in feats]
        aligned_model.interaction.restore.weight.data.normal_(0, .01)
        changed = aligned_model.interact(local_feats, global_feats, samples['crop_norm'])
        check('interaction changes only backbone localP3', not torch.equal(changed[0], local_feats[0]) and
              torch.equal(changed[1], local_feats[1]) and torch.equal(changed[2], local_feats[2]))
        aligned_model.train(); criterion = build_criterion(); criterion.set_epoch(7); criterion.set_model(aligned_model)
        output = aligned_model(samples, targets)
        losses = paired_loss(criterion, output, targets)
        loss = sum(losses.values()); check('native paired losses finite', bool(torch.isfinite(loss)))
        loss.backward()
        check('native dual backward reaches interaction', aligned_model.interaction.restore.weight.grad is not None and
              bool(aligned_model.interaction.restore.weight.grad.abs().sum() > 0))
        check('native dual backward reaches shared backbone', any(p.grad is not None and bool(p.grad.abs().sum() > 0)
              for p in aligned_model.backbone.parameters()))
        check('matching queries remain300 per branch', output['global']['pred_logits'].shape[1] == 300 and output['local']['pred_logits'].shape[1] == 300)
        del output, loss, losses, criterion, aligned_model, independent, p, q, feats

        cfg0 = load_config(ROOT / 'configs/query_default.yml')
        cfg1 = load_config(ROOT / 'configs/query_coverage.yml')
        torch.manual_seed(123); default = build_model(cfg0).eval()
        torch.manual_seed(123); coverage = build_model(cfg1).eval()
        check('query coverage adds no state tensors or parameters', default.state_dict().keys() == coverage.state_dict().keys() and
              all(torch.equal(v, coverage.state_dict()[k]) for k, v in default.state_dict().items()))
        with torch.no_grad():
            result = coverage(query_images[:1]); default_result = default(query_images[:1])
        check('actual coverage decoder outputs300 finite boxes', result['pred_boxes'].shape == (1, 300, 4) and
              bool(torch.isfinite(result['pred_boxes']).all()))
        # Native default _select_topk and reserve0 produce the same gathered tensors.
        memory = torch.randn(1, 500, 256)
        a = default.decoder._select_topk(memory, logits, anchors, 300)
        from engine.deim.dfine_decoder import DFINETransformer
        b = DFINETransformer._select_topk(default.decoder, memory, logits, anchors, 300)
        check('reserve0 gather matches native default method exactly', all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a, b)))
        coverage.train(); criterion = build_criterion(); criterion.set_epoch(7); criterion.set_model(coverage)
        output = coverage(query_images[:1], query_targets[:1]); loss = sum(criterion(output, query_targets[:1]).values())
        check('native coverage training loss finite', bool(torch.isfinite(loss)))
        loss.backward()
        check('query selection preserves encoder gradient via gather', coverage.decoder.enc_score_head.weight.grad is not None and
              bool(coverage.decoder.enc_score_head.weight.grad.abs().sum() > 0))
        check('query selection preserves native bbox gradient', any(p.grad is not None and bool(p.grad.abs().sum() > 0)
              for p in coverage.decoder.enc_bbox_head.parameters()))

    # Verify the shared files again after all tests. Local tests do not alter them.
    check('all shared snapshot sources unchanged after tests', all(hashlib.sha256(Path(r['source']).read_bytes()).hexdigest() == r['sha256']
          for r in manifest['files']))
    receipt = {'status': 'passed_local_CPU_implementation_checks_only', 'checks': results,
               'check_count': len(results), 'seconds': time.time() - started,
               'torch': torch.__version__, 'CPU_threads': 4, 'native_model_input': [1, 3, 640, 640],
               'native_query_loader_batch': 8, 'real_VisDrone_inference': False,
               'GPU_capacity_measured': False, 'APs_improvement_demonstrated': False,
               'formal_training_started': False}
    with Path(args.output).open('x', encoding='utf8') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'status': receipt['status'], 'checks': len(results), 'seconds': receipt['seconds']}))


if __name__ == '__main__':
    main()
