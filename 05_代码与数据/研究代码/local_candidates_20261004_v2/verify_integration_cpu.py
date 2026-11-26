"""Close remaining prepared-runner/evaluator/cache/trace interfaces on synthetic CPU data."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import bootstrap
from bootstrap import ROOT
from factory import build_model, build_criterion, load_config, load_common_init, build_postprocessor
from dataset import PairedCocoImages, paired_collate
from dual_view import paired_loss
from evaluate import evaluate_dataset, infer_original
from run_candidate import optimizer_for, one_step, implementation_hash
from diagnose_queries import coverage_summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); torch.set_num_threads(4); torch.manual_seed(73)
    checks = []
    def check(name, ok):
        if not ok: raise AssertionError(name)
        checks.append({'name': name, 'passed': True})
    with tempfile.TemporaryDirectory(prefix='deim_interface_cpu_') as tmp:
        tmp = Path(tmp)
        Image.fromarray(np.full((700, 900, 3), 100, np.uint8)).save(tmp / '0.png')
        annotation = {'images': [{'id': 0, 'file_name': '0.png', 'width': 900, 'height': 700}],
            'categories': [{'id': i, 'name': str(i)} for i in range(10)],
            'annotations': [{'id': 0, 'image_id': 0, 'category_id': 1,
                             'bbox': [150, 200, 12, 16], 'area': 192, 'iscrowd': 0}], 'info': {}}
        (tmp / 'ann.json').write_text(json.dumps(annotation), encoding='utf8')
        dataset = PairedCocoImages(tmp, tmp / 'ann.json', augment=False)
        sample, target = paired_collate([dataset[0]])
        cfg = load_config(ROOT / 'configs/dual_aligned.yml')
        model = build_model(cfg).eval()
        with torch.no_grad():
            paired = model(sample)
            cached, global_output = model.cache_global(sample['global'])
            local_output = model.predict_local(sample['local'], sample['crop_norm'], cached)
        check('cached global matches paired eval', torch.allclose(global_output['pred_logits'], paired['global']['pred_logits'], atol=2e-5, rtol=0))
        check('cached local matches paired eval', torch.allclose(local_output['pred_boxes'], paired['local']['pred_boxes'], atol=2e-5, rtol=0))
        # A real optimizer update via the actual prepared runner, without CUDA.
        criterion = build_criterion(); criterion.set_epoch(7); criterion.set_model(model); model.train()
        optimizer = optimizer_for(model); scaler = torch.amp.GradScaler('cpu', enabled=False)
        before = model.interaction.restore.weight.detach().clone()
        step = one_step(model, criterion, optimizer, scaler, sample, target, cfg, torch.device('cpu'))
        check('prepared native paired optimizer executes once onCPU', step['optimizer_step_calls'] == 1 and step['amp_skips_proxy'] == 0)
        check('first training update learns nonzero fusion', not torch.equal(before, model.interaction.restore.weight))
        result = evaluate_dataset(model, cfg, tmp, tmp / 'ann.json', tmp / 'dual_val', torch.device('cpu'))
        check('complete original-coordinate COCO path outputs all12', len(result['metrics']) == 12 and result['images'] == 1)
        import gzip
        with gzip.open(tmp / 'dual_val/raw_view_predictions.json.gz','rt',encoding='utf8') as h:
            check('raw whole and all4 crops preserved', len(json.load(h)) == 5)
        check('merged image output cap respected', len(json.loads((tmp / 'dual_val/predictions.json').read_text())) <= 300)
        check('small annotation id0 unchanged', json.loads((tmp / 'ann.json').read_text())['annotations'][0]['id'] == 0)

        query_cfg = load_config(ROOT / 'configs/query_default.yml')
        default = build_model(query_cfg).eval()
        torch.save({'model': default.state_dict()}, tmp / 'synthetic_common.pth')
        receipt = load_common_init(model, tmp / 'synthetic_common.pth')
        check('common baseline init loads exactly with only fusion extras', bool(receipt['new_tensors']) and all(k.startswith('interaction.') for k in receipt['new_tensors']))
        cover_cfg = load_config(ROOT / 'configs/query_coverage.yml')
        cover = build_model(cover_cfg).eval()
        receipt = load_common_init(cover, tmp / 'synthetic_common.pth')
        check('query coverage strict common init has no new tensors', receipt['new_tensors'] == [])
        with torch.no_grad():
            output_a = cover(sample['global'])
            cover.decoder.trace_enabled = True
            output_b = cover(sample['global'])
        check('diagnostic tracing does not change selection/output', torch.equal(output_a['pred_boxes'], output_b['pred_boxes']) and
              torch.equal(output_a['pred_logits'], output_b['pred_logits']))
        trace = {k: v.numpy() for k, v in cover.decoder.last_query_trace.items()}
        summary = coverage_summary(trace, annotation['annotations'], (900, 700))
        check('trace covers all8400 tokens and exactly300 unique selections', trace['boxes'].shape == (1, 8400, 4) and
              len(np.unique(trace['selected_indices'])) == 300)
        check('offline GT diagnostic returns bounded counts', summary['all'] == summary['small'] == 1 and
              0 <= summary['small_selected_covered'] <= summary['small_eligible_covered'] <= 1)
        # Explicitly exercise the empty crop that omitted native DN loss keys.
        empty_target = copy.deepcopy(target)
        for key in ['boxes', 'labels', 'area', 'annotation_ids', 'iscrowd']:
            empty_target['local'][0][key] = empty_target['local'][0][key][:0]
        model.train(); criterion = build_criterion(); criterion.set_epoch(7); criterion.set_model(model)
        empty_step = one_step(model, criterion, optimizer_for(model), torch.amp.GradScaler('cpu', enabled=False),
                              sample, empty_target, cfg, torch.device('cpu'))
        check('empty local crop DN terms handled without changing native loss', empty_step['optimizer_step_calls'] == 1 and np.isfinite(empty_step['loss']))
    result = {'status': 'passed_local_CPU_integration_only', 'checks': checks, 'count': len(checks),
              'implementation_sha256': implementation_hash(), 'GPU_capacity_measured': False,
              'synthetic_data_only': True, 'APs_improvement_demonstrated': False}
    with Path(args.output).open('x', encoding='utf8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'checks': len(checks)}))


if __name__ == '__main__':
    main()
