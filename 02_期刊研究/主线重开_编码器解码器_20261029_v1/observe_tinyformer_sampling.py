"""A0b: observes native sampling without changing its computation or predictions."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from tinyformer_probe_loader import ROOT, load_official_model, state_digest
from tinyformer_stage_probe import Budget, inputs, image_tensor, save_json


def observed_forward(model, x, budget):
    capture, handles, originals = {}, [], []
    for i, layer in enumerate(model.decoder.decoder.layers):
        ca = layer.cross_attn
        native_core = ca.ms_deformable_attn_core
        originals.append((ca, native_core))
        def make_core(index, original):
            def wrapped(value, shapes, locations, weights, points):
                out = original(value, shapes, locations, weights, points)
                capture['locations_' + str(index)] = locations.detach().cpu().numpy()[0]
                capture['weights_' + str(index)] = weights.detach().cpu().numpy()[0]
                capture['shapes_' + str(index)] = np.asarray(shapes, dtype=np.int64)
                capture['points_' + str(index)] = np.asarray(points, dtype=np.int64)
                return out
            return wrapped
        def make_pre(index):
            def pre(module, args):
                capture['reference_boxes_' + str(index)] = args[1].detach().cpu().numpy()[0, :, 0]
            return pre
        ca.ms_deformable_attn_core = make_core(i, native_core)
        handles.append(ca.register_forward_pre_hook(make_pre(i)))
    try:
        out = budget.forward(model, x, 'sampling_observation32')
    finally:
        for ca, original in originals:
            ca.ms_deformable_attn_core = original
        for handle in handles:
            handle.remove()
    capture['final_boxes'] = out['pred_boxes'][0].detach().cpu().numpy()
    capture['final_logits'] = out['pred_logits'][0].detach().cpu().numpy()
    return out, capture


def main():
    child = ROOT / 'sampling_observation_v1'
    child.mkdir(exist_ok=True)
    frozen = json.loads((ROOT / 'sampling_freeze_manifest.json').read_text())
    for row in frozen['files']:
        assert hashlib.sha256((ROOT / row['remote_name']).read_bytes()).hexdigest() == row['sha256']
    with (child / 'started.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'started_unix': time.time()}, f)
    assert json.loads((ROOT / 'receipts/study_complete.json').read_text())['full_model_forwards_cumulative'] == 129
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    rows = inputs()[:32]
    model, _, _ = load_official_model()
    model = model.cuda()
    digest = state_digest(model)
    budget = Budget()
    assert budget.data['attempted_full_model_forwards'] == 129
    checks = []
    started = time.monotonic()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            x = image_tensor(row['path'])
            native = budget.forward(model, x, 'sampling_equivalence_native') if i < 8 else None
            out, capture = observed_forward(model, x, budget)
            with np.load(ROOT / 'traces' / (row['path'].stem + '.npz'), allow_pickle=False) as old:
                old_box_diff = float(np.max(np.abs(capture['final_boxes'] - old['decoder_3_boxes'])))
                assert old_box_diff <= 1e-6
                old_scores = torch.from_numpy(old['decoder_3_logits']).sigmoid()
                current_scores = out['pred_logits'][0].sigmoid().cpu()
                assert (current_scores - old_scores).abs().max().item() <= 1e-6
                ni = torch.topk(old_scores.flatten(), 300).indices
                ti = torch.topk(current_scores.flatten(), 300).indices
                assert torch.equal(ni, ti)
                capture['topk_boxes'] = old['encoder_topk300_boxes']
                for k in old.files:
                    if k.startswith('gt_'):
                        capture[k] = old[k]
            check = {'file_name': row['path'].name, 'A0_final_boxes_maxabs': old_box_diff,
                     'A0_flat_top300_labels_and_queries_exact': True}
            if native is not None:
                bd = (native['pred_boxes'] - out['pred_boxes']).abs().max().item()
                sd = (native['pred_logits'].sigmoid() - out['pred_logits'].sigmoid()).abs().max().item()
                ids1 = torch.topk(native['pred_logits'].sigmoid().flatten(1), 300, dim=-1).indices
                ids2 = torch.topk(out['pred_logits'].sigmoid().flatten(1), 300, dim=-1).indices
                assert bd <= 1e-6 and sd <= 1e-6 and torch.equal(ids1, ids2)
                check.update(native_boxes_maxabs=bd, native_scores_maxabs=sd, native_flat_query_ids_exact=True)
            checks.append(check)
            np.savez_compressed(child / (row['path'].stem + '.npz'), **capture)
        assert state_digest(model) == digest
        result = {'status': 'SAMPLING32_OBSERVATION_AND_EQUIVALENCE_PASS', 'images': 32,
                  'checks': checks, 'training_updates': 0, 'new_model_forwards': 40,
                  'full_model_forwards_cumulative': budget.data['attempted_full_model_forwards'],
                  'model_state_unchanged': True, 'elapsed_seconds': time.monotonic() - started}
        assert result['full_model_forwards_cumulative'] == 169
        save_json(child / 'completed.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
