"""CPU-only comparison to extracted author functions on synthetic random tapes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from a10_dn_geometry import tape_for, numpy_generator, initial_refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root)
    namespace = {'torch': torch}
    sources = []
    for rel, names in [
        ('src/zoo/dome/box_ops.py', ['box_cxcywh_to_xyxy', 'box_xyxy_to_cxcywh']),
        ('src/zoo/dome/utils.py', ['inverse_sigmoid']),
        ('src/zoo/dome/denoising.py', ['get_contrastive_denoising_training_group']),
    ]:
        path = source_root / rel
        text = path.read_text(encoding='utf-8')
        body = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(body) == len(names)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
        sources.append({'relative_path': rel, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    ordinary = np.asarray([[.25,.25,.08,.04],[.3,.25,.08,.04],[.8,.8,.02,.02]], np.float32)
    border = np.asarray([[0,0,.01,.02],[1,1,.1,.05],[.00001,.5,.00001,.3],[.5,.5,.000001,.000001]], np.float32)
    large = np.tile(np.asarray([[.4,.6,.02,.04]], np.float32), (121,1))
    inputs = [[np.zeros((0,4),np.float32)], [ordinary[:1]], [ordinary], [border], [ordinary, ordinary[:1]], [large], [np.zeros((0,4),np.float32), ordinary]]
    records = []
    for ci, boxes in enumerate(inputs):
        labels = [(np.arange(len(b)) % 10 + 1).astype(np.int32) for b in boxes]
        tape = tape_for(boxes, 2026103010 + ci)
        expected = numpy_generator(boxes, labels, tape)
        def rand_like(tensor, **kwargs):
            key = 'label_u' if tensor.ndim == 2 else 'box_u'
            return torch.as_tensor(tape[key], dtype=kwargs.get('dtype', tensor.dtype), device=tensor.device)
        def randint_like(tensor, low, high, **kwargs):
            key = 'new_labels' if tensor.ndim == 2 else 'sign'
            return torch.as_tensor(tape[key], dtype=kwargs.get('dtype', tensor.dtype), device=tensor.device)
        targets = [{'boxes': torch.tensor(b), 'labels': torch.tensor(l, dtype=torch.int64)} for b,l in zip(boxes,labels)]
        with mock.patch.object(torch, 'rand_like', rand_like), mock.patch.object(torch, 'randint_like', randint_like):
            result = namespace['get_contrastive_denoising_training_group'](targets, 12, 5, lambda ids: ids[..., None].clone(), num_denoising=100, label_noise_ratio=.5, box_noise_scale=1., batch_queries_num=[5]*len(boxes), num_heads=8)
        got_ids = result[0][...,0].numpy()
        got_logits = result[1].numpy()
        assert np.array_equal(got_ids, expected[0])
        logit_error = float(np.max(np.abs(got_logits - expected[1]))) if got_logits.size else 0.
        ref_error = float(np.max(np.abs(torch.sigmoid(result[1]).numpy() - initial_refs(expected[1])))) if got_logits.size else 0.
        assert logit_error <= 1e-5 and ref_error <= 2e-7
        assert result[3]['dn_num_group'] == expected[3]
        if expected[3]:
            assert all(np.array_equal(t.numpy(), p) for t,p in zip(result[3]['dn_positive_idx'], expected[2]))
        records.append({'case': ci, 'batch_GT_counts': [len(b) for b in boxes], 'logit_max_abs_error': logit_error, 'initial_ref_max_abs_error': ref_error, 'class_ids_and_positive_indices_equal': True})
    assert not torch.cuda.is_initialized()
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_PARAMETER_FREE_AUTHOR_OPERATOR_EQUIVALENCE', 'cases': records, 'source_files': sources, 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'CPU', 'learned_parameters': 0, 'full_model_forwards': 0, 'fits': 0, 'scope': 'Author random tensor calls receive shared deterministic tapes; identity class embedding. Class IDs,positive indices,group counts,reference logits and initial sigmoid references checked. Not complete detector/training-mask equivalence.'}
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({'status': receipt['status'], 'cases': len(records), 'max_logit_error': max(r['logit_max_abs_error'] for r in records), 'max_initial_ref_error': max(r['initial_ref_max_abs_error'] for r in records)}))


if __name__ == '__main__':
    main()
