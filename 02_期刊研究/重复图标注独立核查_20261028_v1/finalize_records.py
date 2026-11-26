"""Verify source lineage artifacts and quote existing peer results accurately."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, data):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


d = load(HERE / 'source_chain_results.json')
assert d['status'] == 'PASS' and len(d['items']) == 6 and len(d['groups']) == 3
assert not d['changed_inputs']
assert all(x['C_equals_D_compressed_bytes'] and x['C_equals_D_RGB'] and x['all_dimensions_equal']
           and x['COCO_rounding_within_0_01_px'] and x['COCO_equals_own_D_raw_all_positive_keys']
           and x['COCO_equals_own_D_raw_score1_keys'] and x['C_raw_TXT']['bytes_equal_D_raw']
           for x in d['items'])
assert all(x['D_byte_equal'] and x['D_RGB_equal'] and not x['D_raw_eight_field_multisets_equal']
           for x in d['groups'])
assert all(x['old_20261006_val_snapshot']['bytes_equal_current_D'] for x in d['items'] if x['split'] == 'val')

source_ledger = []
for name, digest in d['input_hashes_before_and_after_equal'].items():
    path = Path(name)
    assert sha(path) == digest, name
    source_ledger.append({'path': name, 'sha256': digest,
                          'read_scope': 'Six selected same-input source pairs only; see frozen00 and per-item raw/COCO identity result.'})
for x in d['items']:
    snap = HERE / 'raw_annotation_snapshots' / (x['split'] + '_' + Path(x['raw_TXT_path']).name)
    assert sha(snap) == x['raw_TXT_sha256']

peer_file = PROJECT / '新问题入口重寻_20261028_v1/identical_pixels_labels_attempt01/results.json'
peer = load(peer_file)
assert peer['status'] == 'COMPLETED' and peer['inputs_unchanged'] is True
assert peer['CPU_evaluator_calls'] == 4
assert peer['new_models_checkpoints_heads_forwards_backwards_training'] == 0
readcheck = {'source': str(peer_file), 'sha256': sha(peer_file),
             'scope': 'Read existing peer evaluation; only recompute arithmetic differences, no evaluator call.',
             'seeds': {}}
for seed in ['1', '2']:
    pred = peer['val_prediction_comparison'][seed]
    assert pred['max_final_box_difference'] == pred['max_final_logits_difference'] == pred['max_export_score_difference'] == 0
    sens = peer['annotation_version_sensitivity'][seed]
    base = sens['original_metrics_pp']['AP']
    record = {'native_cap_common_output_pairs': pred['common_native_cap_identities'],
              'max_prediction_differences': [pred[k] for k in ['max_final_box_difference', 'max_final_logits_difference', 'max_export_score_difference']],
              'IoU_counts': {t: {'denominator_common_cap_pairs': v['denominator_common_cap_pairs'],
                                 'A_TP': v['A_TP'], 'A_GT': v['A_GT'], 'B_TP': v['B_TP'], 'B_GT': v['B_GT'],
                                 'pair_status_counts': v['counts']}
                             for t, v in pred['by_IoU'].items()},
              'original_AP_percentage_points': base, 'scenarios': {}}
    for arm, value in sens['scenarios'].items():
        ap = value['metrics_pp']['AP']
        delta = ap - base
        assert abs(delta - value['delta_pp']['AP']) < 1e-10
        record['scenarios'][arm] = {'GT': value['fullGT'], 'AP_percentage_points': ap,
                                    'recomputed_AP_delta_percentage_points': delta}
    readcheck['seeds'][seed] = record
save('peer_result_readcheck.json', readcheck)
source_ledger.append({'path': str(peer_file), 'sha256': readcheck['sha256'],
                      'read_scope': 'Actual prediction equality and annotation-version sensitivity fields; peer executed4 evaluations, this partner0.'})

old = PROJECT / 'raw_EMA逐GT对照_20261026_v1/跨状态逐GT响应4x4_20261026_v1/protocol.json'
assert sha(old) == 'a1ea55445363e9db94d4902b4afd434c07d82c568f8099f86150ded2e766a275'
save('source_ledger.json', {'scope': 'Current local sources and old val snapshots; no independent official-package proof.',
                          'sources': source_ledger, 'no_source_files_edited': True})
save('verification.json', {'status': 'PASS', 'date_utc': datetime.now(timezone.utc).isoformat(),
                          'source_items': 6, 'source_groups': 3,
                          'raw_snapshots_byte_equal': True,
                          'original_4x4_protocol_unchanged': True,
                          'partner_operations': {'new_model_loads': 0, 'forwards': 0, 'training': 0,
                                                 'new_evaluation': 0, 'gradients': 0, 'IoU_assignment': 0,
                                                 'NumPy_BLAS_calls': 0, 'subagents': 0,
                                                 'PIL_image_decodes': 12, 'original_images_visually_reviewed': 3},
                          'peer_results_read_and_AP_deltas_recomputed': True,
                          'scientific_limit': 'Local input/annotation-version inconsistency established; no truth adjudication, official package identity or method benefit established.'})
outputs = []
for path in sorted(HERE.rglob('*')):
    if path.is_file() and path.name != 'delivery_manifest.json':
        outputs.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
save('delivery_manifest.json', {'status': 'PASS', 'artifacts': outputs, 'artifact_count': len(outputs)})
print(json.dumps({'status': 'PASS', 'sources': len(source_ledger), 'artifacts': len(outputs),
                  'source_groups': 3, 'new_forwards': 0, 'new_evaluation': 0, 'training': 0}, ensure_ascii=False))
