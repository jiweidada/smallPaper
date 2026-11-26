from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
prior = ROOT.parent / '前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1'
manifest_path = prior / 'dataset_source_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
rows = {r['file_name']: r for r in manifest['files']}
ranked = sorted(rows, key=lambda n: (hashlib.sha256(('compress-dev-v1:' + n).encode()).hexdigest(), n))
selected = ranked[:100]
result = {'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'scope': 'Unchanged prior diagnostic100 from official train; author checkpoint trained on this train set',
          'ordering': 'Use existing remote diagnostic_images.txt order; require identical membership',
          'classes': ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'],
          'files': [rows[n] for n in selected], 'expected_positive_gt': 4491}
(ROOT / 'diagnostic100_asset_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
details = {'name': 'A0_implementation_details_v1', 'parent_protocol': '06_TinyFormer分阶段最小诊断协议.json',
           'freeze_before_any_model_forward': True,
           'checkpoint_state': 'ema.module, same choice as official tools/inference/torch_inf.py:122',
           'execution': 'Unfused original eval; FP32; TF32 disabled; batch1; no deployment conversion',
           'compatibility': ['Source files unchanged and SHA checked', 'Model-only imports skip training/data package initializers',
                             'Defer type annotations for Python3.9', 'Omit only AST-proven unused DropPath import; no replacement numerical layer'],
           'matching_clarification': 'For each IoU threshold, maximize eligible one-to-one match cardinality, then summed IoU; report any-box coverage separately',
           'crowding_clarification': 'Small GT sqrt(input area)<32 shares a 16-pixel origin-zero center cell with at least one other small GT',
           'preflight_budget': {'capacity': 1, 'native_vs_trace_first8': 16, 'benchmark_warmup': 4, 'benchmark_timed': 8},
           'study_budget': 100, 'planned_total_full_model_forwards': 129,
           'cumulative_hard_cap_from_parent': 200, 'training_updates': 0,
           'no_native_coverage_interpretation_before_equivalence_pass': True,
           'native_final': 'Official postprocessor flat class/query top300, which may repeat query IDs',
           'diagnostic_readouts': 'All encoder candidates and existing intermediate heads are off-path readouts; never an AP upper bound',
           'stage_names': ['encoder_all_valid', 'encoder_topk300', 'decoder_pre', 'decoder_0', 'decoder_1', 'decoder_2', 'decoder_3', 'native_post300'],
           'analysis_precision': 'Float64 CPU IoU from saved Float32 predictions; no score threshold for coverage',
           'no_study_restart_on_failure': True}
(ROOT / '08_A0实施细则_执行前冻结.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'diagnostic_images': len(selected), 'positive_gt_expected': 4491, 'planned_model_forwards': 129}))
