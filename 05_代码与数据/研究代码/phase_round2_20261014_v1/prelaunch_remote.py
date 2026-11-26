"""One short end-to-end four-image check before any new expensive training."""
import json
import os
import pathlib
import subprocess
import sys
import time
import yaml

storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
addons = phase/'artifacts/round2_20261014_v1/addons'
root = addons/'prelaunch_four_images'
root.mkdir(exist_ok=False)
plan = json.loads((addons/'queue_plan.json').read_text())
env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
def run(args, name):
    with (root/(name+'.log')).open('xb') as output:
        p = subprocess.run([sys.executable, '-u']+args, cwd=phase/'code', env=env,
                           stdout=output, stderr=subprocess.STDOUT, timeout=600)
    if p.returncode:
        raise RuntimeError(name+' failed; inspect '+str(root/(name+'.log')))

for row in plan['existing_models']:
    assert pathlib.Path(row['checkpoint']).is_file(), 'Missing planned source checkpoint: '+row['checkpoint']
run([str(addons/'benchmark_soep_lite.py')], 'cost_benchmark')
annotations = json.loads(pathlib.Path(plan['val_annotations']).read_text())
ids = sorted(image['id'] for image in annotations['images'])[:4]
annotations['images'] = [image for image in annotations['images'] if image['id'] in ids]
annotations['annotations'] = [gt for gt in annotations['annotations'] if gt['image_id'] in ids]
mini_ann = root/'four_images.json'
mini_ann.write_text(json.dumps(annotations))
mini_cfg = root/'four_images.yml'
mini_cfg.write_text(yaml.safe_dump({'__include__': [str(phase/'code/configs/experiments/phase/c_full96.yml')],
                                   'val_dataloader': {'dataset': {'ann_file': str(mini_ann)}, 'num_workers': 2, 'total_batch_size': 4},
                                   'score_threshold': 0.0}, sort_keys=False))
checkpoint = next(row['checkpoint'] for row in plan['existing_models'] if row['name'] == 'existing_c')
run(['train.py', '-c', str(mini_cfg), '-r', checkpoint, '--test-only', '--seed', '0',
     '--output-dir', str(root/'test_only')], 'test_only')
run([str(addons/'dump_preds.py'), '--config', str(mini_cfg), '--checkpoint', checkpoint,
     '--out', str(root/'dump'), '--project-eval'], 'dump')
run([str(addons/'analyze_preds.py'), '--annotations', str(mini_ann),
     '--reference', str(root/'dump/predictions.json'), '--candidate', str(root/'dump/predictions.json'),
     '--out', str(root/'analysis')], 'analysis')
first = json.loads((root/'test_only/pred_bbox.json').read_text())
second = json.loads((root/'dump/predictions.json').read_text())
assert len(first) == len(second) == 1200
def canonical(rows):
    return sorted([(row['image_id'], row['category_id'], row['score'], *row['bbox']) for row in rows])
maximum_difference = 0.
for left, right in zip(canonical(first), canonical(second)):
    assert left[:2] == right[:2]
    maximum_difference = max(maximum_difference, max(abs(x-y) for x, y in zip(left[2:], right[2:])))
assert maximum_difference < .001, 'CLI and dump outputs disagree'
report = {'status': 'passed', 'scope': 'four-image implementation smoke only; not a paper result',
          'image_ids': ids, 'prediction_count': 1200, 'CLI_vs_dump_max_abs_difference': maximum_difference,
          'project_evaluator_and_CPU_analysis_completed': True, 'finished': time.time()}
with (root/'receipt.json').open('x') as output:
    json.dump(report, output, indent=2)
print(json.dumps(report, indent=2))
