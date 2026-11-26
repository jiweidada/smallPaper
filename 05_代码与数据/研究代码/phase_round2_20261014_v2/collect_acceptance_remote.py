"""Read-only progress and small acceptance artifacts. No checkpoint transfer."""
import importlib.util
import json
import pathlib
import subprocess

storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
stage = phase/'artifacts/round2_20261014_v1'
result = {'phase_root': str(phase), 'files': {}, 'dependencies': {name: importlib.util.find_spec(name) is not None for name in ['torch', 'pycocotools', 'thop', 'calflops']}}
for name in ['staging_receipt.json', 'activation_receipt.json', 'test_p2_phase_controls.py.log', 'test_p2_round2_acceptance.py.log', 'gpu_preflight.log']:
    p = stage/name
    if p.is_file():
        result['files'][name] = p.read_text(errors='replace')[-24000:]
result['jobs'] = {p.name: json.loads(p.read_text()) for p in (phase/'jobs').glob('*.json')}
result['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True, capture_output=True).stdout
dataset = phase/'code/dataset_visdrone'
result['testdev_candidates'] = [str(p) for p in dataset.glob('VisDrone2019-DET-test-dev/annotations/*.json')]
result['testdev_images'] = str(dataset/'VisDrone2019-DET-test-dev/images')
print(json.dumps(result, indent=2))
