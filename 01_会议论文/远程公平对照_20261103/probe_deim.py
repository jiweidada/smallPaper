from pathlib import Path
import os, json, sys, importlib.metadata as m
out = {'python': sys.executable}
for p in ['/25085400120/envs/deim', '/25085400120/envs/pybase', '/25085400120/DEIM/work', '/25085400120/DEIM/experiments']:
    out[p] = sorted(os.listdir(p))[:65] if os.path.isdir(p) else None
found = []
base = Path('/25085400120/DEIM')
for root, dirs, files in os.walk(base):
    depth = len(Path(root).relative_to(base).parts)
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['outputs','experiments','site-packages','__pycache__','node_modules']]
    if depth >= 5: dirs[:] = []
    if 'train.json' in files or 'val.json' in files or ('images' in dirs and 'annotations' in dirs):
        found.append({'path':root,'annotations':[x for x in ['train.json','val.json'] if x in files], 'dirs':dirs[:12]})
out['dataset_candidates'] = found
for name in ['torch','torchvision','faster-coco-eval','numpy','PyYAML','scipy','tensorboard','tidecv','calflops','psutil','timm','transformers','matplotlib','pycocotools']:
    try: out[name] = m.version(name)
    except m.PackageNotFoundError: out[name] = None
print(json.dumps(out, indent=2))
