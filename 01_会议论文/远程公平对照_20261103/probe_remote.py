from pathlib import Path
import importlib.metadata as m
import json, os, sys, subprocess, shutil
out = {'cwd': os.getcwd(), 'python': sys.executable, 'version': sys.version}
for p in ['/25085400120/DEIM', '/25085400120/datasets_verified_20261017', '/25085400120/envs', '/opt/conda/envs']:
    out[p] = sorted(os.listdir(p))[:60] if os.path.isdir(p) else None
for p in ['/25085400120/DEIM/dataset_visdrone', '/25085400120/DEIM/dataset', '/25085400120/DEIM/datasets']:
    if os.path.isdir(p): out[p] = sorted(os.listdir(p))[:30]
for name in ['torch','torchvision','faster-coco-eval','numpy','PyYAML','scipy','tensorboard','calflops','psutil','timm','transformers','pycocotools']:
    try: out[name] = m.version(name)
    except m.PackageNotFoundError: out[name] = None
out['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
out['gpu_processes'] = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
import torch
out['torch_runtime'] = {'version': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available(), 'cudnn': torch.backends.cudnn.version()}
out['free_bytes'] = shutil.disk_usage('/25085400120').free
out['isolated_directory_exists'] = Path('/25085400120/mpsma_fair_20261103_v1').exists()
print(json.dumps(out, indent=2))
