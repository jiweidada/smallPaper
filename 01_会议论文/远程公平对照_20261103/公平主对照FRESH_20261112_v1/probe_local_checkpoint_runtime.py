from pathlib import Path
import json
import os
import sys

root=Path(__file__).resolve().parents[2]
site=root/'本地CPU核验_20261103/.runtime/site-packages'
assert site.is_dir()
handles=[]
for path in [Path(sys.base_prefix),Path(sys.base_prefix)/'DLLs',site/'torch/lib']:
    if path.is_dir():handles.append(os.add_dll_directory(str(path)))
sys.path.insert(0,str(site))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
print(json.dumps({'status':'CPU_CHECKPOINT_DEPENDENCY_READY','torch':str(torch.__version__),'module':torch.__file__,'python':sys.version,'model_constructed':False,'forwards':0,'evaluator_runs':0}))
