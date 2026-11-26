import os
for name,value in {'ATEN_CPU_CAPABILITY':'avx2','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                   'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}.items():os.environ[name]=value
import datetime
import importlib.metadata as metadata
import json
import sys
import torch
import torchvision
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,
     'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'cuda':torch.version.cuda,
     'cudnn':torch.backends.cudnn.version(),'torch_build':torch.__config__.show(),
     'packages':{d.metadata['Name']:d.version for d in metadata.distributions()},'cuda_available':torch.cuda.is_available()}
print(json.dumps(out))
