from pathlib import Path
import json,sys
import numpy as np
import torch
from semantic_math import pool_numpy
from pool_tensor import pool_tensor

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
rng=np.random.default_rng(380001)
y,x=np.mgrid[:200,:200]
feature=np.stack([rng.normal(0,20,(200,200)),(x+y)%2*20.,(x+10*y)/100.],0).astype(np.float32)
boxes=np.array([[0,0,1,1],[.25,.25,.5,.5],[.49999,.49999,.50001,.50001],
                [0,.2,.003,.21],[.99,.98,1,1],[.178123,.378245,.19345,.39432]],dtype=np.float32)
with torch.inference_mode():observed=pool_tensor(torch.from_numpy(feature)[None],torch.from_numpy(boxes),2)
reference=[pool_numpy(feature,b) for b in boxes]
fine=np.stack([r[0] for r in reference]);contrast=np.stack([r[1] for r in reference])
np.testing.assert_allclose(observed[0].numpy(),fine,atol=5e-4,rtol=2e-5)
np.testing.assert_allclose(observed[1].numpy(),contrast,atol=5e-4,rtol=2e-5)
np.testing.assert_array_equal(observed[2].numpy(),np.array([r[2] for r in reference]))
result=dict(status='PASS_SCALE_MATCHED200x200_CPU_ONLY',feature_shape=list(feature.shape),boxes=len(boxes),
    fine_max_error=float(np.max(np.abs(observed[0].numpy()-fine))),
    contrast_max_error=float(np.max(np.abs(observed[1].numpy()-contrast))),
    atol=5e-4,rtol=2e-5,author_detector_calls=0,diagnostic_data_probe_fits=0)
with (D/'analytic_scaled_remote_pre_freeze.json').open('x',encoding='utf-8') as output:json.dump(result,output,indent=2)
print(json.dumps(result))
