from pathlib import Path
import argparse,json
import numpy as np
from npz_export import save_npz

p=argparse.ArgumentParser();p.add_argument('--out-prefix',required=True);a=p.parse_args()
base=Path(a.out_prefix)
arrays={'file':np.array('example_image.jpg'),'features':np.array([[1.25,-2.5],[3.75,4.]],dtype=np.float32),
        'index':np.array(0,dtype=np.int64),'split':np.array('train')}
save_npz(str(base)+'.npz',arrays)
with np.load(str(base)+'.npz',allow_pickle=False) as observed:
    assert set(observed.files)==set(arrays)
    for name,value in arrays.items():
        np.testing.assert_array_equal(observed[name],value)
        assert observed[name].dtype==value.dtype and observed[name].shape==value.shape
result=dict(status='PASS_STANDARD_NPZ_RESERVED_FILE_KEY_ROUNDTRIP',arrays=len(arrays),
            numerical_arrays_modified=False,author_model_calls=0,data_probe_fits=0)
with Path(str(base)+'.json').open('x',encoding='utf-8') as output:json.dump(result,output,indent=2)
print(json.dumps(result))
