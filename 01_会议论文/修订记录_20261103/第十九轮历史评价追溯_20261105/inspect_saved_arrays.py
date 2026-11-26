"""Inspect stored array formats using an inert restricted pickle reader; no evaluator runs."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import io
import json
import pickle
import struct
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
SOURCE = Path(r'C:\Users\jiwei\Desktop\DEIM\outputs')
OUTPUT = HERE/'saved_array_format_inventory.json'

class InertDtype:
    def __init__(self, name, *args):
        self.name = name
        self.byteorder = '<'
    def __setstate__(self, state):
        self.byteorder = state[1]

class InertArray:
    def __setstate__(self, state):
        self.version, self.shape, self.dtype, self.fortran, self.raw = state
    def to_numpy(self):
        if not isinstance(self.dtype, InertDtype) or not isinstance(self.raw, bytes):
            raise ValueError('unsupported inert numpy representation')
        dtype = np.dtype(self.dtype.name).newbyteorder(self.dtype.byteorder)
        if dtype.hasobject:
            raise ValueError('object arrays forbidden')
        return np.frombuffer(self.raw, dtype=dtype).reshape(self.shape, order='F' if self.fortran else 'C')

class InertParams:
    pass

def reconstruct(*args):
    return InertArray()

def encode(s, encoding):
    if encoding != 'latin1':
        raise ValueError('unsupported encoding')
    return s.encode(encoding)

def scalar(dtype, raw):
    types = {'i8':'q','i4':'i','f8':'d','f4':'f','b1':'?','u8':'Q','u4':'I'}
    return struct.unpack(('>' if dtype.byteorder=='>' else '<') + types[dtype.name],raw)[0]

class SafeUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='faster_coco_eval.core.cocoeval' and name=='Params': return InertParams
        if module=='numpy' and name=='dtype': return InertDtype
        if module=='numpy' and name=='ndarray': return InertArray
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct': return reconstruct
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='scalar': return scalar
        if module=='_codecs' and name=='encode': return encode
        raise ValueError('unsupported pickle global: '+module+'.'+name)

def read_saved(path):
    with ZipFile(path) as z:
        candidates = [n for n in z.namelist() if n.endswith('/data.pkl')]
        assert len(candidates)==1 and z.getinfo(candidates[0]).file_size<20*1024*1024
        return SafeUnpickler(io.BytesIO(z.read(candidates[0]))).load()

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists()
    samples = [
        ('N0_latest',SOURCE/'deim_hgnetv2_n_custom320/eval/latest.pth'),
        ('N4_300',SOURCE/'deim_hgnetv2_MSCA_EnhancedStem_SiLU_320/eval/300.pth'),
        ('N4_latest',SOURCE/'deim_hgnetv2_MSCA_EnhancedStem_SiLU_320/eval/latest.pth'),
        ('S1_latest',SOURCE/'deim_hgnetv2_s_MSCA_EnhancedStem_SiLU_320/eval/latest.pth'),
        ('S1_standalone',SOURCE/'deim_hgnetv2_s_MSCA_EnhancedStem_SiLU_320_eval/eval.pth')]
    items = []
    for label,path in samples:
        try:
            data = read_saved(path)
            arrays = {}
            for name in ['precision','recall','scores']:
                a = data.get(name)
                if isinstance(a,InertArray):
                    array = a.to_numpy()
                    arrays[name] = {'shape':list(array.shape), 'dtype':str(array.dtype),
                                    'finite':bool(np.isfinite(array).all()),
                                    'minimum':float(array.min()), 'maximum':float(array.max())}
            items.append({'label':label,'path':str(path),'sha256':digest(path),'keys':list(data),
                          'date':data.get('date'),'arrays':arrays})
        except Exception as exc:
            items.append({'label':label,'path':str(path),'sha256':digest(path),'error':type(exc).__name__+': '+str(exc)})
    cpu = []
    for run in ['N0','N4','S1']:
        path = PAPER/f'本地CPU核验_20261103/results/{run}/evaluation_arrays.npz'
        with np.load(path, allow_pickle=False) as f:
            cpu.append({'run':run,'path':str(path),'sha256':digest(path),
                        'arrays':{key:{'shape':list(f[key].shape),'dtype':str(f[key].dtype)} for key in f.files}})
    report = {'status':'SAVED_ARRAY_FORMATS_INSPECTED','saved':items,'cpu':cpu,
              'scope':'Read retained arrays only. No model, prediction generation, evaluator, or optimizer was run.'}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n')
    for row in items:
        print(json.dumps(row,ensure_ascii=False))
    for row in cpu:
        print(json.dumps(row,ensure_ascii=False))

if __name__=='__main__':
    main()
