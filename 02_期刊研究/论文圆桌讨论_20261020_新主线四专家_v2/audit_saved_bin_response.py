"""Read only development archives with a restricted NumPy tensor decoder.

No torch import, forward, backward, optimizer, or heldout data access.
This is exploratory local coordinate sensitivity, not training plasticity.
"""
import collections, hashlib, io, json, pickle, tarfile, zipfile
from pathlib import Path
import numpy as np
from audit_saved_support import xyxy, ceiling, iou

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/isolated_v3'
DTYPES={'FloatStorage':np.dtype('<f4'),'LongStorage':np.dtype('<i8')}

def tensor(storage,offset,size,stride,requires_grad,hooks):
    shape=tuple(size); strides=tuple(stride)
    if not all(n>=0 for n in shape) or not all(n>=0 for n in strides): raise ValueError('unsupported tensor')
    last=offset+sum((n-1)*s for n,s in zip(shape,strides)) if all(shape) else offset
    if offset<0 or last>=len(storage):
        if np.prod(shape)!=0: raise ValueError('storage bounds')
    return np.lib.stride_tricks.as_strided(storage[offset:],shape=shape,strides=tuple(s*storage.itemsize for s in strides),writeable=False)

class ArraysOnly(pickle.Unpickler):
    def __init__(self,data,z,prefix): super().__init__(io.BytesIO(data)); self.z=z;self.prefix=prefix;self.storages={}
    def find_class(self,module,name):
        if module=='torch._utils' and name=='_rebuild_tensor_v2': return tensor
        if module=='torch' and name in DTYPES: return DTYPES[name]
        if module=='collections' and name=='OrderedDict': return collections.OrderedDict
        raise pickle.UnpicklingError(f'disallowed global {module}.{name}')
    def persistent_load(self,pid):
        kind,dtype,key,location,n=pid
        if kind!='storage' or location!='cpu' or dtype not in DTYPES.values(): raise pickle.UnpicklingError('unexpected storage')
        if key not in self.storages:
            v=np.frombuffer(self.z.read(self.prefix+'data/'+key),dtype=dtype)
            if len(v)!=n: raise ValueError('storage size mismatch')
            self.storages[key]=v
        return self.storages[key]

def load_arrays(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        prefix=name[:-len('data.pkl')]
        if prefix+'byteorder' in z.namelist(): assert z.read(prefix+'byteorder')==b'little'
        return ArraysOnly(z.read(name),z,prefix).load()

def quant(v): return np.quantile(v,[.1,.25,.5,.75,.9]).tolist() if len(v) else []

def main():
    spec={'scope':'development only: two BASE24 seeds, 8 archived batches each; actual native union pair identities',
          'groups':['current IoU<.5 and support ceiling>=.5','current IoU<.5 and ceiling<.5','current IoU>=.5'],
          'metrics':'per edge pmax, entropy, norm of d(normalized offset)/d(logits), physical edge Jacobian norm',
          'limits':['local output sensitivity only, not actual loss gradients or training dynamics','augmented repeated pair exposures, not validation AP','ordinary final pairs only','exploratory descriptive analysis; no heldout access'],
          'new_forward':0,'new_backward':0,'optimizer_updates':0,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'15_开发组bin响应探索口径.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
    manifest_path=DATA.parent/'shared_inputs/inputs/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    input_rows={r['batch']:r for r in manifest['batches']}
    summaries=[]; verified={str(manifest_path.relative_to(ROOT)):hashlib.sha256(manifest_path.read_bytes()).hexdigest()}; reconstruction_max=0.
    for account,seed in [('A',1),('B',2)]:
        rows=[]
        with tarfile.open(DATA/f'{account}_frozen_predictions.tar.gz') as tf, tarfile.open(DATA/f'{account}_development.tar.gz') as ta:
            seal=json.load(tf.extractfile('development/seal.json'))
            for item in seal['files']:
                name=item['relative']; assert name.startswith('batch') and name.endswith('_predictions.pt')
                blob=tf.extractfile('development/'+name).read(); digest=hashlib.sha256(blob).hexdigest(); assert digest==item['sha256']
                analysis=json.load(ta.extractfile(name.replace('_predictions.pt','_analysis.json')))
                assert analysis['prediction_sha256']==digest
                input_row=input_rows[item['batch']]
                assert input_row['sha256']==item['input_sha256']
                assert input_row['group']=='development'
                height,width=input_row['shape'][-2:]
                saved=load_arrays(blob); o=saved['outputs']; targets=saved['targets']
                verified[f'{account}/development/{name}']=digest
                logits=o['pred_corners'].astype(np.float64); regmax=logits.shape[-1]//4-1
                up=float(o['up'].ravel()[0]); scale=abs(float(o['reg_scale'].ravel()[0])); assert regmax%2==0
                upper=abs(up)*scale; step=(upper+1)**(2/(regmax-2))
                bins=np.array([-2*upper]+[1-step**i for i in range(regmax//2-1,0,-1)]+[0.]+[step**i-1 for i in range(1,regmax//2)]+[2*upper])
                assert len(bins)==regmax+1
                z=logits.reshape(*logits.shape[:2],4,-1); p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
                mu=(p*bins).sum(-1); jac=np.sqrt(np.sum((p*(bins-mu[...,None]))**2,axis=-1))
                refs=o['ref_points'].astype(np.float64); cs=np.tile(refs[...,2:]/scale,(1,1,2))
                edges=np.concatenate((refs[...,:2]-(scale/2+mu[...,:2])*refs[...,2:]/scale,refs[...,:2]+(scale/2+mu[...,2:])*refs[...,2:]/scale),axis=-1)
                reconstruction_max=max(reconstruction_max,float(np.max(np.abs(edges-xyxy(o['pred_boxes'])))))
                assert np.max(np.abs(edges-xyxy(o['pred_boxes'])))<1e-5
                for bi,record in enumerate(analysis['inputs']):
                    branch=record['branches']['final']['native_union']; qs=branch['pair_query'];gs=branch['pair_GT']
                    for k,(q,g) in enumerate(zip(qs,gs)):
                        gt=targets[bi]['boxes'][g]; actual=iou(edges[bi,q],xyxy(gt)); assert abs(actual-branch['pair_iou'][k])<1e-4
                        cap=ceiling(refs[bi,q],xyxy(gt)); group='good' if actual>=.5 else ('bad_reachable' if cap>=.5 else 'bad_unreachable')
                        size=float(np.sqrt(gt[2]*width*gt[3]*height))
                        sizebin='<4' if size<4 else ('4-8' if size<8 else ('8-16' if size<16 else ('16-32' if size<32 else '>=32')))
                        for e in range(4): rows.append((group,sizebin,float(p[bi,q,e].max()),float(-np.sum(p[bi,q,e]*np.log(np.maximum(p[bi,q,e],1e-300)))),float(jac[bi,q,e]),float(jac[bi,q,e]*cs[bi,q,e]),actual,cap))
        for sizebin in ['ALL','<4','4-8','8-16','16-32','>=32']:
            for group in ['bad_reachable','bad_unreachable','good']:
                rs=[r for r in rows if r[0]==group and (sizebin=='ALL' or r[1]==sizebin)]
                summaries.append({'seed':seed,'size':sizebin,'group':group,'pair_exposures':len(rs)//4,
                                  **{key:quant([r[ix] for r in rs]) for key,ix in [('edge_pmax_q10_q25_q50_q75_q90',2),('entropy',3),('normalized_jacobian_norm',4),('physical_jacobian_norm',5)]}})
    result={'spec':spec,'verified_prediction_sha256':verified,'box_reconstruction_max_abs':reconstruction_max,'summaries':summaries}
    (OUT/'16_CPU开发组实际bin响应.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'verified':len(verified),'box_reconstruction_max_abs':reconstruction_max,'summaries':[r for r in summaries if r['size'] in ['ALL','<4','4-8']]},ensure_ascii=False))

if __name__=='__main__': main()
