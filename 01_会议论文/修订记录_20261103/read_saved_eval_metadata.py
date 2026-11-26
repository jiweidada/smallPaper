"""Decode inert metadata and already-recorded scalars; never call an evaluator."""
from pathlib import Path
import pickle,io,struct,json,hashlib,zipfile,math,collections,shutil
OUT=Path(__file__).resolve().parent/'dataset_eval_audit'
DEIM=Path(r'C:\Users\jiwei\Desktop\DEIM')
class InertDtype:
    def __init__(self,name,*args):self.name=name;self.byteorder='<'
    def __setstate__(self,state):self.byteorder=state[1]
class InertArray:
    def __setstate__(self,state):self.version,self.shape,self.dtype,self.fortran,self.raw=state
class InertParams:pass
def decode_scalar(dtype,raw):
    codes={'i8':'q','i4':'i','f8':'d','f4':'f','b1':'?','u8':'Q','u4':'I'}
    if dtype.name not in codes:raise ValueError('Unsupported dtype '+str(dtype.name))
    return struct.unpack(('>' if dtype.byteorder=='>' else '<')+codes[dtype.name],raw)[0]
def reconstruct(*args):return InertArray()
def encode(s,encoding):
    if encoding!='latin1':raise ValueError(encoding)
    return s.encode(encoding)
class SafeMetadata(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='faster_coco_eval.core.cocoeval' and name=='Params':return InertParams
        if module=='numpy' and name=='dtype':return InertDtype
        if module=='numpy' and name=='ndarray':return InertArray
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct':return reconstruct
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='scalar':return decode_scalar
        if module=='_codecs' and name=='encode':return encode
        raise ValueError('Unsupported global '+module+'.'+name)
def plain(x):
    if isinstance(x,InertArray):
        if math.prod(x.shape)>1100:return {'shape':x.shape,'dtype':x.dtype.name,'contents':'not decoded'}
        size=int(x.dtype.name[1:]);values=[decode_scalar(x.dtype,x.raw[i:i+size]) for i in range(0,len(x.raw),size)]
        return {'shape':x.shape,'dtype':x.dtype.name,'values':values}
    if isinstance(x,InertParams):return {k:plain(v) for k,v in vars(x).items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    raise ValueError(str(type(x)))
probes=json.loads((OUT/'saved_eval_probe.json').read_text(encoding='utf-8'))
records=[]
for item in probes:
    p=Path(item['path'])
    with zipfile.ZipFile(p) as z:
        n=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        d=SafeMetadata(io.BytesIO(z.read(n))).load()
    r={'path':str(p),'sha256':item['sha256'],'keys':list(d),'params':plain(d['params']),'date':d.get('date'),'counts':plain(d.get('counts')),'stored_array_shapes':{k:plain(v) for k,v in d.items() if isinstance(v,InertArray)},'scope':'Metadata of an existing evaluation, not a new evaluation. Does not identify the generating checkpoint hash.'}
    records.append(r)
    print(json.dumps({'path':str(p),'date':r['date'],'params':{k:v for k,v in r['params'].items() if k not in ['imgIds','recThrs']},'counts':r['counts']},ensure_ascii=True),flush=True)
(OUT/'saved_eval_metadata.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

# Minimal protobuf wire reader for TensorBoard's existing Summary.Value simple_value.
def varint(b,i):
    v=0;s=0
    while True:
        x=b[i];i+=1;v|=(x&127)<<s
        if x<128:return v,i
        s+=7
        if s>70:raise ValueError('Invalid varint')
def fields(b):
    i=0
    while i<len(b):
        tag,i=varint(b,i);wire=tag&7;key=tag>>3
        if wire==0:value,i=varint(b,i)
        elif wire==1:value=b[i:i+8];i+=8
        elif wire==2:n,i=varint(b,i);value=b[i:i+n];i+=n
        elif wire==5:value=b[i:i+4];i+=4
        else:raise ValueError('Unsupported wire type')
        yield key,wire,value
events=[]
for p in (DEIM/'outputs/deim_hgnetv2_s_MSCA_EnhancedStem_SiLU_320_eval/summary').glob('events*'):
    b=p.read_bytes();i=0;values=[]
    while i<len(b):
        n=struct.unpack('<Q',b[i:i+8])[0];i+=12;event=b[i:i+n];i+=n+4
        f=list(fields(event));step=next((v for k,w,v in f if k==2),0)
        for k,w,s in f:
            if k!=5:continue
            for k2,w2,v2 in fields(s):
                if k2!=1:continue
                fs=list(fields(v2));tag=next((v.decode('utf-8','replace') for k,w,v in fs if k==1),'')
                value=next((struct.unpack('<f',v)[0] for k,w,v in fs if k==2 and w==5),None)
                if value is not None:values.append({'step':step,'tag':tag,'simple_value':value})
    events.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'scalars':values,'scope':'Existing float32 summary values only; no computation of detection metrics.'})
(OUT/'saved_eval_tensorboard_scalars.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
print('Existing S1 evaluation scalar tags',json.dumps(events[0]['scalars'] if events else [],ensure_ascii=True))

# Static annotation correspondence, not prediction evaluation.
comparisons={}
for split,fn in [('train','train.json'),('val','val.json')]:
    directory=DEIM/'dataset_visdrone'/f'VisDrone2019-DET-{split}'
    data=json.loads((directory/'annotations'/fn).read_text(encoding='utf-8'))
    grouped=collections.defaultdict(list)
    for a in data['annotations']:grouped[a['image_id']].append(a)
    count_mismatch=[];category_mismatch=0;coords=0;paired=0
    for im in data['images']:
        raw=[]
        for line in (directory/'annotations'/Path(im['file_name']).with_suffix('.txt')).read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                row=[float(v) for v in line.strip().rstrip(',').split(',')]
                if row[4]==1 and 1<=row[5]<=10:raw.append(row)
        converted=grouped[im['id']]
        if len(raw)!=len(converted):count_mismatch.append(im['file_name']);continue
        for a,b in zip(raw,converted):
            category_mismatch+=int(int(a[5])-1!=b['category_id'])
            coords=max(coords,max(abs(a[j]-b['bbox'][j]) for j in range(4)));paired+=1
    comparisons[split]={'pairing':'Original score=1 categories1..10, original within-image order against current COCO within-image order; category mapped minus1','paired_boxes':paired,'images_with_count_mismatch':count_mismatch,'category_mismatches':category_mismatch,'maximum_absolute_bbox_coordinate_difference_pixels':coords}
(OUT/'annotation_correspondence.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2),encoding='utf-8')
print('Annotation correspondence',json.dumps(comparisons,ensure_ascii=True))
