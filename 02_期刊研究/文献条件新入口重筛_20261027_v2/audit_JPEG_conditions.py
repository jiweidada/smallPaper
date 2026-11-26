"""Natural JPEG headers and fixed native hits; no pixel decode or detector."""
import pathlib,json,hashlib,collections,time,traceback
from PIL import Image,__version__ as pil_version
import numpy as np
B=pathlib.Path(__file__).resolve().parent;R=B.parent
DATA=pathlib.Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone')
TRAIN_D=pathlib.Path('D:/数据集/visdrone2019/VisDrone2019-DET-train/images')
PREV=R/'具体问题重新筛选_20261027_v1'
PROTO=B/'01_自然JPEG编码条件_冻结协议.json';OUT=B/'JPEG_condition_attempt01'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
 return h.hexdigest()
def save(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def header(p):
 with Image.open(p) as im:
  assert im.format=='JPEG',('unexpected_format',str(p),im.format)
  q={str(k):list(v) for k,v in sorted(im.quantization.items())};assert all(len(v)==64 for v in q.values())
  layers=[list(v) for v in im.layer];key={'quantization':q,'sampling_and_table':[v[1:] for v in layers]}
  h=hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  return {'signature':h,'width':im.width,'height':im.height,'mode':im.mode,'format':im.format,'component_layers':layers,'progressive':bool(im.info.get('progressive',im.info.get('progression',False))),'signature_fields':key}
def main():
 started=time.monotonic();OUT.mkdir(exist_ok=False);bindings={str(PROTO):sha(PROTO),str(pathlib.Path(__file__)):sha(__file__)}
 rows=[];signatures={};datasets={};copy_differences=[]
 for split in ['train','val']:
  ddir=DATA/f'VisDrone2019-DET-{split}';annp=ddir/'annotations'/f'{split}.json';ann=read(annp);bindings[str(annp)]=sha(annp)
  if split=='train':assert bindings[str(annp)]=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
  else:assert bindings[str(annp)]=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
  datasets[split]=ann;by=collections.defaultdict(list)
  for a in ann['annotations']:by[a['image_id']].append(a)
  for j,im in enumerate(ann['images']):
   p=(TRAIN_D if split=='train' else ddir/'images')/im['file_name'];bindings[str(p)]=sha(p);head=header(p)
   assert [head['width'],head['height']]==[im['width'],im['height']]
   sig=head.pop('signature_fields');signatures[head['signature']]=sig
   row={'split':split,'image_id':im['id'],'file_name':im['file_name'],'GT':len(by[im['id']]),'class_GT':dict(collections.Counter(a['category_id'] for a in by[im['id']])),'path':str(p),'sha256':bindings[str(p)],**head}
   if split=='train':
    cp=ddir/'images'/im['file_name'];bindings[str(cp)]=sha(cp);row['C_copy_sha256']=bindings[str(cp)];row['current_C_D_bytes_equal']=bindings[str(cp)]==bindings[str(p)]
    if not row['current_C_D_bytes_equal']:copy_differences.append({'image_id':im['id'],'D_header':head,'C_header':header(cp)})
   rows.append(row)
   if (j+1)%2000==0:print(json.dumps({'split':split,'headers':j+1,'seconds':time.monotonic()-started}),flush=True)
 train=collections.Counter(r['signature'] for r in rows if r['split']=='train');val=collections.Counter(r['signature'] for r in rows if r['split']=='val')
 summary={'scope':read(PROTO)['precise_question'],'status':'complete','pillow':pil_version,'split_counts':{s:{'images':sum(r['split']==s for r in rows),'GT':sum(r['GT'] for r in rows if r['split']==s),'unique_encoding_signatures':len(set(r['signature'] for r in rows if r['split']==s))} for s in ['train','val']},'current_C_D_training_byte_mismatches':len(copy_differences),'groups':{},'unseen_validation':{},'total_variation_descriptive':.5*sum(abs(train[k]/6471-val[k]/548) for k in set(train)|set(val)),'inputs':bindings,'model_forwards':0,'RGB_decodes':0,'new_CPU_evaluations':0,'training':0}
 for k,sig in signatures.items():
  summary['groups'][k]={'fields':sig}
  for split in ['train','val']:
   rr=[r for r in rows if r['split']==split and r['signature']==k]
   summary['groups'][k][split]={'images':len(rr),'GT':sum(r['GT'] for r in rr),'class_GT':dict(sum((collections.Counter(r['class_GT']) for r in rr),collections.Counter()))}
 unseen=[r for r in rows if r['split']=='val' and r['signature'] not in train]
 summary['unseen_validation']={'images':len(unseen),'image_denominator':548,'GT':sum(r['GT'] for r in unseen),'GT_denominator':38759,'image_ids':[r['image_id'] for r in unseen]}
 valann=datasets['val'];gts={a['id']:a for a in valann['annotations']};idsig={r['image_id']:r['signature'] for r in rows if r['split']=='val'}
 seal={x['path']:x['sha256'] for x in read(PREV/'delivery_manifest.json')['inputs']+read(PREV/'delivery_manifest.json')['outputs']}
 for seed in [1,2]:
  path=PREV/f'06_seed{seed}_fixed_rank_arrays.npz';bindings[str(path)]=sha(path);assert bindings[str(path)]==seal[str(path)]
  hits={50:set(),75:set()}
  with np.load(path) as z:
   for c in range(10):
    for th,ti in [(50,0),(75,5)]:
     m=z[f'a0_c{c}_matches'][ti];ig=z[f'a0_c{c}_ignored'][ti];hits[th].update(int(x)-1 for x in m[(m>0)&~ig])
  assert [len(hits[t]) for t in [50,75]]==([29682,18702] if seed==1 else [29745,18889])
  for k in signatures:
   group=summary['groups'][k]['val'];group[f'seed{seed}']={f'TP{t}':sum(idsig[gts[g]['image_id']]==k for g in hits[t]) for t in [50,75]}
  summary['unseen_validation'][f'seed{seed}']={f'TP{t}':sum(idsig[gts[g]['image_id']] not in train for g in hits[t]) for t in [50,75]}
 for p,h in bindings.items():assert sha(p)==h,('source_changed',p)
 summary['condition_status']='uniform_current_encoding' if len(signatures)==1 else 'natural_encoding_heterogeneity_found_not_causal'
 summary['seconds']=time.monotonic()-started
 save(OUT/'rows.json',rows);save(OUT/'copy_differences.json',copy_differences);save(OUT/'results.json',summary);save(OUT/'sha256.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
 print(json.dumps({'split_counts':summary['split_counts'],'unique_all':len(signatures),'copy_mismatches':len(copy_differences),'unseen_validation':summary['unseen_validation'],'TV':summary['total_variation_descriptive'],'condition_status':summary['condition_status'],'seconds':summary['seconds']}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'scientific_result':'NA'})
  raise
