from pathlib import Path
import json, csv, hashlib, shutil, statistics, zipfile, pickle, io, collections, math

OUT=Path(__file__).resolve().parent
SRC=Path(r'C:\Users\jiwei\Desktop\DEIM')
RUNS={
 'N0':'deim_hgnetv2_n_custom320',
 'N1':'deim_hgnetv2_EnhancedStem_NOMSCA_NOSilu_n_visdrone_320',
 'N2':'deim_hgnetv2_EnhancedStemNO_MSCA_NOSilu_n_visdrone_320',
 'N3':'deim_hgnetv2_EnhancedStemNO_NOMSCA_Silu_n_visdrone_320',
 'N4':'deim_hgnetv2_MSCA_EnhancedStem_SiLU_320',
 'S1':'deim_hgnetv2_s_MSCA_EnhancedStem_SiLU_320'}
STATS=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
manifest=[]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()
def snap(p,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest)
    manifest.append({'source':str(p),'snapshot':str(dest),'sha256':sha(p)})
def flat(d,p=''):
    o={}
    if isinstance(d,dict):
        for k,v in d.items():o.update(flat(v,f'{p}.{k}' if p else k))
    else:o[p]=d
    return o

class TensorMeta:
    def __init__(self,storage,offset,size,stride,*args): self.shape=tuple(size); self.numel=math.prod(size)
class MetaUnpickler(pickle.Unpickler):
    # Tensor storage bytes are never loaded; only inert shapes/keys are read.
    def find_class(self,module,name):
        if module=='collections' and name=='OrderedDict':return collections.OrderedDict
        if module=='torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):return TensorMeta
        if module=='torch' and name.endswith('Storage'):return name
        if module=='torch._utils' and name=='_rebuild_parameter':return lambda data,*args:data
        if module=='builtins' and name=='set':return set
        raise ValueError(f'Unsupported metadata global {module}.{name}')
    def persistent_load(self,pid):return ('inert_storage',pid)
def shapes(p):
    with zipfile.ZipFile(p) as z:
        name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        data=MetaUnpickler(io.BytesIO(z.read(name))).load()
    sd=data.get('model',{})
    sm={k:{'shape':v.shape,'numel':v.numel} for k,v in sd.items() if isinstance(v,TensorMeta)}
    totals={}
    for k,v in sm.items():
        if k.endswith(('running_mean','running_var','num_batches_tracked')):continue
        part='.'.join(k.split('.')[:3]) if k.startswith('backbone.stages.') else '.'.join(k.split('.')[:2])
        totals[part]=totals.get(part,0)+v['numel']
    return {'epoch':data.get('last_epoch',data.get('epoch')),'top_keys':list(data),'model_parameter_numel_excluding_bn_buffers':sum(totals.values()),'parts':totals,'model_shapes':sm}

report={}; configs={}
for ident,name in RUNS.items():
    d=SRC/'outputs'/name; snapdir=OUT/'source_snapshot'/ident
    for fn in ['args.json','log.txt']:snap(d/fn,snapdir/fn)
    args=json.loads((d/'args.json').read_text(encoding='utf-8'));cfg=args['yaml_cfg'];configs[ident]=cfg
    rows=[]
    for no,line in enumerate((d/'log.txt').read_text(encoding='utf-8').splitlines(),1):
        try:r=json.loads(line)
        except ValueError:continue
        if len(r.get('test_coco_eval_bbox',[]))>=12: rows.append((no,r))
    no,best=max(rows,key=lambda r:r[1]['test_coco_eval_bbox'][0]);tail=rows[-20:]
    backbone=cfg['DEIM']['backbone']
    rec={'run':name,'source_dir':str(d),'n_logged_evaluations':len(rows),'unique_epochs':len(set(r['epoch'] for _,r in rows)),
      'first_epoch':rows[0][1]['epoch'],'last_epoch':rows[-1][1]['epoch'],'best_epoch':best['epoch'],'best_log_line':no,
      'n_parameters':best['n_parameters'],'metrics':dict(zip(STATS,best['test_coco_eval_bbox'])),
      'last_AP':rows[-1][1]['test_coco_eval_bbox'][0], 'last20_AP_mean':statistics.mean(r['test_coco_eval_bbox'][0] for _,r in tail),
      'last20_AP_sample_std':statistics.stdev(r['test_coco_eval_bbox'][0] for _,r in tail),
      'last20_AP_range':max(r['test_coco_eval_bbox'][0] for _,r in tail)-min(r['test_coco_eval_bbox'][0] for _,r in tail),
      'plateau_count_within_0.002':sum(r['test_coco_eval_bbox'][0]>=best['test_coco_eval_bbox'][0]-0.002 for _,r in rows),
      'seed':args.get('seed'),'resume':args.get('resume'),'tuning':args.get('tuning'),'backbone_class':backbone,'backbone':cfg[backbone],
      'train_batch':cfg['train_dataloader'].get('total_batch_size'),'val_batch':cfg['val_dataloader'].get('total_batch_size'),
      'optimizer':cfg['optimizer'],'train_transforms':cfg['train_dataloader']['dataset']['transforms'],
      'collate':cfg['train_dataloader']['collate_fn'],'encoder':cfg['HybridEncoder'],'decoder':cfg['DFINETransformer']}
    cp=d/'best_stg2.pth'
    try:
        meta=shapes(cp); (snapdir/'checkpoint_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        rec['checkpoint']={k:v for k,v in meta.items() if k!='model_shapes'};rec['checkpoint']['sha256']=sha(cp)
    except Exception as ex:rec['checkpoint_metadata_error']=repr(ex)
    report[ident]=rec
    print(ident, name, json.dumps({k:rec[k] for k in ['best_epoch','n_parameters','metrics','seed','train_batch','backbone_class','backbone','checkpoint'] if k in rec},ensure_ascii=True))

diffs={}
for ident,cfg in configs.items():
    a,b=flat(configs['N0']),flat(cfg)
    diffs[ident]={k:{'N0':a.get(k),'this':b.get(k)} for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
for rel in ['engine/backbone/hgnetv2.py','engine/backbone/hgnetv2_silu.py','engine/backbone/hgnetv2_MSCA_EnhancedStem.py','engine/backbone/hgnetv2_MSCA.py','engine/extre_module/custom_nn/attention/MSCA.py','engine/extre_module/custom_nn/attention/simam.py','engine/solver/det_solver.py','engine/data/dataset/coco_eval.py','engine/data/transforms/container.py','engine/data/dataloader.py']:
    p=SRC/rel
    if p.exists():snap(p,OUT/'source_snapshot'/'current_code'/rel)
(OUT/'run_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'configuration_differences.json').write_text(json.dumps(diffs,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
