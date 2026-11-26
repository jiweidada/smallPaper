"""Finite frozen native-state crossing. No training, fitting or BN update."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,hashlib,copy,time,contextlib,io,traceback,collections,gc
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E=ROOT/'data/00_实验计划与证据索引'
SNAP=ROOT/'code/local_candidates_20261004_v2';sys.path.insert(0,str(SNAP));import bootstrap
import numpy as np
import torch,torchvision
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.ops import box_convert
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from engine.deim.postprocessor import PostProcessor
from engine.data.transforms._transforms import ConvertPILImage
ER=E/'单一候选评价依赖补齐与构建核验_20261008_v1/isolated_runtime';sys.path.insert(0,str(ER))
from pycocoeval.coco import COCO
from pycocoeval.cocoeval import COCOeval
torch.set_num_threads(4);torch.set_num_interop_threads(1)
VAL=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
ORIG=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
INPUT=E/'正式132轮基线模型下载_20261008_v1'
OUT=HERE/'BN132_attempt01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def evaluate(ds,rows,name):
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        gt=COCO();gt.dataset=copy.deepcopy(ds);gt.createIndex();dt=gt.loadRes(copy.deepcopy(rows))
        ev=COCOeval(gt,dt,'bbox',using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=100);ev.evaluate();ev.accumulate();ev.summarize()
    (OUT/f'{name}_eval.txt').write_text(log.getvalue(),encoding='utf-8')
    metrics={k:float(v) for k,v in zip(['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl'],ev.stats)}
    hits={50:set(),75:set()};images=collections.defaultdict(dict)
    for e in ev.evalImgs:
        if e is None or e['aRng']!=ev.params.areaRng[0]:continue
        for th,ti in [(50,0),(75,5)]:
            hh={int(a)-1 for a,m,ig in zip(e['gtIds'],e['gtMatches'][ti],e['gtIgnore']) if m>0 and not ig}
            hits[th]|=hh;images[int(e['image_id'])].setdefault(th,set()).update(hh)
    return metrics,hits,{str(i):{str(t):sorted(v) for t,v in h.items()} for i,h in images.items()}

@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter()
    proto=HERE/'02_BN132交叉前向_冻结协议.json';ph=sha(proto);(HERE/'02_BN132交叉前向_冻结协议.sha256').write_text(ph,encoding='utf-8')
    ann=VAL/'annotations/val.json';ds=read(ann)
    infos=sorted(sorted(ds['images'],key=lambda x:hashlib.sha256(f'BN132-cross-v1|{x["id"]}'.encode()).hexdigest())[:16],key=lambda x:x['id'])
    selected={i['id'] for i in infos};sub=copy.deepcopy(ds);sub['images']=infos;sub['annotations']=[a for a in sub['annotations'] if a['image_id'] in selected]
    original_ids=[a['id'] for a in sub['annotations']]
    for a in sub['annotations']:a['id']+=1
    sources={str(ann):sha(ann),str(proto):ph,str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    for p in [ER/'pycocoeval/coco.py',ER/'pycocoeval/cocoeval.py',SNAP/'bootstrap.py']:sources[str(p)]=sha(p)
    transform=T.Compose([T.Resize([640,640]),ConvertPILImage(dtype='float32',scale=True)])
    batches=[]
    for start in range(0,16,4):
        batch=infos[start:start+4];ims=[]
        for im in batch:
            p=VAL/'images'/im['file_name'];sources[str(p)]=sha(p)
            with Image.open(p) as x:
                x=x.convert('RGB');assert x.size==(im['width'],im['height']);ims.append(transform(x).as_subclass(torch.Tensor))
        batches.append((batch,torch.stack(ims)))
    write(OUT/'preflight.json',{'protocol_sha256':ph,'selected_images':infos,'GT_count':len(sub['annotations']),
        'GT_class_counts':dict(collections.Counter(a['category_id'] for a in sub['annotations'])),
        'torch':torch.__version__,'torchvision':torchvision.__version__,'batch_size':4,'threads':4,'selection_uses_outcomes':False})
    result={'protocol_sha256':ph,'GT':len(sub['annotations']),'image_ids':sorted(selected),'seeds':{},'source_hashes':sources,
        'native_batch_calls':0,'image_equivalent_forwards':0,'additional_head_calls':0,'CPU_evaluator_calls':0,'training':0,'buffer_updates':0}
    for seed in [1,2]:
        sd=ORIG/f'seed{seed}';mf=read(sd/'manifest.json');sources[str(sd/'manifest.json')]=sha(sd/'manifest.json')
        for rel,h in mf['source_hashes'].items():
            p=SNAP/rel;assert sha(p)==h,(rel,'native_source_mismatch');sources[str(p)]=h
        cp=INPUT/f'seed{seed}/last.pth';sources[str(cp)]=sha(cp);assert sources[str(cp)]==mf['checkpoint_sha256']
        cfg=INPUT/f'seed{seed}/run/args.json';sources[str(cfg)]=sha(cfg);y=read(cfg)['yaml_cfg']
        ck=torch.load(cp,map_location='cpu',weights_only=True);assert ck['last_epoch']==131
        raw=ck['model'];ema=ck['ema']['module'];assert set(raw)==set(ema)
        bnkeys=sorted(k for k in raw if k.endswith('.running_mean') or k.endswith('.running_var'));assert len(bnkeys)==198
        model=DEIM(HGNetv2(**copy.deepcopy(y['HGNetv2'])),HybridEncoder(**copy.deepcopy(y['HybridEncoder']),eval_spatial_size=[640,640]),
            DFINETransformer(**copy.deepcopy(y['DFINETransformer']),num_classes=10,eval_spatial_size=[640,640]),num_classes=10).cpu().eval()
        assert all(not m.training for m in model.modules())
        assert sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules())==99
        pp=PostProcessor(num_classes=10,num_top_queries=300,use_focal_loss=True,remap_mscoco_category=False,score_threshold=.001).eval()
        origp=sd/'predictions.json';sources[str(origp)]=sha(origp);old=read(origp);oldby=collections.defaultdict(list)
        for d in old:
            if d['image_id'] in selected:oldby[d['image_id']].append(d)
        cachemanifest={v['file']:v['sha256'] for v in read(sd/'array_manifest.json')};sources[str(sd/'array_manifest.json')]=sha(sd/'array_manifest.json')
        per={};hitdict={}
        for arm in ['EE','RR','RE','ER']:
            owner=ema if arm[0]=='E' else raw;bs=ema if arm[1]=='E' else raw;state=dict(owner)
            for k in bnkeys:state[k]=bs[k]
            model.load_state_dict(state,strict=True)
            assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
            before=state_sha(model);rows=[];parity={'logits_equal_images':0,'boxes_equal_images':0,'export_equal_images':0}
            od=OUT/f'seed{seed}_{arm}';od.mkdir()
            for batch,x in batches:
                o=model(x);result['native_batch_calls']+=1;result['image_equivalent_forwards']+=len(batch)
                assert torch.isfinite(o['pred_boxes']).all() and torch.isfinite(o['pred_logits']).all()
                sizes=torch.tensor([[im['width'],im['height']] for im in batch]);pred=pp(o,sizes,for_eval=True)
                for j,im in enumerate(batch):
                    iid=im['id'];np.savez_compressed(od/f'{iid:06d}.npz',boxes=o['pred_boxes'][j].numpy(),logits=o['pred_logits'][j].numpy())
                    b=box_convert(pred[j]['boxes'],'xyxy','xywh').numpy();local=[]
                    assert (b[:,2:]>0).all(),('invalid_export_box',seed,arm,iid)
                    for c,s,box in zip(pred[j]['labels'].tolist(),pred[j]['scores'].tolist(),b.tolist()):local.append({'image_id':iid,'category_id':c,'score':s,'bbox':box})
                    if arm=='EE':
                        p=sd/'arrays'/f'{iid:06d}.npz';sources[str(p)]=sha(p);assert sources[str(p)]==cachemanifest[p.name]
                        with np.load(p) as z:
                            assert np.array_equal(o['pred_logits'][j].numpy(),z['logits']),('EMA_logits_parity',seed,iid)
                            assert np.array_equal(o['pred_boxes'][j].numpy(),z['layer_boxes'][-1]),('EMA_boxes_parity',seed,iid)
                        assert local==oldby[iid],('EMA_export_parity',seed,iid)
                        for k in parity:parity[k]+=1
                    rows.extend(local)
            assert state_sha(model)==before,('inference_changed_state',seed,arm)
            metrics,hits,perim=evaluate(sub,rows,f'seed{seed}_{arm}');result['CPU_evaluator_calls']+=1;hitdict[arm]=hits
            write(od/'predictions.json',rows)
            per[arm]={'metrics':metrics,'TP50':len(hits[50]),'TP75':len(hits[75]),'GT':len(sub['annotations']),
                'hit_GT_ids':{str(t):sorted(h) for t,h in hits.items()},'per_image_hit_GT_ids':perim,
                'state_hash_unchanged':before,'intended_state_load_exact':True,'reference_parity':parity,'outputs':len(rows)}
            print(json.dumps({'seed':seed,'arm':arm,'AP_pp':metrics['AP']*100,'APs_pp':metrics['APs']*100,'TP50':len(hits[50]),'GT':len(sub['annotations']),
                'EE_parity_images':parity['logits_equal_images'],'elapsed_seconds':time.perf_counter()-started},ensure_ascii=False),flush=True)
        comparisons={}
        for a,b in [('ER','EE'),('RE','RR'),('EE','RR')]:
            comparisons[f'{a}_minus_{b}']={'delta_pp':{k:(per[a]['metrics'][k]-per[b]['metrics'][k])*100 for k in per[a]['metrics']},
                '50':{'new':len(hitdict[a][50]-hitdict[b][50]),'lost':len(hitdict[b][50]-hitdict[a][50]),'both':len(hitdict[a][50]&hitdict[b][50]),'neither':len(set(original_ids)-hitdict[a][50]-hitdict[b][50])},
                '75':{'new':len(hitdict[a][75]-hitdict[b][75]),'lost':len(hitdict[b][75]-hitdict[a][75]),'both':len(hitdict[a][75]&hitdict[b][75]),'neither':len(set(original_ids)-hitdict[a][75]-hitdict[b][75])}}
        primary=comparisons['ER_minus_EE'];advance=primary['delta_pp']['AP']>1e-4 and primary['delta_pp']['APs']>1e-4 and primary['50']['new']>primary['50']['lost']
        sr={'arms':per,'comparisons':comparisons,'advance_fixed_hypothesis':bool(advance),'BN_tensors_crossed':bnkeys,'scope':'16-image frozen screen only; raw stats are not known correct population stats'}
        result['seeds'][str(seed)]=sr;write(OUT/f'seed{seed}_results.json',sr)
        del model,ck,raw,ema;gc.collect()
    assert all(sha(Path(p))==h for p,h in sources.items());assert sha(proto)==ph
    result['both_seeds_advance']=all(x['advance_fixed_hypothesis'] for x in result['seeds'].values())
    result['elapsed_seconds']=time.perf_counter()-started;result['source_hashes_unchanged']=True
    write(OUT/'results.json',result);write(OUT/'completion.json',{'status':'complete','source_hashes_unchanged':True,'training':0,'new_model_forwards':result['image_equivalent_forwards'],
        'CPU_evaluator_calls':result['CPU_evaluator_calls'],'both_seeds_advance':result['both_seeds_advance'],'seconds':result['elapsed_seconds']})
    print(json.dumps({'completed':True,'both_seeds_advance':result['both_seeds_advance'],'forwards':result['image_equivalent_forwards'],'seconds':result['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        (OUT/'failure.json').write_text(json.dumps({'status':'FAILED','traceback':traceback.format_exc(),'scientific_interpretation':'none if parity failed'},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
