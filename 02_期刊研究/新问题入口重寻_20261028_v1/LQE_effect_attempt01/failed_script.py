"""Cached output-node intervention. No model/checkpoint/head imports or fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, copy, hashlib, io, contextlib, time, traceback, collections
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
E=ROOT/'data/00_实验计划与证据索引'
ER=E/'单一候选评价依赖补齐与构建核验_20261008_v1/isolated_runtime'
sys.path.insert(0,str(ER))
from pycocoeval.coco import COCO
from pycocoeval.cocoeval import COCOeval
BASE=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
ANN=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
PROTO=HERE/'01_LQE作用检验_事前协议.json'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def cap_rows(rows):
    groups=collections.defaultdict(list)
    for i,d in enumerate(rows):groups[(d['image_id'],d['category_id'])].append(i)
    ids=[]
    for ix in groups.values():ids.extend(sorted(ix,key=lambda i:-rows[i]['score'])[:100])
    return sorted(ids)
def metrics(e):
    p=e.eval['precision']
    def avg(a,t=None,c=None):
        v=p[:,:,:,a,2] if t is None else p[t,:,:,a,2]
        if c is not None:v=v[:,:,c] if t is None else v[:,c]
        return float(v[v>=0].mean()*100) if np.any(v>=0) else None
    return {'AP':avg(0),'AP50':avg(0,0),'AP75':avg(0,5),'APs':avg(1),'APs50':avg(1,0),'APs75':avg(1,5),
            'by_IoU_AP':[avg(0,t) for t in range(10)],'by_class_AP':[avg(0,c=c) for c in range(10)]}
def evaluate(rows,ds,tag,out):
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        gt=COCO();gt.dataset=copy.deepcopy(ds);gt.createIndex();dt=gt.loadRes(copy.deepcopy(rows))
        e=COCOeval(gt,dt,'bbox',using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=100)
        e.params.imgIds=sorted(x['id'] for x in ds['images']);e.evaluate();e.accumulate();e.summarize()
    (out/(tag+'_eval.txt')).write_text(log.getvalue(),encoding='utf-8')
    hits={a:{t:set() for t in [0,5]} for a in [0,1]}
    for z in e.evalImgs:
        if z is None:continue
        a=e.params.areaRng.index(z['aRng'])
        if a not in hits:continue
        for t in hits[a]:
            v=np.asarray(z['dtMatches'])[t];k=~np.asarray(z['dtIgnore'])[t].astype(bool)&(v>0)
            hits[a][t].update(int(x)-1 for x in v[k])
    return metrics(e),hits
def main():
    start=time.perf_counter();out=HERE/'LQE_effect_attempt01';out.mkdir(exist_ok=False)
    sources={str(PROTO):sha(PROTO),str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(ANN):sha(ANN)}
    assert sources[str(ANN)]=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    (HERE/'01_LQE作用检验_事前协议.sha256').write_text(sources[str(PROTO)],encoding='utf-8')
    for p in [ER/'pycocoeval/coco.py',ER/'pycocoeval/cocoeval.py',ER/'pycocoeval/_mask.cp310-win_amd64.pyd']:
        sources[str(p)]=sha(p)
    oldstate=ROOT/'raw_EMA逐GT对照_20261026_v1/work_state_v2.json'
    (HERE/'previous_work_state_v8.json').write_bytes(oldstate.read_bytes())
    ds=read(ANN);assert len(ds['images'])==548 and len(ds['annotations'])==38759
    images={x['id']:x for x in ds['images']}
    gtall={int(x['id']) for x in ds['annotations']};gtsmall={int(x['id']) for x in ds['annotations'] if x['area']<=1024 and not x.get('iscrowd',0)}
    shifted=copy.deepcopy(ds)
    for a in shifted['annotations']:a['id']+=1
    result={'status':'RUNNING','protocol_sha256':sha(PROTO),'seeds':{},'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':2},'evaluator_calls':0,
            'new_models_checkpoints_heads_forwards_backwards_training_GPU_remote_test_reads':0,'sources':sources}
    for seed in [1,2]:
        b=BASE/f'seed{seed}'
        for name in ['predictions.json','prediction_sources.json','array_manifest.json','manifest.json','run_unused']:
            p=b/name
            if p.exists():sources[str(p)]=sha(p)
        original=read(b/'predictions.json');links=read(b/'prediction_sources.json');am=read(b/'array_manifest.json')
        assert len(original)==len(links)==164400
        iby=collections.defaultdict(list)
        for i,(d,l) in enumerate(zip(original,links)):
            assert l['row']==i and l['image_id']==d['image_id'] and l['class']==d['category_id']
            iby[d['image_id']].append(i)
        assert set(iby)==set(images)
        hashes={v['file']:v['sha256'] for v in am}
        raw_of_export=np.zeros(len(original),np.float32);raw_pipeline=[];raw_pipeline_links=[]
        max_common=0.;max_score=0.;max_box=0.;shifts=[];nquery=0;argmax_changed=0
        for iid in sorted(images):
            p=b/'arrays'/f'{iid:06d}.npz';sources[str(p)]=sha(p);assert sources[str(p)]==hashes[p.name]
            with np.load(p) as d:
                assert int(d['image_id'])==iid
                z=torch.from_numpy(d['logits']);raw=torch.from_numpy(d['layer_logits_raw'][-1]);bb=torch.from_numpy(d['layer_boxes'][-1]);wh=d['image_wh']
            assert tuple(z.shape)==tuple(raw.shape)==(300,10)
            dz=(z-raw).numpy();max_common=max(max_common,float(np.max(np.ptp(dz,axis=1))))
            assert max_common<=3e-6,('not_common_LQE',seed,iid,max_common)
            shifts.extend(dz.mean(1).tolist());nquery+=300;argmax_changed+=int(torch.count_nonzero(z.argmax(1)!=raw.argmax(1)))
            score=z.sigmoid();rscore=raw.sigmoid()
            cx,cy,w,h=bb.unbind(-1)
            xyxy=torch.stack([cx-.5*w,cy-.5*h,cx+.5*w,cy+.5*h],-1)*torch.tensor([wh[0],wh[1],wh[0],wh[1]])
            x0,y0,x1,y1=xyxy.unbind(-1);xywh=torch.stack([x0,y0,x1-x0,y1-y0],-1).numpy()
            ss,idx=score.flatten().topk(300);idx=idx[ss>.001];ss=ss[ss>.001]
            ids=iby[iid];assert len(ids)==len(idx)==300
            for j,i in enumerate(ids):
                l=links[i];q,c=int(idx[j]//10),int(idx[j]%10)
                assert (q,c)==(l['query'],l['class'])
                max_score=max(max_score,abs(float(ss[j])-original[i]['score']))
                max_box=max(max_box,float(np.max(np.abs(xywh[q]-original[i]['bbox']))))
                raw_of_export[i]=float(rscore[q,c])
            rr,ii=rscore.flatten().topk(300);keep=rr>.001
            for s,k in zip(rr[keep].tolist(),ii[keep].tolist()):
                q,c=k//10,k%10
                raw_pipeline_links.append({'row':len(raw_pipeline),'image_id':iid,'query':q,'class':c})
                raw_pipeline.append({'image_id':iid,'category_id':c,'score':s,'bbox':xywh[q].tolist()})
        assert max_score==0 and max_box==0,(max_score,max_box)
        assert argmax_changed==0,argmax_changed
        ids=cap_rows(original);fixed_native=[original[i] for i in ids]
        fixed_raw=[dict(original[i],score=float(raw_of_export[i])) for i in ids]
        arms={'native':original,'native_fixed_cap_identity':fixed_native,'raw_fixed_cap_identity':fixed_raw,'raw_native_pipeline':raw_pipeline}
        R={'counts':{'allGT':len(gtall),'smallGT':len(gtsmall),'exported_rows':len(original),'native_cap_rows':len(ids),'raw_pipeline_rows':len(raw_pipeline),'queries':nquery},
           'parity':{'max_native_score_error':max_score,'max_native_box_error':max_box,'max_class_common_delta_error':max_common,'argmax_changes':argmax_changed},
           'LQE_logit_shift_quantiles':{str(q):float(np.quantile(shifts,q)) for q in [0,.1,.25,.5,.75,.9,1]},'arms':{}}
        H={}
        for tag,rows in arms.items():
            print(json.dumps({'seed':seed,'arm':tag,'rows':len(rows),'elapsed_s':time.perf_counter()-start}),flush=True)
            m,hits=evaluate(rows,shifted,f'seed{seed}_{tag}',out);result['evaluator_calls']+=1;H[tag]=hits
            R['arms'][tag]={'metrics_pp':m,'TP_counts':{str(a):{str(.5+t*.05):len(v) for t,v in h.items()} for a,h in hits.items()}}
            save(out/f'seed{seed}_{tag}_predictions.json',rows)
        assert max(abs(R['arms']['native']['metrics_pp'][k]-R['arms']['native_fixed_cap_identity']['metrics_pp'][k]) for k in ['AP','APs','AP50','AP75'])<1e-10
        expected={1:27.4278001473,2:27.4748391766}[seed]
        assert abs(R['arms']['native']['metrics_pp']['AP']-expected)<1e-8
        R['comparisons']={}
        for tag in ['raw_fixed_cap_identity','raw_native_pipeline']:
            comp={'delta_pp':{k:R['arms'][tag]['metrics_pp'][k]-R['arms']['native']['metrics_pp'][k] for k in ['AP','APs','AP50','AP75','APs50','APs75']},'GT_transition':{}}
            for area,den in [(0,gtall),(1,gtsmall)]:
                comp['GT_transition'][str(area)]={}
                for t in [0,5]:
                    a=H['native'][area][t];bset=H[tag][area][t]
                    assert a<=den and bset<=den
                    comp['GT_transition'][str(area)][str(.5+t*.05)]={'denominator':len(den),'both':len(a&bset),'recovered':len(bset-a),'reverse_loss':len(a-bset),'neither':len(den-(a|bset))}
            R['comparisons'][tag]=comp
        nkeys={(l['image_id'],l['query'],l['class']) for l in links};rkeys={(l['image_id'],l['query'],l['class']) for l in raw_pipeline_links}
        R['selection']={'same_pairs':len(nkeys&rkeys),'raw_new_pairs':len(rkeys-nkeys),'native_lost_pairs':len(nkeys-rkeys),'note':'paircounts, notGT recovery'}
        save(out/f'seed{seed}_raw_pipeline_sources.json',raw_pipeline_links)
        result['seeds'][str(seed)]=R
    result['status']='COMPLETED';result['elapsed_seconds']=time.perf_counter()-start
    result['harmful_correction_screen_supported_bothseeds']={tag:all(result['seeds'][str(s)]['comparisons'][tag]['delta_pp']['AP']>1e-10 for s in [1,2]) for tag in ['raw_fixed_cap_identity','raw_native_pipeline']}
    save(out/'results.json',result)
    save(out/'inputs.json',sources)
    save(out/'outputs.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps({s:{a:r['delta_pp'] for a,r in z['comparisons'].items()} for s,z in result['seeds'].items()}),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    try:main()
    except Exception:
        p=HERE/'LQE_effect_attempt01'/'FAILED.txt'
        p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc(),encoding='utf-8');raise
