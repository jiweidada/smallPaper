"""Local all-GT diagnostic, immutable source inputs, no model construction/forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import sys, json, hashlib, time, csv, copy, contextlib, io, gzip
from pathlib import Path
from collections import Counter, defaultdict
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
WORK = ROOT/'code/unit_instance_completion_20261021_v1'
CACHE = WORK/'342_codex_stage3_20261022_v1'
DATA = ROOT/'data/00_实验计划与证据索引'
BASE = DATA/'本次24轮双基线模型下载_20261009_v1'
VAL = Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
TRAIN = Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-train/annotations/train.json')
EVALROOT = DATA/'单一候选评价依赖补齐与构建核验_20261008_v1/isolated_runtime'
sys.path.insert(0,str(EVALROOT))
from pycocoeval.coco import COCO
from pycocoeval.cocoeval import COCOeval
METRICS = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write(name,obj):
    with open(OUT/name,'x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def quant(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return {'n':len(x),'mean':float(x.mean()),'q':np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist()} if len(x) else {'n':0}

def iou_np(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    wh=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0)
    inter=wh.prod(-1)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(-1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-300)

def scalar_iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return ix*iy/max(aa+bb-ix*iy,1e-300)

def edges_norm(pb):
    x,y,w,h=pb.unbind(-1)
    return torch.stack((x-w/2,y-h/2,x+w/2,y+h/2),-1)

def cap_indices(labels,cap=100):
    count=Counter();keep=[]
    for j,c in enumerate(labels.tolist()):
        if count[c]<cap: keep.append(j);count[c]+=1
    return np.array(keep,dtype=np.int64)

def maxmatch(mask):
    if not mask.size:return 0
    return int((maximum_bipartite_matching(csr_matrix(mask),perm_type='column')>=0).sum())

def evaluate(ds,rows,name,cap=100):
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        gt=COCO();gt.dataset=copy.deepcopy(ds);gt.createIndex()
        dt=gt.loadRes(copy.deepcopy(rows))
        ev=COCOeval(gt,dt,'bbox',using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=cap)
        ev.params.imgIds=sorted(gt.getImgIds());ev.evaluate();ev.accumulate();ev.summarize()
    (OUT/(name+'_eval.txt')).write_text(log.getvalue(),encoding='utf-8')
    return ev,{k:float(v) for k,v in zip(METRICS,ev.stats)}

def hits_from_evaluator(ev,offset=1):
    hits={50:set(),75:set()};dtmatch={50:set(),75:set()}
    matched_ids={50:{},75:{}}
    for e in ev.evalImgs:
        if e is None or tuple(e['aRng'])!=tuple(ev.params.areaRng[0]):continue
        for th,index in ((50,0),(75,5)):
            for aid,did,ign in zip(e['gtIds'],e['gtMatches'][index],e['gtIgnore']):
                if did>0 and not ign:
                    hits[th].add(int(aid)-offset);matched_ids[th][int(aid)-offset]=int(did)
            for did,aid,ign in zip(e['dtIds'],e['dtMatches'][index],e['dtIgnore'][index]):
                if aid>0 and not ign:dtmatch[th].add(int(did))
    return hits,dtmatch,matched_ids

def main():
    start=time.perf_counter();protocol_sha=sha(OUT/'protocol.json')
    assert (OUT/'protocol.sha256').read_text('utf-8').strip()==protocol_sha
    sources={};input_rows=[]
    def track(p,kind):
        p=Path(p);h=sha(p);sources[str(p)]=h
        input_rows.append({'path':str(p),'kind':kind,'bytes':p.stat().st_size,'sha256':h})
        return h
    ap=VAL/'annotations/val.json';ann_hash=track(ap,'validation_annotation')
    assert ann_hash=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    ds=json.loads(ap.read_text('utf-8'));images=sorted(ds['images'],key=lambda z:z['id'])
    anns=ds['annotations'];aindex={a['id']:a for a in anns};by=defaultdict(list)
    for a in anns:by[a['image_id']].append(a)
    assert len(images)==548 and len(anns)==38759 and len(aindex)==len(anns)
    assert all(a['bbox'][2]>0 and a['bbox'][3]>0 and a['iscrowd']==0 for a in anns)
    train=json.loads(TRAIN.read_text('utf-8'));train_scope={'images':len(train['images']),'GT':len(train['annotations'])}
    track(TRAIN,'training_annotation_count_only');del train
    for p in [ROOT/'CLAUDE.md',ROOT/'code/fine_residual_screen24_20261009_v2/geometry.py',
              WORK/'stage1_cache_v3.py',CACHE/'342_build_val_v3.py',
              ROOT/'code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/deim/postprocessor.py',
              Path(r'C:/Users/jiwei/Desktop/DEIM/yolo2coco.py'),EVALROOT/'pycocoeval/coco.py',EVALROOT/'pycocoeval/cocoeval.py',
              EVALROOT/'pycocoeval/mask.py',EVALROOT/'pycocoeval/_mask.cp310-win_amd64.pyd']:
        track(p,'source')
    # Original annotation text is a separate universe, never silently inserted into converted COCO GT.
    raw_counts=Counter();raw_map={};raw_diffs=[];raw_mapping_failures=[]
    for im in images:
        path=VAL/'annotations'/Path(im['file_name']).with_suffix('.txt')
        track(path,'original_annotation_text')
        rows=[[float(v) for v in line.split(',') if v.strip()] for line in path.read_text('utf-8-sig').splitlines() if line.strip()]
        for r in rows:
            raw_counts['all_lines']+=1;raw_counts[f'class_{int(r[5])}']+=1
            raw_counts[f'score_{int(r[4])}']+=1
        valid=[r for r in rows if int(r[4])==1 and 1<=int(r[5])<=10 and r[2]>0 and r[3]>0]
        raw_counts['score1_class1to10_positive']+=len(valid)
        aa=by[im['id']]
        if len(valid)!=len(aa):raw_mapping_failures.append({'image':im['id'],'raw':len(valid),'converted':len(aa)})
        for cls in range(10):
            rr=[r for r in valid if int(r[5])-1==cls];gg=[a for a in aa if a['category_id']==cls]
            if not rr or not gg:continue
            cost=np.abs(np.array([r[:4] for r in rr])[:,None,:]-np.array([a['bbox'] for a in gg])[None,:,:]).sum(-1)
            ri,gi=linear_sum_assignment(cost)
            for i,j in zip(ri,gi):
                delta=float(np.max(np.abs(np.array(rr[i][:4])-np.array(gg[j]['bbox']))));raw_diffs.append(delta)
                if delta<=.02:raw_map[gg[j]['id']]={'occlusion':int(rr[i][7]),'truncation':int(rr[i][6]),'raw_xywh':rr[i][:4]}
                else:raw_mapping_failures.append({'annotation_id':gg[j]['id'],'max_abs_px':delta})
    checkpoints={}
    for seed in (1,2):
        cp=BASE/f'seed{seed}/run/last.pth';h=track(cp,'checkpoint_metadata_only')
        c=torch.load(cp,map_location='cpu',weights_only=True)
        assert c['last_epoch']==23
        cfg=BASE/f'seed{seed}/bundle/resolved.yml';track(cfg,'model_configuration')
        resolved=json.loads(cfg.read_text('utf-8'));state=c['model'];ema=c['ema']['module']
        floatkeys=[k for k in state if state[k].is_floating_point()]
        checkpoints[str(seed)]={'path':str(cp),'sha256':h,'last_epoch':c['last_epoch'],'active_state':'model',
            'ema_available':True,'model_ema_unequal_tensor_count':sum(not torch.equal(state[k],ema[k]) for k in floatkeys),
            'float_state_tensors':len(floatkeys),'recipe_epochs':resolved['epoches'],'seed':resolved['seed'],
            'decoder':resolved['DFINETransformer'],'backbone':resolved['HGNetv2'],'active_arm':resolved['ResidualGeometryDEIM']['arm']}
        del c,state,ema
    positive=copy.deepcopy(ds)
    for a in positive['annotations']:a['id']+=1
    records=[];image_records=[];stage_counts={};metrics={};proofs={'cached_fullp_replays':0,'cached_gt_replays':0,'raw_full_cache_bit_matches':0,
        'scalar_iou_pairs':0,'scalar_iou_max_error':0.,'geometry_fp32_fp64_crossings_50':0,'geometry_fp32_fp64_crossings_75':0}
    detection_records=[];capture_refs={};precheck={};seed_gt_rows={}
    for seed in (1,2):
        print(f'seed {seed}: reading all 548 original raw/complete cache pairs',flush=True)
        allrows={300:[],600:[]};stage=Counter();perseed=[];perimage={};image_arrays={}
        for pos,im in enumerate(images):
            iid=im['id'];rp=CACHE/f'342_raw_val_seed{seed}/342_{iid:06d}.pt';fp=CACHE/f'342_cache_val_v2_seed{seed}/342_{iid:06d}.pt'
            track(rp,'raw_cache');track(fp,'complete_cache')
            r=torch.load(rp,map_location='cpu',weights_only=True);e=torch.load(fp,map_location='cpu',weights_only=True)
            assert r['seed']==seed and r['image_id']==iid and r['checkpoint_sha256']==checkpoints[str(seed)]['sha256']
            assert r['annotation_sha256']==ann_hash and e['annotation_sha256']==ann_hash
            assert e['raw_cache_file_sha256']==sources[str(rp)]
            assert torch.equal(r['final300_boxes'],e['final300_boxes']) and torch.equal(r['final300_logits'],e['final300_logits'])
            proofs['raw_full_cache_bit_matches']+=1
            pb=r['final300_boxes'];logits=r['final300_logits'];scores=logits.sigmoid()
            assert pb.shape==(300,4) and logits.shape==(300,10) and torch.isfinite(pb).all() and torch.isfinite(logits).all()
            edge=edges_norm(pb);bb=(edge*torch.tensor([im['width'],im['height']]*2)).numpy()
            # Exact 342 640-plane operation order.
            x,y,w,h=(pb.float()*640).unbind(-1);plane=torch.stack((x-w/2,y-h/2,x+w/2,y+h/2),-1)
            aa=by[iid];gxywh=np.array([a['bbox'] for a in aa],dtype=np.float64).reshape(-1,4)
            gt=gxywh.copy();gt[:,2:]+=gt[:,:2]
            gl=np.array([a['category_id'] for a in aa],dtype=np.int64);gids=[a['id'] for a in aa]
            g32=torch.tensor(gxywh,dtype=torch.float32)
            xx,yy,ww,hh=g32.unbind(-1)
            gplane=torch.stack((xx*640/im['width'],yy*640/im['height'],(xx+ww)*640/im['width'],(yy+hh)*640/im['height']),-1)
            assert e['gt_annotation_ids']==gids and torch.equal(e['gt_labels'],torch.tensor(gl))
            # The original builder multiplies by 640/orig_dimension; use that operation order for parity.
            gplane2=torch.stack((xx*(640/im['width']),yy*(640/im['height']),(xx+ww)*(640/im['width']),(yy+hh)*(640/im['height'])),-1)
            assert torch.equal(gplane2,e['gt_boxes_640']);proofs['cached_gt_replays']+=1
            nativev,nativeflat=scores.flatten().topk(300)
            nk=nativev>.001;flat=nativeflat[nk];nv=nativev[nk];q=(flat//10);cl=(flat%10)
            assert torch.equal(flat,e['fullp_flatten_index']) and torch.equal(nv,e['fullp_scores']) and torch.equal(q,e['fullp_query_index'])
            assert torch.equal(plane[q],e['fullp_boxes']);proofs['cached_fullp_replays']+=1
            M=iou_np(bb,gt);Mplane=iou_np(plane.numpy(),e['gt_boxes_640'].numpy())
            for th,key in ((.5,'geometry_fp32_fp64_crossings_50'),(.75,'geometry_fp32_fp64_crossings_75')):
                proofs[key]+=int(((M>=th)!=(Mplane>=th)).sum())
            if len(gt):
                for qi,gj in ((0,0),(299,len(gt)-1),(int(M.argmax()//len(gt)),int(M.argmax()%len(gt)))):
                    err=abs(scalar_iou(bb[qi],gt[gj])-M[qi,gj]);proofs['scalar_iou_max_error']=max(proofs['scalar_iou_max_error'],err);proofs['scalar_iou_pairs']+=1
            valid=(pb[:,2:]>0).all(-1).numpy();stage['raw_queries']+=300;stage['raw_invalid_queries']+=int((~valid).sum())
            stage['all_query_class_pairs']+=3000;stage['native_top300_before_score']+=300
            stage['native_score_dropped_pairs']+=int((~nk).sum());stage['native_post_score_pairs']+=len(q)
            stage['native_unique_queries']+=len(set(q.tolist()));stage['native_extra_pair_slots_same_query']+=len(q)-len(set(q.tolist()))
            short=(e['gt_boxes_640'][:,2:]-e['gt_boxes_640'][:,:2]).min(-1).values.numpy()
            centers=(e['gt_boxes_640'][:,2:]+e['gt_boxes_640'][:,:2]).numpy()/2
            density=((centers[:,None,:]-centers[None,:,:])**2).sum(-1)<=32**2
            dens=density.sum(-1)-1
            rawmax=M.max(0);argmax=scores.argmax(-1).numpy();s=scores.numpy()
            gt_scores=s[:,gl]
            argmaxmax=np.where(argmax[:,None]==gl[None,:],M,0).max(0)
            floormax=np.where(gt_scores>.001,M,0).max(0)
            rank=np.zeros(3000,dtype=np.int64);globalorder=scores.flatten().argsort(descending=True,stable=True).numpy();rank[globalorder]=np.arange(1,3001)
            nativeM=M[q.numpy()];nativelab=cl.numpy();capix=cap_indices(nativelab)
            capM=nativeM[capix];caplab=nativelab[capix]
            stage['cap100_pairs']+=len(capix);stage['cap100_dropped_pairs']+=len(q)-len(capix)
            topmax=np.where(nativelab[:,None]==gl[None,:],nativeM,0).max(0)
            capmax=np.where(caplab[:,None]==gl[None,:],capM,0).max(0)
            max_matches={}
            for th in (.5,.75):
                suffix=str(int(th*100))
                max_matches['raw_any_'+suffix]=maxmatch(M>=th)
                max_matches['native_pairs_'+suffix]=maxmatch((nativeM>=th)&(nativelab[:,None]==gl[None,:]))
                max_matches['cap100_pairs_'+suffix]=maxmatch((capM>=th)&(caplab[:,None]==gl[None,:]))
            groupcounts={'image_id':iid,'seed':seed,'n_gt':len(gt),'raw_invalid':int((~valid).sum()),'native_unique_queries':len(set(q.tolist())),
                'native_pair_n':len(q),'cap100_pair_n':len(capix),'maxmatch':max_matches}
            perimage[iid]=groupcounts
            for K in (300,600):
                v,f=scores.flatten().topk(K);keep=v>.001;v=v[keep];f=f[keep];qs=(f//10).numpy();labs=(f%10).numpy()
                loc=[]
                for query,c,score in zip(qs,labs,v.tolist()):
                    b=torch.tensor(bb[query]);xywh=torch.cat((b[:2],b[2:]-b[:2])).tolist()
                    loc.append({'image_id':iid,'category_id':int(c),'bbox':xywh,'score':score})
                allrows[K].extend(loc)
            image_arrays[iid]={'M':M,'native_q':q.numpy(),'native_label':nativelab,'capix':capix,'native_scores':nv.numpy(),'gt_labels':gl,'gids':gids}
            for j,a in enumerate(aa):
                qualifying50=M[:,j]>=.5
                qualscore=s[qualifying50,int(gl[j])]
                ranks50=rank[np.where(qualifying50)[0]*10+gl[j]]
                row={'seed':seed,'image_id':iid,'annotation_id':a['id'],'category_id':a['category_id'],
                    'short_side_640':float(short[j]),'area_original':a['area'],'density32':int(dens[j]),'image_n_gt':len(gt),
                    'occlusion':raw_map.get(a['id'],{}).get('occlusion',-1),'truncation':raw_map.get(a['id'],{}).get('truncation',-1),
                    'raw_max_iou':float(rawmax[j]),'argmax_class_max_iou':float(argmaxmax[j]),'gtclass_floor_max_iou':float(floormax[j]),
                    'top300_same_max_iou':float(topmax[j]),'cap100_same_max_iou':float(capmax[j]),
                    'best_gtclass_score_qualifying50':float(qualscore.max()) if len(qualscore) else None,
                    'best_global_pair_rank_qualifying50':int(ranks50.min()) if len(ranks50) else None}
                for th,suffix in ((.5,'50'),(.75,'75')):
                    row['partition_'+suffix]=('G' if rawmax[j]<th else 'S' if floormax[j]<th else 'K' if topmax[j]<th else 'C' if capmax[j]<th else 'pending')
                perseed.append(row)
            if (pos+1)%100==0:print(f'seed {seed}: {pos+1}/548 cache pairs validated',flush=True)
        # Replay archived native output: cap100 has no effect on evaluated maxDets100 metrics.
        old=WORK/f'349_cpu_val_20261023_v1/349_seed{seed}_baseline_predictions.json';track(old,'archived_baseline_predictions')
        oldrows=json.loads(old.read_text('utf-8'));newrows=[]
        for im in images:
            loc=[r for r in allrows[300] if r['image_id']==im['id']]
            cap=cap_indices(np.array([r['category_id'] for r in loc]))
            newrows.extend(loc[j] for j in cap)
        prediction_comparison={'archived_n':len(oldrows),'replay_n':len(newrows),'same_row_metadata':len(oldrows)==len(newrows),'max_bbox_abs_error':0.,'max_score_abs_error':0.}
        assert len(oldrows)==len(newrows)
        for a,b in zip(oldrows,newrows):
            assert (a['image_id'],a['category_id'])==(b['image_id'],b['category_id'])
            prediction_comparison['max_bbox_abs_error']=max(prediction_comparison['max_bbox_abs_error'],max(abs(x-y) for x,y in zip(a['bbox'],b['bbox'])))
            prediction_comparison['max_score_abs_error']=max(prediction_comparison['max_score_abs_error'],abs(a['score']-b['score']))
        proofs[f'seed{seed}_archived_prediction_comparison']=prediction_comparison
        # These should be bit-identical, because native FP32 coordinate operation order is retained.
        assert prediction_comparison['max_bbox_abs_error']==0 and prediction_comparison['max_score_abs_error']==0
        evstored,stored=evaluate(ds,allrows[300],f'seed{seed}_stored_ids_native300')
        ev300,m300=evaluate(positive,allrows[300],f'seed{seed}_positive_ids_native300')
        ev600,m600=evaluate(positive,allrows[600],f'seed{seed}_positive_ids_top600')
        hits300,dt300,matched300=hits_from_evaluator(ev300)
        hits600,dt600,_=hits_from_evaluator(ev600)
        metrics[str(seed)]={'stored_id_native300':stored,'positive_id_native300':m300,'positive_id_top600':m600,
            'id_shift_delta_pp':{k:100*(m300[k]-stored[k]) for k in METRICS},'K600_delta_pp':{k:100*(m600[k]-m300[k]) for k in METRICS}}
        # Read matched EMA reference, without conflating its geometry with raw-cache geometry.
        endpoint=BASE/f'seed{seed}/run/endpoint_epoch24'
        for n in ('summary.json','GT_hits.npz','matching_all_area.json.gz','manifest.json'):track(endpoint/n,'captured_EMA24_reference')
        ep=json.loads((endpoint/'summary.json').read_text('utf-8'))
        z=np.load(endpoint/'GT_hits.npz',allow_pickle=False)
        capture_refs[str(seed)]={'summary':ep,'GT_hits_schema':{k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}}
        # Evaluator detection IDs follow the prediction list (starting at1).
        eval_kept_dids=set()
        for e in ev300.evalImgs:
            if e is not None and tuple(e['aRng'])==tuple(ev300.params.areaRng[0]):eval_kept_dids.update(e['dtIds'])
        dids_by_iid=defaultdict(list)
        for did,r in enumerate(allrows[300],1):
            if did in eval_kept_dids:dids_by_iid[r['image_id']].append((did,r))
        for iid,info in image_arrays.items():
            qs=info['native_q'];labs=info['native_label'];M=info['M'];gl=info['gt_labels'];capix=info['capix']
            loc=dids_by_iid[iid];assert len(loc)==len(capix)
            for j,(did,r) in zip(capix,loc):
                assert r['category_id']==int(labs[j])
                q=qs[j];same=M[q,gl==labs[j]];same_max=float(same.max()) if len(same) else 0.;anymax=float(M[q].max())
                kind='TP' if did in dt300[50] else 'same_class_overlap50_unmatched' if same_max>=.5 else 'other_class_overlap50' if anymax>=.5 else 'same_class_overlap10to50' if same_max>=.1 else 'other_overlap10to50' if anymax>=.1 else 'all_overlap_below10'
                detection_records.append({'seed':seed,'image_id':iid,'query':int(q),'class':int(labs[j]),'score':r['score'],'max_iou':anymax,'max_same_iou':same_max,'FP_map50':kind})
        for row in perseed:
            aid=row['annotation_id']
            for th in (50,75):
                hit=int(aid in hits300[th]);row[f'hit{th}']=hit;row[f'K600_hit{th}']=int(aid in hits600[th])
                p=f'partition_{th}'
                if row[p]=='pending':row[p]='T' if hit else 'M'
                assert not hit or row[p]=='T'
        for iid,r in perimage.items():
            local=[r0 for r0 in perseed if r0['image_id']==iid]
            for th in (50,75):
                for k in ('hit','K600_hit'):r[f'{k}{th}']=sum(a[f'{k}{th}'] for a in local)
            image_records.append(r)
        stage_counts[str(seed)]=dict(stage);records.extend(perseed);seed_gt_rows[seed]=perseed
        precheck[str(seed)]={'delta_pp':metrics[str(seed)]['K600_delta_pp'],
            'TP50_native':len(hits300[50]),'TP50_K600':len(hits600[50]),'TP75_native':len(hits300[75]),'TP75_K600':len(hits600[75]),
            'new_hit50':len(hits600[50]-hits300[50]),'lost_hit50':len(hits300[50]-hits600[50]),
            'new_hit75':len(hits600[75]-hits300[75]),'lost_hit75':len(hits300[75]-hits600[75]),
            'exported_rows_native':len(allrows[300]),'exported_rows_K600':len(allrows[600]),
            'evaluated_rows_native':len(eval_kept_dids),'evaluated_rows_K600':sum(len(e['dtIds']) for e in ev600.evalImgs if e is not None and tuple(e['aRng'])==tuple(ev600.params.areaRng[0])),
            'joint_support':all(m600[k]-m300[k]>1e-6 for k in ('AP','APs')) and len(hits600[50])>len(hits300[50])}
        print(f'seed{seed}: stages {dict(stage)}; metrics {metrics[str(seed)]}; precheck {precheck[str(seed)]}',flush=True)
        write(f'seed{seed}_stage_done.json',{'stage_counts':dict(stage),'metrics':metrics[str(seed)],'precheck':precheck[str(seed)]})
        del image_arrays,allrows,evstored,ev300,ev600
    # Aggregate only after the complete population has been processed.
    groupmap=defaultdict(list)
    for r in records:
        short=r['short_side_640'];area=r['area_original'];d=r['density32'];n=r['image_n_gt']
        size='[0,4)' if short<4 else '[4,8)' if short<8 else '[8,16)' if short<16 else '[16,32)' if short<32 else '[32,inf)'
        ar='small<1024' if area<1024 else 'medium[1024,9216)' if area<9216 else 'large>=9216'
        den='0' if d==0 else '1-4' if d<5 else '5-14' if d<15 else '15+'
        ni='<=50' if n<=50 else '51-100' if n<=100 else '101-200' if n<=200 else '>200'
        for typ,label in [('all','all'),('short_side640',size),('area',ar),('class',str(r['category_id'])),('density32',den),('image_GT',ni),('occlusion',str(r['occlusion'])),('truncation',str(r['truncation']))]:
            groupmap[r['seed'],typ,label].append(r)
    groups=[]
    for (seed,typ,label),rr in groupmap.items():
        row={'seed':seed,'group_type':typ,'group':label,'n_gt':len(rr)}
        for th in (50,75):
            c=Counter(r[f'partition_{th}'] for r in rr)
            for k in 'GSKCMT':row[f'{k}{th}']=c[k]
            assert sum(c.values())==len(rr)
            row[f'K600_T{th}']=sum(r[f'K600_hit{th}'] for r in rr)
            row[f'argmax_coverage{th}']=sum(r['argmax_class_max_iou']>=th/100 for r in rr)
        groups.append(row)
    def csvout(name,rows):
        with open(OUT/name,'x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvout('per_GT.csv',records);csvout('failure_map_by_group.csv',groups);csvout('per_detection.csv',detection_records)
    write('per_image.json',image_records)
    paired={}
    for th in (50,75):
        aa={r['annotation_id']:r for r in seed_gt_rows[1]};bb={r['annotation_id']:r for r in seed_gt_rows[2]}
        paired[str(th)]={'both_missed':sum(not aa[i][f'hit{th}'] and not bb[i][f'hit{th}'] for i in aa),
            'both_hit':sum(aa[i][f'hit{th}'] and bb[i][f'hit{th}'] for i in aa),
            'seed1_only_hit':sum(aa[i][f'hit{th}'] and not bb[i][f'hit{th}'] for i in aa),
            'seed2_only_hit':sum(not aa[i][f'hit{th}'] and bb[i][f'hit{th}'] for i in aa),
            'partition_cross_table':dict(Counter(aa[i][f'partition_{th}']+'|'+bb[i][f'partition_{th}'] for i in aa))}
    fpmap={}
    for seed in (1,2):
        rr=[r for r in detection_records if r['seed']==seed]
        fpmap[str(seed)]={}
        for thr in (0,.05,.1,.3,.5):
            group=[r for r in rr if r['score']>thr]
            fpmap[str(seed)][str(thr)]={'n':len(group),'counts':dict(Counter(r['FP_map50'] for r in group))}
    bootstrap={};rng=np.random.default_rng(20261026);ix=rng.integers(0,548,(1000,548))
    for seed in (1,2):
        rr=sorted([r for r in image_records if r['seed']==seed],key=lambda r:r['image_id']);n=np.array([r['n_gt'] for r in rr])
        bootstrap[str(seed)]={}
        for th in (50,75):
            delta=np.array([r[f'K600_hit{th}']-r[f'hit{th}'] for r in rr])
            boot=delta[ix].sum(1)/n[ix].sum(1)
            bootstrap[str(seed)][f'TP{th}_delta_rate_percentile95_pp']=(100*np.quantile(boot,[.025,.975])).tolist()
    distributions={str(seed):{field:quant([r[field] for r in seed_gt_rows[seed] if r[field] is not None]) for field in ['short_side_640','area_original','density32','raw_max_iou','argmax_class_max_iou','gtclass_floor_max_iou','top300_same_max_iou','cap100_same_max_iou','best_gtclass_score_qualifying50','best_global_pair_rank_qualifying50']} for seed in (1,2)}
    scope={'validation_images':len(images),'validation_GT':len(anns),'training_context_counts':train_scope,
        'original_text_counts':dict(raw_counts),'raw_to_COCO_mapping':{'matched':len(raw_map),'max_bbox_abs_diff_px':max(raw_diffs),'failures':raw_mapping_failures},
        'converted_ignore_counts':dict(Counter(str((a.get('iscrowd'),a.get('ignore'))) for a in anns)),
        'checkpoints':checkpoints,'evaluator':{'type':'project bound custom pycocoeval','maxDets':[1,10,100],'using_tiny_metrice':False,'using_vtiny_metrice':False,
        'cross_platform_binary':'Windows .pyd, not bit-certified equal to recorded Linux .so'},
        'runtime':{'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__},
        'operations':{'checkpoint_reads':2,'model_builds':0,'model_forwards':0,'optimizer_steps':0,'test_reads':0,'GPU':False,'network':False}}
    write('scope_audit.json',scope);write('capture_EMA24_reference.json',capture_refs)
    write('results.json',{'scope':scope,'stage_counts':stage_counts,'metrics':metrics,'groups':groups,'seed_pairing':paired,'FP_map':fpmap,
        'distributions':distributions,'precheck':precheck,'count_bootstrap':bootstrap,'verification':proofs,'elapsed_seconds':time.perf_counter()-start,
        'protocol_sha256':protocol_sha,'protocol_unchanged':sha(OUT/'protocol.json')==protocol_sha})
    write('input_manifest.json',{'entries':input_rows})
    # Rehash every input at the end: neither data nor historical outputs may be modified.
    bad=[str(p) for p,h in sources.items() if sha(p)!=h]
    assert not bad
    write('verification.json',{'input_files_rehashed':len(sources),'changed_files':bad,'proofs':proofs,'protocol_sha256':protocol_sha,
        'script_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-start,'operations':scope['operations']})
    print('COMPLETE',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
