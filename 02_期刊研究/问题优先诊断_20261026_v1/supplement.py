"""Read-only supplementary references and independent aggregate checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,gzip,hashlib,math,collections
from pathlib import Path
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
BASE=ROOT/'data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
S3=ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(n,j):
    with open(OUT/n,'x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
main=read(OUT/'results.json')
with open(OUT/'per_GT.csv',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
groups=main['groups'];imgs=read(OUT/'per_image.json')
for g in groups:
    if g['group_type']=='all':
        rr=[r for r in rows if int(r['seed'])==g['seed']]
        assert len(rr)==g['n_gt']
        for th in (50,75):
            c=collections.Counter(r[f'partition_{th}'] for r in rr)
            for k in 'GSKCMT':assert c[k]==g[k+str(th)]
archive=read(ROOT/'code/unit_instance_completion_20261021_v1/349_cpu_val_20261023_v1/349_评估_result.json')['metrics']
metric_diff={s:max(abs(main['metrics'][s]['stored_id_native300'][k]-archive[s]['baseline'][k]) for k in main['metrics'][s]['stored_id_native300']) for s in ('1','2')}
assert max(metric_diff.values())<1e-12
ema={};derived={}
for seed in (1,2):
    rr=[r for r in rows if int(r['seed'])==seed];folder=BASE/f'seed{seed}/run/endpoint_epoch24'
    z=np.load(folder/'GT_hits.npz',allow_pickle=False);aids=z['annotation_id'].tolist();hits=z['hits'];hitdict={int(a):hits[j] for j,a in enumerate(aids)}
    ep=read(folder/'summary.json');matched=json.load(gzip.open(folder/'matching_all_area.json.gz','rt',encoding='utf-8'))
    replays=np.zeros(hits.shape,dtype=bool);aix={a:j for j,a in enumerate(aids)};gt_seen=[]
    for e in matched:
        gt_seen.extend(e['gtIds'])
        for t in range(10):
            for a,ignore in zip(e['dtMatches'][t],e['dtIgnore'][t]):
                if a!=0 and not ignore:replays[aix[int(a)],t]=True
    assert np.array_equal(replays,hits) and sorted(gt_seen)==sorted(aids)
    emagr=[]
    for g in groups:
        if g['seed']!=seed or g['group_type'] not in ('all','short_side640','area','class','density32','occlusion'):continue
        typ=g['group_type'];name=g['group']
        def has(r):
            if typ=='all':return True
            if typ=='class':return r['category_id']==name
            if typ=='occlusion':return r['occlusion']==name
            if typ=='short_side640':
                x=float(r['short_side_640']);return ('[0,4)' if x<4 else '[4,8)' if x<8 else '[8,16)' if x<16 else '[16,32)' if x<32 else '[32,inf)')==name
            if typ=='area':
                x=float(r['area_original']);return ('small<1024' if x<1024 else 'medium[1024,9216)' if x<9216 else 'large>=9216')==name
            if typ=='density32':
                x=int(r['density32']);return ('0' if x==0 else '1-4' if x<5 else '5-14' if x<15 else '15+')==name
        selected=[r for r in rr if has(r)];assert len(selected)==g['n_gt']
        emagr.append({'seed':seed,'group_type':typ,'group':name,'n_gt':len(selected),
            'TP50_EMA':sum(bool(hitdict[int(r['annotation_id'])][0]) for r in selected),
            'TP75_EMA':sum(bool(hitdict[int(r['annotation_id'])][5]) for r in selected)})
    ema[str(seed)]={'captured_not_new_inference':True,'all_GT_matching_reconstructed_exact':True,
        'stored_ID0_sentinel_policy':True,'source_hash':sha(folder/'summary.json'),'metrics_percent':ep['overall_percent'],'groups':emagr}
    im=[i for i in imgs if i['seed']==seed]
    derived[str(seed)]={
        'both_raw_geom_below50':sum(float(r['raw_max_iou'])<.5 for r in rr),
        'raw_geom_between50_and75':sum(.5<=float(r['raw_max_iou'])<.75 for r in rr),
        'hit50_not75':sum(r['hit50']=='1' and r['hit75']=='0' for r in rr),
        'hit75_not50':sum(r['hit75']=='1' and r['hit50']=='0' for r in rr),
        'native_query_unique_range':[min(i['native_unique_queries'] for i in im),max(i['native_unique_queries'] for i in im)],
        'image_GT_range':[min(i['n_gt'] for i in im),max(i['n_gt'] for i in im)],
        'images_over300_GT':sum(i['n_gt']>300 for i in im),
        'native_maxmatching50':sum(i['maxmatch']['native_pairs_50'] for i in im),
        'cap100_maxmatching50':sum(i['maxmatch']['cap100_pairs_50'] for i in im),
        'native_maxmatching75':sum(i['maxmatch']['native_pairs_75'] for i in im),
        'cap100_maxmatching75':sum(i['maxmatch']['cap100_pairs_75'] for i in im),
        'K600_FP50':main['precheck'][str(seed)]['evaluated_rows_K600']-main['precheck'][str(seed)]['TP50_K600'],
        'native_FP50':main['precheck'][str(seed)]['evaluated_rows_native']-main['precheck'][str(seed)]['TP50_native']}
    # Descriptive matched-stratum comparison, with explicit overlap coverage.
    # Category x short-side bin x occlusion. No fitted model, no causal reading.
    strata=collections.defaultdict(lambda:[[],[]])
    for r in rr:
        d=int(r['density32'])
        if d!=0 and d<5:continue
        s=float(r['short_side_640']);sb=0 if s<4 else 1 if s<8 else 2 if s<16 else 3 if s<32 else 4
        strata[(r['category_id'],sb,r['occlusion'])][int(d>=5)].append(r)
    overlap=[v for v in strata.values() if v[0] and v[1]]
    total=sum(len(v[0])+len(v[1]) for v in overlap)
    stats=[]
    for idx in (0,1):
        value=sum((len(v[0])+len(v[1]))*sum(int(r['hit50']) for r in v[idx])/len(v[idx]) for v in overlap)/total
        stats.append(value)
    derived[str(seed)]['density_standardization']={'strata':'category x short-side5bins x original occlusion3levels',
        'reference':'density32==0','exposed':'density32>=5','common_strata':len(overlap),
        'reference_in_overlap':sum(len(v[0]) for v in overlap),'dense_in_overlap':sum(len(v[1]) for v in overlap),
        'target_weights':'combined GT counts within common strata','standardized_TP50_rate':stats,
        'interpretation':'association within coarse shared strata; residual confounding and image dependence remain'}
# Isolate borderline geometry conversions and truly independent FP64 scalar arithmetic.
crossings=[];max_scalar_error=0.;comparisons=0
for seed in (1,2):
    for p in sorted((S3/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        e=torch.load(p,map_location='cpu',weights_only=True);pb=e['final300_boxes'];gb=e['gt_boxes_640'].numpy().astype(float)
        x,y,w,h=(pb*640).unbind(-1);a=torch.stack((x-w/2,y-h/2,x+w/2,y+h/2),-1).numpy().astype(float)
        wh=np.maximum(np.minimum(a[:,None,2:],gb[None,:,2:])-np.maximum(a[:,None,:2],gb[None,:,:2]),0)
        inter=wh.prod(-1);aa=np.maximum(a[:,2:]-a[:,:2],0).prod(-1);ab=(gb[:,2:]-gb[:,:2]).prod(-1)
        mm=inter/np.maximum(aa[:,None]+ab[None,:]-inter,1e-300)
        if len(gb):
            q,j=map(int,np.unravel_index(mm.argmax(),mm.shape));v=list(map(float,a[q]));g=list(map(float,gb[j]))
            ix=max(0.,min(v[2],g[2])-max(v[0],g[0]));iy=max(0.,min(v[3],g[3])-max(v[1],g[1]));ia=ix*iy
            uni=max(0.,v[2]-v[0])*max(0.,v[3]-v[1])+(g[2]-g[0])*(g[3]-g[1])-ia
            sc=ia/max(uni,1e-300);max_scalar_error=max(max_scalar_error,abs(sc-mm[q,j]));comparisons+=1
        # Check only a narrow threshold band in original coordinates, preserving raw FP32 edges.
        qq,jj=np.where(np.abs(mm-.75)<2e-5)
        if len(qq):
            cx,cy,bw,bh=pb.unbind(-1);be=torch.stack((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),-1)*torch.tensor(e['orig_wh']*2)
            for q,j in zip(qq,jj):
                aa0=be[q].double().numpy();g0=main['scope'] # avoid reading unrelated dataset
                # Derive original GT directly from primary per-GT source metadata via val JSON below.
                aid=e['gt_annotation_ids'][j]
                if 'ann_lookup' not in globals():
                    ann=read(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json');ann_lookup={a['id']:a for a in ann['annotations']}
                b=ann_lookup[aid]['bbox'];gt=np.array([b[0],b[1],b[0]+b[2],b[1]+b[3]],dtype=float)
                iw=max(0.,min(aa0[2],gt[2])-max(aa0[0],gt[0]));ih=max(0.,min(aa0[3],gt[3])-max(aa0[1],gt[1]));ip=iw*ih
                ar=max(0.,aa0[2]-aa0[0])*max(0.,aa0[3]-aa0[1]);ag=(gt[2]-gt[0])*(gt[3]-gt[1]);mo=ip/(ar+ag-ip)
                if (mo>=.75)!=(mm[q,j]>=.75):crossings.append({'seed':seed,'image_id':e['image_id'],'query':int(q),'annotation_id':aid,'original_iou':mo,'cached640_iou':float(mm[q,j]),'original_raw_max':next(float(r['raw_max_iou']) for r in rows if int(r['seed'])==seed and int(r['annotation_id'])==aid)})
assert max_scalar_error<1e-12
history=[]
for n,scope in [
 ('357_研究证据账本_355b停止与后续边界.txt','全文；JSON entries读取候选/状态/边界摘要'),
 ('360_创新点整理_线程A_全量对照_20261026_v1.md','全文读取；其中文献与实验转述未冒充逐项原件复核'),
 ('360_线程B_宽度门槛诊断_20261026_v3.md','全文读取；本轮未重做模型前向'),
 ('360_Geo0七行_独立审查_20261026.txt','全文读取；主要口径勘误'),
 ('368_集合目标零训练预检_结论.txt','全文读取；未重跑合成量化实验'),
 ('369_问题优先诊断_寻找新创新点_Prompt_v1.txt','全文；本轮执行范围')]:
    p=ROOT/'持续寻找研究主线_20261020_v2'/n;history.append({'path':str(p),'sha256':sha(p),'read_scope':scope,'lines':len(p.read_text('utf-8').splitlines())})
p=ROOT/'data/00_实验计划与证据索引/查询类别槽竞争与廉价对照核验_20261007_v1/05_作者可审查的类别槽竞争裁决.txt'
history.append({'path':str(p),'sha256':sha(p),'read_scope':'全文；旧TRAIN16硬单类对照阴性，未重新执行','lines':len(p.read_text('utf-8').splitlines())})
plan=OUT/'followup_plan.json';(OUT/'followup_plan.sha256').write_text(sha(plan),encoding='utf-8')
write('supplementary_checks.json',{'CSV_all_GT_partitions_match':True,'archived12metrics_max_abs_diff':metric_diff,
    'EMA24_reference':ema,'derived_counts':derived,'independent_FP64_scalar':{'comparisons':comparisons,'max_abs_error':max_scalar_error},
    'geometry_75_crossings':crossings,'history_reading':history,'followup_plan_sha256':sha(plan),
    'remaining_limits':['No new EMA or complete96/132-epoch raw-query replay','No internal candidate/feature/attention trace','No independent blind reviewer','AP bootstrap not computed; count bootstrap only']})
print(json.dumps({'derived_counts':derived,'geometry_75_crossings':crossings,'archive_metric_diff':metric_diff,'EMA_all':[r for v in ema.values() for r in v['groups'] if r['group_type']=='all']},ensure_ascii=False))
