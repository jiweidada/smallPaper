"""Frozen observational screen on existing37cases; no new model/feature response."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A17_新编码器机制与多实例映射审计_v1'
BASE=ROOT/'A16_实际特征空间变化审计_v1'
SOURCE=Path('D:/数据集/visdrone2019/VisDrone2019-DET-val')
assert not (OUT/'cached_cell_protocol.json').exists()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[BASE/(cohort+'_patches')/('case%02d.npz'%i) for cohort,n in [('old',12),('new',25)] for i in range(n)]
protocol={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_COLLISION_ERROR_ASSOCIATION',
    'question':'At matched annotation size/class/occlusion/truncation/local density,do tiny GT sharing native stride4 cells with a same-class GT show excess assigned geometric misses?',
    'data':'All37existingA12/A15/A16officialvalcases;853tinyGT. Geometry/nativeoutcomespreviouslyseen;collision-associationnotyetcalculated. Not blind or independent-data test.',
    'source_commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861','state_digest':'82c3ba49d69370a3d0105e47fc0ab5d715a9e7b1ad15feb275b1b497b5b0ccce',
    'exposure':'Stride4phase0=floor(cx/4),floor(cy/4)after800square. Exposed:cellhasanotherclass-identicalvalidGTcentre. Control:doesnot. GTcanbeanysize;targetsqrtarea<8.',
    'pairs':'Sameimage,GTclass,annotationocclusion,truncation,localdensitybin;GTsqrtsizeratio<=1.5. Density=countofotherallclassGTcentreswithin16inputpixels;bin=floor(log2(count+1)). Usealladmissibleexposed/controlpairs.',
    'primary':'Native author Hungarian assigned-box geometric failure IoU<.5. Mean exposed-minus-control failure within eachimage,then equal image mean. Not class-awareAP or unassigned all-query recall.',
    'why_no_initial_box_matching':'Encoder/proposalgeometrycanmediatethehypothesizedeffect;conditioningonitwouldansweranotherquestion.',
    'secondary':'Same pairs meanIoU andIoU75 failure descriptive;no rescue by secondary/other phase.',
    'bootstrap_seed':20261030171,'resamples':2000,
    'minimum_data':{'images_with_pairs':8,'pairs':60,'distinct_exposed':15,'distinct_controls':30},
    'resource_gate':'Minimumdataandprimarymean>=.10andbootstraplower>0. Positiveonlyprioritizescausalsource-specificdiagnosis,notnewalgorithmormainline.',
    'stop':'Onlythese37cases,phase0stride4andfixedcontrols. Inadequate=>inconclusive;nonewcase/threshold/controlrelaxationinthisturn.',
    'new_complete_model_forwards':0,'new_suffix_forwards':0,'new_feature_response_calls':0,'new_fits':0,
    'mainline_validated':False,'code_sha256':sha(__file__),
    'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in inputs]}
write('cached_cell_protocol.json',protocol)
rows=[];pairs=[];images=[];annotation_inputs=[]
known=json.loads((BASE/'per_GT_response.json').read_text(encoding='utf-8'))
known={(r['case'],r['gt']):r for r in known}
for cohort,n in [('old',12),('new',25)]:
    for index in range(n):
        case=cohort+('%02d'%index);path=BASE/(cohort+'_patches')/('case%02d.npz'%index)
        with np.load(path,allow_pickle=False) as z:
            gt=z['gt_boxes800'].astype(np.float64);labels=z['gt_labels'];sizes=z['gt_sizes800'];uids=z['gt_uids'];gs=z['gt_indices'];qs=z['query_indices']
            filename=str(z['image_filename'].item());prefix=str(z['prefix'].item())
        anno=SOURCE/'annotations'/(Path(filename).stem+'.txt');raw=anno.read_bytes()
        source_rows=raw.decode('utf-8-sig').splitlines()
        annotation_inputs.append({'file':anno.name,'sha256':hashlib.sha256(raw).hexdigest()})
        centres=(gt[:,:2]+gt[:,2:])*.5;groups=defaultdict(list)
        for g,(cx,cy) in enumerate(centres):groups[math.floor(cx/4),math.floor(cy/4)].append(g)
        rr=[]
        for q,g in zip(qs,gs):
            assert sizes[g]<8
            uid=str(uids[g]);lineid=int(uid.rsplit('_',1)[1]);source=[float(x) for x in source_rows[lineid].split(',') if x.strip()]
            assert int(source[5])==int(labels[g]) and source[4]>0
            cx,cy=centres[g];members=groups[math.floor(cx/4),math.floor(cy/4)]
            exposed=any(j!=g and labels[j]==labels[g] for j in members)
            distance=((centres-centres[g])**2).sum(-1)
            density=int(np.count_nonzero(distance<=256)-1)
            original=known[case,int(g)]
            assert original['query']==int(q) and abs(original['size']-float(sizes[g]))<1e-10
            r={'case':case,'cohort':cohort,'prefix':prefix,'gt':int(g),'query':int(q),'label':int(labels[g]),
                'size':float(sizes[g]),'occlusion':int(source[6]),'truncation':int(source[7]),
                'neighbours_within16':density,'density_bin':int(math.floor(math.log2(density+1))),
                'same_class_same_cell':bool(exposed),'any_class_same_cell':len(members)>1,
                'native_iou':original['native_iou'],'failure50':float(original['native_iou']<.5),
                'failure75':float(original['native_iou']<.75)}
            rr.append(r)
        pp=[]
        for exp in [r for r in rr if r['same_class_same_cell']]:
            for control in [r for r in rr if not r['same_class_same_cell']]:
                if any(exp[k]!=control[k] for k in ['label','occlusion','truncation','density_bin']):continue
                if max(exp['size'],control['size'])/min(exp['size'],control['size'])>1.5:continue
                pp.append({'case':case,'exposed_gt':exp['gt'],'control_gt':control['gt'],
                    'failure50_difference':exp['failure50']-control['failure50'],
                    'failure75_difference':exp['failure75']-control['failure75'],
                    'meanIoU_difference':exp['native_iou']-control['native_iou']})
        image={'case':case,'prefix':prefix,'tiny_GT':len(rr),'exposed':sum(r['same_class_same_cell'] for r in rr),
            'any_exposed':sum(r['any_class_same_cell'] for r in rr),'pairs':len(pp),
            'effects':{k:float(np.mean([x[k] for x in pp])) if pp else None for k in ['failure50_difference','failure75_difference','meanIoU_difference']}}
        rows+=rr;pairs+=pp;images.append(image)
assert len(rows)==853 and len({i['prefix'] for i in images})==37
counts={'images_with_pairs':sum(i['pairs']>0 for i in images),'pairs':len(pairs),
    'distinct_exposed':len({(p['case'],p['exposed_gt']) for p in pairs}),
    'distinct_controls':len({(p['case'],p['control_gt']) for p in pairs})}
stats={}
for key in ['failure50_difference','failure75_difference','meanIoU_difference']:
    a=np.asarray([i['effects'][key] for i in images if i['pairs']]);rng=np.random.default_rng(protocol['bootstrap_seed'])
    if len(a):
        b=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(1)
        stats[key]={'mean':float(a.mean()),'low':float(np.quantile(b,.025)),'high':float(np.quantile(b,.975))}
    else:stats[key]={'mean':None,'low':None,'high':None}
enough=all(counts[k]>=n for k,n in protocol['minimum_data'].items())
primary=stats['failure50_difference'];passed=enough and primary['mean']>=.10 and primary['low']>0
decision=('WORTH_SOURCE_SPECIFIC_CAUSAL_MULTIPLICITY_DIAGNOSIS_ONLY' if passed else
          'INCONCLUSIVE_TOO_FEW_MATCHED_CELL_COINCIDENCES_NO_AUTOMATIC_EXTENSION' if not enough else
          'NO_SUFFICIENT_CONDITIONAL_EXCESS_MISS_SIGNAL_DEPRIORITIZE_FIXED_CELL_MAINLINE')
summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_CACHED_CELL_COINCIDENCE_ASSOCIATION',
    'images':37,'tiny_GT':len(rows),'same_class_exposed_GT':sum(r['same_class_same_cell'] for r in rows),
    'any_class_exposed_GT':sum(r['any_class_same_cell'] for r in rows),'counts':counts,'statistics':stats,
    'adequate_matched_cohort':bool(enough),'gate_passed':bool(passed),'decision':decision,
    'new_complete_model_forwards':0,'new_suffix_forwards':0,'new_feature_response_calls':0,'new_fits':0,
    'main_model_updates':0,'mainline_validated':False,
    'scope':'Observational association on previously seen fixed-checkpoint cases. Not causal encoding failure,hard capacity limit or deployable improvement.'}
write('cached_cell_per_GT.json',rows);write('cached_cell_pairs.json',pairs);write('cached_cell_per_image.json',images)
write('cached_cell_annotation_inputs.json',annotation_inputs);write('cached_cell_summary.json',summary)
print(json.dumps(summary))
