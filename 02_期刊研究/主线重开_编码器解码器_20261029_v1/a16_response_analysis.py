"""Prespecified, image-clustered association analysis; no detector or training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from a16_spatial_operator import response_metrics

def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def paired_iou(box,gt):
    b=np.asarray(box,dtype=np.float64);g=np.asarray(gt,dtype=np.float64)
    b=np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)*800
    wh=np.maximum(0,np.minimum(b[...,2:],g[...,2:])-np.maximum(b[...,:2],g[...,:2]))
    inter=wh[...,0]*wh[...,1];bw=np.maximum(0,b[...,2:]-b[...,:2]);gw=np.maximum(0,g[...,2:]-g[...,:2])
    return inter/np.maximum(bw[...,0]*bw[...,1]+gw[...,0]*gw[...,1]-inter,1e-30)
def geometry(data,case,cohort):
    qs=data['query_indices'];gs=data['gt_indices'];gt=data['gt_boxes800']
    final=paired_iou(data['raw_boxes'][qs],gt[gs]);prior=paired_iou(data['reference'][:,0],gt[gs])
    rows=[]
    for i,(q,g,f,b,wh) in enumerate(zip(qs,gs,final,prior,data['reference'][:,0,2:]*800)):
        rows.append({'case':case,'cohort':cohort,'row':i,'gt':int(g),'query':int(q),
            'label':int(data['gt_labels'][g]),'size':float(data['gt_sizes800'][g]),
            'native_iou':float(f),'input_reference_iou':float(b),'input_reference_positive':bool(np.all(wh>0)),
            'input_reference_size':float(np.sqrt(max(0,wh.prod()))),'input_reference_iou_bin':min(3,int(b*4)),
            'good':bool(f>=.75)})
    pairs=[]
    for good in [r for r in rows if r['good'] and r['input_reference_positive']]:
        for bad in [r for r in rows if not r['good'] and r['input_reference_positive']]:
            if good['label']!=bad['label'] or good['input_reference_iou_bin']!=bad['input_reference_iou_bin']:continue
            if max(good['size'],bad['size'])/min(good['size'],bad['size'])>1.5:continue
            if max(good['input_reference_size'],bad['input_reference_size'])/min(good['input_reference_size'],bad['input_reference_size'])>1.5:continue
            pairs.append({'case':case,'cohort':cohort,'good_gt':good['gt'],'bad_gt':bad['gt'],
                'good_query':good['query'],'bad_query':bad['query'],'good_row':good['row'],'bad_row':bad['row']})
    return rows,pairs
def boot(values,seed):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return {'mean':None,'low':None,'high':None,'images':0}
    rng=np.random.default_rng(seed);samples=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(1)
    return {'mean':float(a.mean()),'low':float(np.quantile(samples,.025)),
            'high':float(np.quantile(samples,.975)),'images':len(a)}
def main(root):
    root=Path(root);assert not (root/'analysis_summary.json').exists()
    p=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    capture=json.loads((root/'capture_summary.json').read_text(encoding='utf-8'))
    assert capture['new_full_model_forwards']==26 and capture['main_model_updates']==0
    for phase in ['before_capture','after_capture']:
        assert json.loads((root/('source_validation_'+phase+'.json')).read_text())['status']=='PASS_SOURCE_CORE_CPU_DOUBLE_AGREEMENT'
    old_pairs=json.loads((root/'geometry_pairs.json').read_text(encoding='utf-8'))
    rows=[];pairs=[];images=[];files=[]
    metricdir=root/'response_matrices';metricdir.mkdir()
    fields=['cancellation','within_level_cancellation','across_level_cancellation','trace_stack','trace_level','trace_pool']
    for cohort,count in [('old',12),('new',25)]:
        for index in range(count):
            path=root/(cohort+'_patches')/('case%02d.npz'%index)
            with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
            key=cohort+('%02d'%index);rr,pp=geometry(data,key,cohort)
            if cohort=='old':
                keys=['good_gt','bad_gt','good_query','bad_query']
                assert [{k:x[k] for k in keys} for x in pp]==[{k:x[k] for k in keys} for x in old_pairs if x['case']==index]
            result=response_metrics(data,p['epsilon_input_pixels'])
            np.savez_compressed(metricdir/(key+'.npz'),**result)
            for row in rr:
                for field in fields:row[field]=float(result[field][row['row']])
                row['zero_response']=bool(result['zero_response'][row['row']])
            effects={field:float(np.mean([rr[x['bad_row']][field]-rr[x['good_row']][field] for x in pp])) if pp else None for field in fields}
            image={'case':key,'cohort':cohort,'prefix':str(data['prefix'].item()),'tiny_GT':int((data['gt_sizes800']<8).sum()),
                'matched_tiny_GT':len(rr),'pairs':len(pp),'distinct_good':len({x['good_gt'] for x in pp}),
                'distinct_bad':len({x['bad_gt'] for x in pp}),'effects':effects}
            rows+=rr;pairs+=pp;images.append(image)
            files.append({'file':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len({i['prefix'] for i in images})==37
    stats={};decisions={}
    for cohort in ['all','new']:
        ii=[i for i in images if cohort=='all' or i['cohort']==cohort]
        pp=[x for x in pairs if cohort=='all' or x['cohort']==cohort]
        limits=p['minimum_data'][cohort]
        counts={'images_with_pairs':sum(i['pairs']>0 for i in ii),'pairs':len(pp),
            'distinct_good':len({(x['case'],x['good_gt']) for x in pp}),
            'distinct_bad':len({(x['case'],x['bad_gt']) for x in pp})}
        enough=all(counts[k]>=limits[k] for k in limits)
        ss={field:boot([i['effects'][field] for i in ii if i['pairs']],p['bootstrap_seed']+(cohort=='new')) for field in fields}
        stat=ss['cancellation'];passed=enough and stat['mean']>=.10 and stat['low']>0
        stats[cohort]={'counts':counts,'minimum_data':limits,'enough_data':bool(enough),'statistics':ss,'gate_passed':bool(passed)}
    passed=stats['all']['gate_passed'] and stats['new']['gate_passed']
    enough=stats['all']['enough_data'] and stats['new']['enough_data']
    decision=('ASSOCIATION_SUPPORTS_NEXT_CAUSAL_OPERATOR_COMPARISON_ONLY' if passed else
              'INCONCLUSIVE_MATCHED_COHORT_TOO_SMALL_NO_AUTOMATIC_EXTENSION' if not enough else
              'NO_SUFFICIENT_ERROR_SPECIFIC_CANCELLATION_SIGNAL_NO_MODULE_TRAINING')
    summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_FIXED_FEATURE_RESPONSE_ASSOCIATION_ANALYSIS',
        'images':37,'new_images':25,'total_tiny_GT':sum(i['tiny_GT'] for i in images),
        'matched_tiny_GT':len(rows),'zero_response_rows':sum(r['zero_response'] for r in rows),
        'statistics':stats,'gate_passed':bool(passed),'decision':decision,
        'new_complete_model_forwards':26,'cumulative_complete_model_forwards':252,
        'new_suffix_forwards':0,'new_fits':0,'main_model_updates':0,'mainline_validated':False,
        'limits':['Partial response with fixedquery/offset/weights,notfullimageequivarianceorencoderinformationproof.',
            'Jensencontractionisanalgebraicidentity,notnoveltheory;highcancellationaloneisnotafailurecause.',
            'Sameimage/class/sizeratios/inputIoUbinmatchingdoesnotremoveallconfounding;associationisnotcausation.',
            'Newprefixesannotation-selectedbeforepredictionsbutofficialvalmayhavebeenseenbyauthorcheckpointselection.',
            'Gatesareresourceprioritizationcriteria,notconfirmatorysignificance,AP,noveltyortrainingfeasibility.']}
    write(root/'per_GT_response.json',rows);write(root/'matched_geometry_pairs.json',pairs)
    write(root/'per_image_response.json',images);write(root/'analysis_inputs.json',files)
    write(root/'analysis_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','images','total_tiny_GT','matched_tiny_GT','gate_passed','decision']}))
    print(json.dumps({s:{'counts':stats[s]['counts'],'primary':stats[s]['statistics']['cancellation']} for s in stats}))
if __name__=='__main__':
    here=Path(__file__).resolve().parent
    main(here if (here/'protocol.json').exists() else here/'A16_实际特征空间变化审计_v1')
