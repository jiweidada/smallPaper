"""Local NumPy/stdlib verification of the terminal archive, counts and AP math."""
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent
E=ROOT/'verified_task_evidence'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ap_from_flags(tp,fp,n):
    if n==0:return None
    tp=np.cumsum(tp.astype(np.float64),axis=1);fp=np.cumsum(fp.astype(np.float64),axis=1)
    precision=tp/(tp+fp+np.spacing(1));recall=tp/n
    values=np.zeros((10,101))
    for t in range(10):
        if precision.shape[1]==0:continue
        p=precision[t].copy()
        for j in range(len(p)-2,-1,-1):
            if p[j]<p[j+1]:p[j]=p[j+1]
        for j,threshold in enumerate(np.linspace(0,1,101)):
            at=int(np.searchsorted(recall[t],threshold,side='left'))
            if at<len(p):values[t,j]=p[at]
    return values

def independent_AP(blocks,w):
    out={0:[],1:[]};at50=[]
    for b in blocks:
        count=int(b['ngt']@w)
        if count==0:continue
        m=w[b['image_indices']];sel=m>0
        value=ap_from_flags(b['tp'][:,sel]*m[sel],b['fp'][:,sel]*m[sel],count)
        out[b['area_index']].append(float(value.mean()))
        if b['area_index']==0:at50.append(float(value[0].mean()))
    return {'AP':float(np.mean(out[0])) if out[0] else 0.0,
            'APsmall':float(np.mean(out[1])) if out[1] else 0.0,
            'AP50':float(np.mean(at50)) if at50 else 0.0}

def direct_region_mask(rows,h,w,boxes):
    bitmap=np.zeros((h,w),dtype=np.uint8)
    for row in rows:
        if row[5]!=0:continue
        x,y,bw,bh=np.maximum(1,np.asarray(row[:4],dtype=np.int64))
        bitmap[y-1:min(h,y+bh),x-1:min(w,x+bw)]=1
    cumulative=bitmap.cumsum(0,dtype=np.int64).cumsum(1,dtype=np.int64)
    result=[]
    for rect in boxes:
        # Independent scalar implementation of the pinned MATLAB expression.
        pos=[max(1,int(math.floor(v+.5) if v>=0 else math.ceil(v-.5))) for v in rect]
        x,y,bw,bh=pos;x=min(x,w);y=min(y,h);x2=min(w,x+bw);y2=min(h,y+bh)
        fraction=(int(cumulative[y-1,x-1])+int(cumulative[y2-1,x2-1])-int(cumulative[y-1,x2-1])-int(cumulative[y2-1,x-1]))/(bw*bh)
        result.append(fraction<.5)
    return np.asarray(result,dtype=bool)

def overlap(a,b):
    a=np.asarray(a);b=np.asarray(b)
    wh=np.maximum(0,np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    i=wh[:,:,0]*wh[:,:,1]
    return i/(a[:,2,None]*a[:,3,None]+b[None,:,2]*b[None,:,3]-i)

def independent_coverage(raw_cases,formal):
    observed=[]
    for i,c in enumerate(raw_cases):
        rows=np.asarray(c['rows'],dtype=float)
        keep=(rows[:,4]>0)&(rows[:,5]>=1)&(rows[:,5]<=10)
        keep &= direct_region_mask(rows,c['height'],c['width'],rows[:,:4])
        gt=rows[keep,:4];classes=rows[keep,5].astype(int);small=gt[:,2]*gt[:,3]<=1024
        dt={}
        for arm in ['BASE250','NATIVE500','ONEWAY500']:
            with np.load(E/'predictions'/('case%02d_%s.npz'%(i,arm)),allow_pickle=False) as z:
                b=z['boxes'][0];l=z['logits'][0]
            prob=np.float32(1)/(np.float32(1)+np.exp(-l,dtype=np.float32))
            labels=prob.argmax(1);scores=prob[np.arange(len(prob)),labels]
            xyxy=np.stack([b[:,0]-.5*b[:,2],b[:,1]-.5*b[:,3],b[:,0]+.5*b[:,2],b[:,1]+.5*b[:,3]],axis=1)
            xyxy=xyxy*np.array([c['width'],c['height'],c['width'],c['height']],dtype=np.float32)
            xywh=np.column_stack([xyxy[:,:2],xyxy[:,2:]-xyxy[:,:2]]).astype(float)
            ok=(labels>=1)&(labels<=10)&direct_region_mask(rows,c['height'],c['width'],xywh)
            dt[arm]={'boxes':xywh[ok],'labels':labels[ok],'scores':scores[ok]}
        b=dt['BASE250'];iou=overlap(b['boxes'],gt);used=np.zeros(len(gt),dtype=bool)
        for d in np.argsort(-b['scores'],kind='mergesort'):
            available=np.flatnonzero((classes==b['labels'][d])&~used)
            if available.size:
                j=available[np.argmax(iou[d,available])]
                if iou[d,j]>=.5:used[j]=True
        protected=np.flatnonzero(used&small);lost=[]
        for arm in ['NATIVE500','ONEWAY500']:
            d=dt[arm];o=overlap(d['boxes'],gt)
            lost.append(sum(not np.any((d['labels']==classes[j])&(o[:,j]>=.5)) for j in protected))
        row={'case_index':i,'base_correct_small':len(protected),'native_uncompensated_small_losses':lost[0],
             'oneway_uncompensated_small_losses':lost[1]}
        for key in ['base_correct_small','native_uncompensated_small_losses','oneway_uncompensated_small_losses']:
            assert row[key]==formal[i][key],(i,key,row,formal[i])
        observed.append(row)
    return observed

def main():
    assert not (ROOT/'independent_verification.json').exists()
    assert not E.exists()
    receipt=read(ROOT/'export_receipt.json')
    assert sha(ROOT/'task_evidence.zip')==receipt['sha256']
    E.mkdir()
    with zipfile.ZipFile(ROOT/'task_evidence.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            (E/name).resolve().relative_to(E.resolve())
        z.extractall(E)
    manifest=read(E/'evidence_manifest.json')
    for row in manifest['files']:assert sha(E/row['path'])==row['sha256']
    p=read(E/'task_protocol_FROZEN.json');result=read(E/'task_result.json');decoded=read(E/'decoder_result.json')
    assert sha(ROOT/'task_protocol_FROZEN.json')==sha(E/'task_protocol_FROZEN.json')==result['protocol_sha256']
    for f in p['files']:assert sha(E/f['path'])==f['sha256']
    assert decoded['decoder_entries']==decoded['decoder_completed']==decoded['validated_outputs']==45
    assert decoded['layer_entries']==135 and decoded['reused_A26_outputs']==3 and len(decoded['outputs'])==48
    assert decoded['forbidden_full_model_attempts']==decoded['whole_detector_calls']==decoded['optimizer_updates']==0
    for row in decoded['outputs']:assert sha(E/row['path'])==row['sha256']
    with np.load(NEW/'A26_查询扩容接口与缓存核验_v1'/'verified_engineering_evidence'/'predictions_no_GT.npz',allow_pickle=False) as old:
        for arm in p['reuse_A26_arms']:
            with np.load(E/'predictions'/('case00_'+arm+'.npz'),allow_pickle=False) as z:
                for kind in ['boxes','logits']:assert np.array_equal(z[kind],old[arm+'_'+kind])
    assert decoded['state_digest_before']==decoded['state_digest_after']==p['state_digest']
    f=read(E/'prefix_fidelity_observed.json')
    for i in range(12):
        with np.load(E/'predictions'/('case%02d_BASE250.npz'%i),allow_pickle=False) as z:
            b={k:z[k] for k in ['boxes','logits']}
        for arm in ['NATIVE500','TWOWAY500','ONEWAY500']:
            with np.load(E/'predictions'/('case%02d_%s.npz'%(i,arm)),allow_pickle=False) as z:
                for kind in ['boxes','logits']:
                    delta=float(np.abs(z[kind][:,:250]-b[kind]).max())
                    assert delta==f['cases'][i]['prefix_max_abs_differences'][arm][kind]
                    if arm!='NATIVE500':assert delta<=p['prefix_absolute_tolerance'][kind]
    metadata=read(E/'coco_matching_blocks_metadata.json');blocks={a:[] for a in p['arms']}
    with np.load(E/'coco_matching_blocks.npz',allow_pickle=False) as z:
        for row in metadata['rows']:
            b={k:z[row['prefix']+'_'+k] for k in ['tp','fp','image_indices','ngt','presence']}
            b.update(area_index=row['area_index'],cat=row['cat']);blocks[row['arm']].append(b)
    errors=[]
    for arm in p['arms']:
        own=independent_AP(blocks[arm],np.ones(12,dtype=int))
        for key in own:
            errors.append(abs(own[key]-result['metrics'][arm][key]))
            assert errors[-1]<1e-12,(arm,key,own,result['metrics'][arm])
    coverage=read(E/'coverage_counts_observed.json')['cases']
    independent_rows=independent_coverage(read(E/'raw_annotations_verified.json')['cases'],coverage)
    with np.load(E/'bootstrap_weights.npz',allow_pickle=False) as z:w=z['weights']
    assert w.shape==(10000,12) and np.all(w.sum(1)==12) and np.all(w>=0)
    expected=np.random.default_rng(p['statistics']['seed']).multinomial(12,np.ones(12)/12,size=10000)
    assert np.array_equal(w,expected)
    denom=np.array([c['base_correct_small'] for c in coverage]);lost=np.array([c['native_uncompensated_small_losses'] for c in coverage])
    with np.load(E/'paired_bootstrap_samples.npz',allow_pickle=False) as z:
        samples={k:z[k].copy() for k in z.files}
    den=w@denom;num=w@lost
    harm=np.divide(num,den,out=np.zeros(len(w),dtype=float),where=den!=0)
    assert np.array_equal(harm,samples['harm_rate'])
    for key,v in samples.items():
        assert len(v)==10000 and np.isfinite(v).all()
        ordered=np.sort(v);ci=[]
        for q in [.025,.975]:
            at=q*9999;lo=math.floor(at);hi=math.ceil(at);alpha=at-lo
            ci.append(float(ordered[lo]+alpha*(ordered[hi]-ordered[lo])))
        assert np.allclose(ci,result['paired_95pct_percentile_intervals'][key],rtol=0,atol=1e-14)
    for replicate in [0,5000,9999]:
        vals={a:independent_AP(blocks[a],w[replicate]) for a in p['arms']}
        for key,left,right,metric in [('APsmall_O_minus_N','ONEWAY500','NATIVE500','APsmall'),
                                     ('AP_O_minus_N','ONEWAY500','NATIVE500','AP'),
                                     ('APsmall_O_minus_T','ONEWAY500','TWOWAY500','APsmall'),
                                     ('APsmall_O_minus_B','ONEWAY500','BASE250','APsmall')]:
            assert abs((vals[left][metric]-vals[right][metric])-samples[key][replicate])<1e-12
    x=result['point'];ci=result['paired_95pct_percentile_intervals'];c=result['coverage']
    gates={'baseline_small_exposure':c['base_correct_small']>=24 and c['prefixes_with_base_small_TP']>=6,
      'native_uncompensated_harm':x['harm_rate']>=.02 and ci['harm_rate'][0]>0,
      'APsmall_vs_native':x['APsmall_O_minus_N']>=.005 and ci['APsmall_O_minus_N'][0]>0,
      'AP_no_material_loss':x['AP_O_minus_N']>=0 and ci['AP_O_minus_N'][0]>=-.002,
      'specificity_vs_group_mask':x['APsmall_O_minus_T']>=.002 and ci['APsmall_O_minus_T'][0]>0,
      'coverage_vs_base':x['APsmall_O_minus_B']>=.003 and ci['APsmall_O_minus_B'][0]>0}
    assert gates==result['decision']['gates']
    assert all(gates.values())==result['decision']['pilot_all_gates_pass']
    assert result['decision']['mainline_validated'] is False and result['decision']['novelty_cleared'] is False
    verified={'status':'PASS_ARCHIVE_COUNTS_POINT_AP_COVERAGE_BOOTSTRAP_AND_DECISION',
      'archive_sha256':sha(ROOT/'task_evidence.zip'),'manifest_files_verified':len(manifest['files']),
      'raw_arm_outputs_verified':48,'new_complete_decoder_calls':45,'new_layers':135,'A26_reused_outputs':3,
      'independent_AP_max_absolute_error':max(errors),'coverage_cases_reaggregated':len(independent_rows),
      'bootstrap_harm_samples_verified':10000,'bootstrap_AP_replicates_independently_recomputed':[0,5000,9999],
      'all_interval_quantiles_verified':True,'all_six_gates_independently_verified':True,
      'new_whole_detector_calls':0,'new_optimizer_updates':0,'mainline_validated':False,
      'verification_scope':'Integrity and arithmetic, not independent scientific confirmation or novelty proof.'}
    (ROOT/'independent_verification.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verified))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
