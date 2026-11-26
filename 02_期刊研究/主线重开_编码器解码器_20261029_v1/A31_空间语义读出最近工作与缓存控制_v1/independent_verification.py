"""Independent interval-union filtering,scalar score reads,and affine COCO checks."""
from pathlib import Path
import collections,copy,itertools,json,math,sys,time
import numpy as np
from cache_components import R,N,A30,D,read,sha,functions,raw_original_corners
sys.stdout.reconfigure(encoding='utf-8')
def round_away(x):return math.floor(x+.5) if x>=0 else math.ceil(x-.5)
def scalar_keep(box,regions,width,height):
    if not regions:return True
    x,y,w,h=[max(1,round_away(float(v))) for v in box];x=min(x,width);y=min(y,height);endx=min(width,x+w);endy=min(height,y+h);pieces=[]
    for region in regions:
        a,b,c,d=[max(1,int(v)) for v in region];left=max(x,a-1);right=min(endx,width,a+c);top=max(y,b-1);bottom=min(endy,height,b+d)
        if right>left and bottom>top:pieces.append((left,right,top,bottom))
    xs=sorted({v for q in pieces for v in q[:2]});covered=0
    for a,b in zip(xs,xs[1:]):
        intervals=sorted((q[2],q[3]) for q in pieces if q[0]<=a and q[1]>=b);length=0;lo=hi=None
        for c,d in intervals:
            if lo is None:lo,hi=c,d
            elif c>hi:length+=hi-lo;lo,hi=c,d
            else:hi=max(hi,d)
        if lo is not None:length+=hi-lo
        covered+=(b-a)*length
    return covered/(w*h)<.5
def scalar_score(data,corners,case,anchor,category,alpha):
    if anchor<6400:offset,h,w,stride=0,80,80,8
    elif anchor<8000:offset,h,w,stride=6400,40,40,16
    else:offset,h,w,stride=8000,20,20,32
    p=data['anchors_px'][anchor].astype(float);z=case['geometry'];l,t=z['pad_ltrb'][:2];s=z['ratio'];b=corners[anchor];cx=(b[0]+b[2])*.5*s+l;cy=(b[1]+b[3])*.5*s+t
    x=min(w-1,max(0,(p[0]+alpha*(cx-p[0]))/stride-.5));y=min(h-1,max(0,(p[1]+alpha*(cy-p[1]))/stride-.5));x0=math.floor(x);y0=math.floor(y);x1=min(w-1,x0+1);y1=min(h-1,y0+1);tx=x-x0;ty=y-y0
    value=0.
    for xx,yy,weight in [(x0,y0,(1-tx)*(1-ty)),(x1,y0,tx*(1-ty)),(x0,y1,(1-tx)*ty),(x1,y1,tx*ty)]:
        logit=max(-80,min(80,float(data['O_scores'][category,offset+yy*w+xx])));value+=weight/(1+math.exp(-logit))
    return value
def main():
    start=time.monotonic();assert not (R/'independent_verification.json').exists();cfg=read(R/'cache_protocol_FROZEN.json');out=read(R/'cache_result.json');assert out['status'].endswith('COMPLETED') and not (R/'failure.json').exists();status=read(R/'status.json');assert status['status']=='COMPLETED'
    for name,h in read(R/'execution_inputs_manifest.json').items():assert sha(R/name)==h,name
    raw=read(R/'raw_annotations_verified.json')['cases'];cases={x['image_id']:x for x in raw};gt=read(R/'region_policy_GT.json');uids={x['stable_uid'] for x in gt['annotations']};expected_uids=set();cached_region={};checked_annotations=0
    for c in raw:
        regions=[x['values'][:4] for x in c['rows'] if x['values'][5]==0];cached_region[c['image_id']]=regions
        for row in c['rows']:
            a=row['values']
            if a[4]>0 and 1<=a[5]<=10:
                checked_annotations+=1
                if scalar_keep(a[:4],regions,c['width'],c['height']):expected_uids.add(c['file_name']+':'+str(row['source_row']))
    assert expected_uids==uids and checked_annotations==4491
    predictions={name:read(R/'predictions'/(name+'.json')) for name in out['metrics']};by={name:collections.defaultdict(list) for name in predictions}
    retained_checks=0;native_full_checks=0
    for name,records in predictions.items():
        for record in records:by[name][record['image_id']].append(record)
    for c in raw:
        cache={}
        for name,records in by.items():
            for d in records[c['image_id']]:
                key=tuple(d['bbox'])
                if key not in cache:cache[key]=scalar_keep(key,cached_region[c['image_id']],c['width'],c['height'])
                assert cache[key],(name,c['image_id']);retained_checks+=1
        def signature(rows):return collections.Counter(tuple(x['bbox'])+(x['category_id'],) for x in rows)
        assert signature(by['CENTER_KEEP_POOL'][c['image_id']])==signature(by['O_CACHED_NATIVE'][c['image_id']])==signature(by['REVERSE_KEEP_POOL'][c['image_id']])
    for old,new in [('Mbox_Mscore_NMS','M_NATIVE_NMS_REFERENCE'),('Obox_Oscore_TOPK','O_CACHED_NATIVE')]:
        source=read(D/(old+'_predictions_original.json'));kept=[]
        for record in source:
            c=cases[record['image_id']];native_full_checks+=1
            if scalar_keep(record['bbox'],cached_region[c['image_id']],c['width'],c['height']):kept.append(record)
        assert kept==predictions[new],new
    scalar_reads=0
    for c in raw:
        with np.load(D/'captures'/('case_%03d.npz'%c['index']),allow_pickle=False) as data:
            corners=raw_original_corners(data['O_decoded_xyxy'],c);boxindex=collections.defaultdict(list)
            for i,b in enumerate(corners):boxindex[(float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1]))].append(i)
            for name,alpha in [('CENTER_FLOW',1.),('REVERSE_FLOW',-1.),('HALF_CENTER_FLOW',.5),('HALF_REVERSE_FLOW',-.5),('CENTER_KEEP_POOL',1.),('REVERSE_KEEP_POOL',-1.)]:
                records=by[name][c['image_id']]
                for index in sorted({0,len(records)//2,len(records)-1}):
                    if index<0 or not records:continue
                    d=records[index];ids=boxindex[tuple(d['bbox'])];assert ids
                    assert any(abs(scalar_score(data,corners,c,i,d['category_id'],alpha)-d['score'])<1e-12 for i in ids),(name,c['image_id'],index);scalar_reads+=1
    core=functions(A30/'independent_coco_reaggregation.py',['matrix_iou','match_image','category_precision','ap','operating_points'],{'np':np,'THRESHOLDS':np.linspace(.5,.95,10),'RECALL':np.linspace(0,1,101)})
    images=sorted(cases);cats=list(range(10));small={x['id'] for x in gt['annotations'] if x['_area_input']<1024};all_gt=collections.defaultdict(list)
    for g in gt['annotations']:all_gt[(g['category_id'],g['image_id'])].append(g)
    maximum_error=0.;boot_records={};summary={}
    for name,records in predictions.items():
        dp=collections.defaultdict(list)
        for i,d in enumerate(records):x=dict(d,_id=i+1);dp[(x['category_id'],x['image_id'])].append(x)
        sets={}
        for label in ['all','originalsmall','inputsmall']:
            res={}
            for cat in cats:
                for im in images:
                    gs=copy.deepcopy(all_gt[(cat,im)]);ds=copy.deepcopy(dp[(cat,im)]);area=[0,1e10] if label=='all' else [0,math.nextafter(1024.,-math.inf)]
                    if label=='inputsmall':
                        ratio=cases[im]['geometry']['ratio'];pad=cases[im]['geometry']['pad_ltrb'][:2]
                        for x in gs+ds:x['bbox']=[x['bbox'][0]*ratio+pad[0],x['bbox'][1]*ratio+pad[1],x['bbox'][2]*ratio,x['bbox'][3]*ratio]
                        for x in gs:x['area']=x['_area_input']
                    res[(cat,im)]=core['match_image'](ds,gs,area)
            sets[label]=res
        a,ap50=core['ap'](sets['all'],images,cats);aps,_=core['ap'](sets['originalsmall'],images,cats);apsi,_=core['ap'](sets['inputsmall'],images,cats);observed={'AP_original_all':a,'AP50_original_all':ap50,'AP_original_small':aps,'AP_input_small':apsi}
        for k,v in observed.items():err=abs(v-out['metrics'][name][k]);maximum_error=max(maximum_error,err);assert err<1e-12,(name,k,err)
        budgets=[x['FP_budget'] for x in out['metrics'][name]['same_FP']];points=core['operating_points'](sets['all'],images,cats,small,budgets)
        for x,y in zip(points,out['metrics'][name]['same_FP']):
            for key in ['FP_budget','actual_FP','TP','small_TP','small_FN','global_score_threshold']:assert x[key]==y[key],(name,key)
        if name in ['O_CACHED_NATIVE','ZERO_GRID','CENTER_FLOW','REVERSE_FLOW','CENTER_KEEP_POOL']:boot_records[name]=sets['inputsmall']
        summary[name]=observed;print(json.dumps({'verified_group':name,'elapsed':time.monotonic()-start}),flush=True)
    b=np.load(R/'bootstrap_samples.npz',allow_pickle=False);want=np.random.default_rng(cfg['metrics']['bootstrap']['seed']).integers(0,100,size=(1000,100));assert np.array_equal(b['draws'],want)
    for i in [0,499,999]:
        values={name:core['ap'](records,images,cats,b['draws'][i])[0] for name,records in boot_records.items()}
        for a,c in cfg['metrics']['bootstrap']['contrasts']:assert abs(values[a]-values[c]-b[a+'__minus__'+c][i])<1e-12
    for a,c in cfg['metrics']['bootstrap']['contrasts']:assert np.allclose(np.quantile(b[a+'__minus__'+c],[.025,.975]),out['bootstrap95'][a+'__minus__'+c],rtol=0,atol=1e-15)
    result={'status':'PASS_SOURCE_UNION_SCALAR_READS_ALL9_AP_AND_FIXED_FP','raw_valid_GT_filter_checks':checked_annotations,'retained_GT':len(uids),'retained_prediction_union_checks':retained_checks,'native_prefilter_predictions_checked':native_full_checks,'shifted_scalar_score_reads':scalar_reads,'groups_AP_verified':9,'AP_max_abs_error':maximum_error,'bootstrap_seed_draws_checked':1000,'bootstrap_draws_independently_reaggregated':[0,499,999],'fixed_pool_identity_verified_images':100,'new_model_calls':0,'scope':'Source-alignedignored-regionCOCO101,not officialbenchmark or new algorithm validation.','elapsed_seconds':time.monotonic()-start}
    with (R/'independent_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
