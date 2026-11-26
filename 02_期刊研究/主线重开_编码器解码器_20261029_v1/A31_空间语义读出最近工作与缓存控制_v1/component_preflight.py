"""Synthetic geometry plus native cache format checks;no new intervention AP."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
from cache_components import A30,D,N,R,read,sha,region_functions,original_from_input,raw_original_corners,probabilities,bilinear,topk,score_flow
sys.stdout.reconfigure(encoding='utf-8')
def main():
    assert not (R/'component_preflight.json').exists()
    receipt=read(N/'65_A27任务指标与候选判定核验回执.json');p=N/'A27_查询扩容同预算任务诊断_v1'/'o27_metrics.py';assert sha(p)==receipt['files_sha256'][str(p)]
    manifest=read(N/'A27_查询扩容同预算任务诊断_v1'/'official_toolkit_source_manifest.json')
    for x in manifest['files']:
        data=(Path(manifest['source_root'])/x['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==x['sha256'];assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==x['git_blob_sha1']
    f=region_functions();assert f['matlab_round']([.5,1.5,-.5,-1.5]).tolist()==[1,2,-1,-2]
    integral=f['ignore_integral']([[1,1,4,9,0,0,0,0],[1,1,2,9,0,0,0,0]],12,12);assert integral[-1,-1]==50;assert f['region_keep']([[1,1,8,8],[1,1,9,8]],integral).tolist()==[False,True]
    rng=np.random.default_rng(431031);rects=rng.uniform(-2,17,(64,4));rects[:,2:]=rng.uniform(.1,20,(64,2));observed=f['region_keep'](rects,integral);bitmap=np.zeros((12,12),dtype=np.uint8);bitmap[:10,:5]=1;expected=[]
    for box in rects:
        values=[max(1,int(math.floor(v+.5) if v>=0 else -math.floor(-v+.5))) for v in box];x,y,w,h=values;x=min(12,x);y=min(12,y);covered=sum(int(bitmap[yy,xx]) for yy in range(y,min(12,y+h)) for xx in range(x,min(12,x+w)));expected.append(covered/(w*h)<.5)
    assert observed.tolist()==expected
    yy,xx=np.mgrid[:4,:6];field=np.stack([xx+10*yy,2*xx-yy]).astype(float);sample=bilinear(field,np.array([0,2,2.5,-4,20]),np.array([0,3,1.5,-3,30]));assert np.allclose(sample[:,2],[17.5,3.5]);assert np.array_equal(sample[:,0],field[:,0,0]) and np.array_equal(sample[:,-1],field[:,-1,-1])
    cfg=read(A30/'native_dual_head_protocol_FROZEN.json');meta=cfg['cases'];sources={k:read(D/(name+'_predictions_input.json')) for k,name in [('M','Mbox_Mscore_NMS'),('O','Obox_Oscore_TOPK')]};originals={k:read(D/(name+'_predictions_original.json')) for k,name in [('M','Mbox_Mscore_NMS'),('O','Obox_Oscore_TOPK')]}
    decoded_meta=read(D/'capture_metadata.json');by={};original_by={}
    for k in ['M','O']:
        by[k]={c['image_id']:[x for x in sources[k] if x['image_id']==c['image_id']] for c in meta};original_by[k]={c['image_id']:[x for x in originals[k] if x['image_id']==c['image_id']] for c in meta}
    inverse_error=raw_error=score_error=0.;rows=[];checked=0
    for case,capture in zip(meta,decoded_meta):
        for k in ['M','O']:
            restored=original_from_input(by[k][case['image_id']],case);known=original_by[k][case['image_id']];assert len(restored)==len(known)
            for x,y in zip(restored,known):inverse_error=max(inverse_error,max(abs(a-b) for a,b in zip(x['bbox'],y['bbox'])));assert x['category_id']==y['category_id'] and x['score']==y['score'];checked+=1
        data=np.load(D/'captures'/('case_%03d.npz'%case['index']),allow_pickle=False);corners=raw_original_corners(data['O_decoded_xyxy'],case);records=original_by['O'][case['image_id']];ids=capture['lineage']['Obox_Oscore_TOPK'];assert len(ids)==len(records)
        for i,x in zip(ids,records):
            b=corners[i];want=[b[0],b[1],b[2]-b[0],b[3]-b[1]];raw_error=max(raw_error,max(abs(a-b) for a,b in zip(want,x['bbox'])))
        prob=probabilities(data['O_scores']);zero,_=score_flow(prob,data['anchors_px'],corners,case,[8,16,32],[[80,80],[40,40],[20,20]],0);assert np.array_equal(prob,zero)
        a,c,s=topk(zero);observed_pairs={(int(i),int(j)) for i,j,z in zip(a,c,s) if z>=.05};wanted_pairs={(int(i),x['category_id']) for i,x in zip(ids,records) if x['score']>=.05};assert observed_pairs==wanted_pairs
        for i,x in zip(ids,records):score_error=max(score_error,abs(float(prob[x['category_id'],i])-x['score']))
        rows.append({'image_id':case['image_id'],'zero_grid_bitwise_identity':True,'native_high_score_pairs_identical':True})
    assert inverse_error<1e-10 and raw_error<2e-4 and score_error<1e-7,(inverse_error,raw_error,score_error)
    out={'status':'PASS_REGION64_BRUTE_FIXTURES_AND100_NATIVE_FORMAT_IDENTITY','official_git_blobs_verified':len(manifest['files']),'input_to_original_records_checked':checked,'inverse_roundtrip_max_abs':inverse_error,'raw_original_corner_max_abs':raw_error,'analytic_sigmoid_vs_native_max_abs':score_error,'zero_grid_identity_images':100,'native_high_score_pair_identity_images':100,'scope':'No shifted-score intervention or newAP measured. CUDA scalar-divide arithmetic is byte-supported by cached native corners on100images;not a new model run.','rows':rows,'new_model_calls':0}
    with (R/'component_preflight.json').open('x',encoding='utf-8') as g:json.dump(out,g,ensure_ascii=False,indent=2);g.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
