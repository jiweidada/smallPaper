"""Original annotation/actual cached-match screen. No model runtime imports."""
import json, hashlib, time, collections, csv, traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
ANN=DATA/'annotations/val.json'
PROTO=HERE/'04_边界条件_事前协议.json'
PRIOR=ROOT/'具体问题重新筛选_20261027_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def iou(a):
    wh=np.maximum(0,np.minimum(a[:,None,:2]+a[:,None,2:],a[None,:,:2]+a[None,:,2:])-np.maximum(a[:,None,:2],a[None,:,:2]))
    it=wh.prod(-1);ar=a[:,2:].prod(-1)
    v=it/np.maximum(ar[:,None]+ar[None,:]-it,1e-12);np.fill_diagonal(v,0)
    return v.max(1,initial=0)
def main():
    start=time.perf_counter();out=HERE/'boundary_condition_attempt01';out.mkdir(exist_ok=False)
    sources={str(ANN):sha(ANN),str(PROTO):sha(PROTO),str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    (HERE/'04_边界条件_事前协议.sha256').write_text(sources[str(PROTO)],encoding='utf-8')
    assert sources[str(ANN)]=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    ds=read(ANN);ims={x['id']:x for x in ds['images']};by=collections.defaultdict(list)
    for a in ds['annotations']:by[a['image_id']].append(a)
    hits={s:{t:set() for t in [0,5]} for s in [1,2]}
    # Bind stored matching arrays to their sealed producer output hashes.
    sealed=read(PRIOR/'delivery_manifest.json')
    sealmap={z['path']:z['sha256'] for z in sealed['outputs']}
    for s in [1,2]:
        p=PRIOR/f'06_seed{s}_fixed_rank_arrays.npz';sources[str(p)]=sha(p)
        assert sources[str(p)]==sealmap[str(p)]
        with np.load(p) as a:
            for c in range(10):
                m=a[f'a0_c{c}_matches'];ignore=a[f'a0_c{c}_ignored']
                for t in [0,5]:hits[s][t].update(int(x)-1 for x in m[t][(m[t]>0)&~ignore[t]])
    rows=[];join_ambiguous_same_attributes=0;maxjoin=0.;rawpositive=0
    for iid in sorted(ims):
        im=ims[iid];p=DATA/'annotations'/(Path(im['file_name']).stem+'.txt');sources[str(p)]=sha(p)
        raw=np.array([[float(x) for x in line.strip().split(',')[:8]] for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()])
        raw=raw[(raw[:,4]>0)&(raw[:,5]>=1)&(raw[:,5]<=10)];rawpositive+=len(raw)
        aa=by[iid];assert len(raw)==len(aa),(iid,len(raw),len(aa))
        used=np.zeros(len(raw),bool);bb=np.array([x['bbox'] for x in aa]);crowd=iou(bb)
        for j,a in enumerate(aa):
            dif=np.max(np.abs(raw[:,:4]-bb[j]),axis=1)
            ix=np.flatnonzero((dif<.01)&(raw[:,5]==a['category_id']+1)&~used)
            assert len(ix)>0,('missing_source',iid,a['id'],float(dif.min()))
            if len(ix)>1:
                assert np.all(raw[ix,4:]==raw[ix[0],4:]),('ambiguous_flags',iid,a['id'])
                join_ambiguous_same_attributes+=1
            k=int(ix[np.argmin(dif[ix])]);used[k]=True;maxjoin=max(maxjoin,float(dif[k]))
            x,y,w,h=bb[j];rx,ry,rw,rh=raw[k,:4];W,H=im['width'],im['height']
            wh640=np.array([w*640/W,h*640/H]);margins=np.array([x*640/W,y*640/H,(W-x-w)*640/W,(H-y-h)*640/H])
            d=float(margins.min());side=int(margins.argmin());group='invalid' if d<0 else 'near' if d<32 else 'middle' if d<64 else 'inner'
            visible=bool(raw[k,6]==0 and raw[k,7]==0 and rx>0 and ry>0 and rx+rw<W and ry+rh<H)
            shortbin=int(np.digitize(wh640.min(),[4,8,16,32,64]));aspectbin=int(np.digitize(w*H/(h*W),[.5,1,2]));crowdbin=int(crowd[j]>=.1)
            row={'GTid':int(a['id']),'image_id':iid,'class':int(a['category_id']),'group':group,'near_side':side,'margin640':d,
                 'visible_proxy':int(visible),'raw_truncation':int(raw[k,6]),'raw_occlusion':int(raw[k,7]),'short640':float(wh640.min()),'long640':float(wh640.max()),
                 'short_bin':shortbin,'aspect_bin':aspectbin,'neighbour_bin':crowdbin,'neighbour_maxIoU':float(crowd[j])}
            for s in [1,2]:
                for t in [0,5]:row[f'seed{s}_hit{50+t*5}']=int(a['id'] in hits[s][t])
            rows.append(row)
        assert used.all()
    assert len(rows)==rawpositive==38759
    with (out/'per_GT.csv').open('x',encoding='utf-8',newline='') as f:
        ww=csv.DictWriter(f,fieldnames=list(rows[0]));ww.writeheader();ww.writerows(rows)
    def counts(rr):
        z={'GT':len(rr),'images':len({x['image_id'] for x in rr})}
        for s in [1,2]:
            z[f'seed{s}']={str(t):{'hit':sum(x[f'seed{s}_hit{t}'] for x in rr),'denominator':len(rr),
                                  'rate':sum(x[f'seed{s}_hit{t}'] for x in rr)/len(rr) if rr else None} for t in [50,75]}
        return z
    result={'status':'COMPLETED','protocol_sha256':sha(PROTO),'sources':sources,'GT':len(rows),'raw_join':{'positive':rawpositive,'max_original_px_difference':maxjoin,'ambiguous_identical_attribute_matches':join_ambiguous_same_attributes},
            'groups':{},'per_class':{},'standardized':{},'sourceflags':dict(collections.Counter((x['raw_truncation'],x['raw_occlusion']) for x in rows))}
    # JSON keys for tuple flags are rendered explicitly below.
    result['sourceflags']={f'trunc{k[0]}_occ{k[1]}':v for k,v in result['sourceflags'].items()}
    for scope in ['allGT','visible_proxy']:
        rr=rows if scope=='allGT' else [x for x in rows if x['visible_proxy']]
        result['groups'][scope]={g:counts([x for x in rr if x['group']==g]) for g in ['near','middle','inner','invalid']}
        result['per_class'][scope]={str(c):{g:counts([x for x in rr if x['class']==c and x['group']==g]) for g in ['near','inner']} for c in range(10)}
    rr=[x for x in rows if x['visible_proxy'] and x['group'] in ['near','inner']]
    strata=collections.defaultdict(lambda: {'near':[],'inner':[]})
    for x in rr:
        key=(x['image_id'],x['class'],x['short_bin'],x['aspect_bin'],x['neighbour_bin']);strata[key][x['group']].append(x)
    near=[];inner=[];strows=[]
    for key,z in strata.items():
        if not z['near'] or not z['inner']:continue
        near.extend(z['near']);inner.extend(z['inner']);sr={'stratum':list(key),'near_GT':len(z['near']),'inner_GT':len(z['inner'])}
        for s in [1,2]:
            for t in [50,75]:
                sr[f'seed{s}_hit{t}_near']=sum(x[f'seed{s}_hit{t}'] for x in z['near'])
                sr[f'seed{s}_hit{t}_inner']=sum(x[f'seed{s}_hit{t}'] for x in z['inner'])
        strows.append(sr)
    result['standardized']={'support_near':counts(near),'support_inner_unique':counts(inner),'support_strata':len(strows),
                            'near_total':result['groups']['visible_proxy']['near']['GT'],'seeds':{}}
    for s in [1,2]:
        zz={}
        for t in [50,75]:
            expected=sum(v['near_GT']*v[f'seed{s}_hit{t}_inner']/v['inner_GT'] for v in strows)
            actual=sum(x[f'seed{s}_hit{t}'] for x in near);n=len(near)
            zz[str(t)]={'actual_near_hits':actual,'near_denominator':n,'weighted_inner_expected_hits':expected,
                        'weighted_inner_rate':expected/n if n else None,'near_minus_control_rate':(actual-expected)/n if n else None,
                        'note':'weighted expected count is descriptive and fractional, not actualGT hits'}
        result['standardized']['seeds'][str(s)]=zz
    result['elapsed_seconds']=time.perf_counter()-start
    save(out/'strata.json',strows);save(out/'results.json',result)
    save(out/'outputs.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:result[k] for k in ['GT','raw_join','groups','standardized','elapsed_seconds']},ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except Exception:
        p=HERE/'boundary_condition_attempt01/FAILED.txt';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc(),encoding='utf-8');raise
