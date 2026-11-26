"""All-source exact/pixel/perceptual retrieval audit, no learned model."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import json,hashlib,time,collections,concurrent.futures,io,traceback
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DATA=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone')
PROTO=HERE/'07_原图重复条件_事前协议.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
RS=getattr(Image,'Resampling',Image)
TT=getattr(Image,'Transpose',Image)
TRANS=[None,TT.ROTATE_90,TT.ROTATE_180,TT.ROTATE_270,TT.FLIP_LEFT_RIGHT,TT.FLIP_TOP_BOTTOM,TT.TRANSPOSE,TT.TRANSVERSE]
n=32;x=np.arange(n);k=np.arange(n)[:,None];D=np.sqrt(2/n)*np.cos(np.pi*(x+.5)*k/n);D[0]/=np.sqrt(2)
POPCOUNT=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
def descriptor(task):
    split,im,path=task;b=path.read_bytes();h=hashlib.sha256(b).hexdigest()
    with Image.open(io.BytesIO(b)) as z:
        rgb=z.convert('RGB');assert rgb.size==(im['width'],im['height'])
        pix=hashlib.sha256(str(rgb.size).encode()+b'|'+rgb.tobytes()).hexdigest()
        # Transform after32x32 downsample for retrieval; not a transformed-pixel identityhash.
        gray=rgb.convert('L').resize((32,32),resample=RS.LANCZOS)
    med=[];mean=[];dh=[]
    for mode in TRANS:
        v=gray if mode is None else gray.transpose(mode)
        # Explicit contractions avoid this runtime's broken delay-loaded BLAS DLL.
        first=np.einsum('ij,jk->ik',D,np.asarray(v,dtype=float),optimize=False)
        f=np.einsum('ij,kj->ik',first,D,optimize=False)[:8,:8].reshape(-1)
        bits=f>np.median(f[1:]);bits[0]=False;med.append(np.packbits(bits))
        mean.append(np.packbits(f>f.mean()))
        q=np.asarray(v.resize((9,8),resample=RS.BILINEAR),dtype=float);dh.append(np.packbits((q[:,1:]>q[:,:-1]).reshape(-1)))
    return {'split':split,'image_id':int(im['id']),'file_name':im['file_name'],'path':str(path),'width':im['width'],'height':im['height'],'bytes_sha256':h,'pixels_sha256':pix},np.stack(med),np.stack(mean),np.stack(dh)
def hamming(a,b):return POPCOUNT[np.bitwise_xor(a[:,None,:],b[None,:,:])].sum(-1,dtype=np.int16)
def exact_groups(rows,field):
    g=collections.defaultdict(list)
    for i,r in enumerate(rows):g[r[field]].append(i)
    return [v for v in g.values() if len(v)>1]
def within(feats,ids):
    nn=[];pairs=[];f=feats[ids,0]
    for st in range(0,len(ids),128):
        dist=hamming(f[st:st+128],f)
        for j in range(len(dist)):dist[j,st+j]=32767
        for j,row in enumerate(dist):
            ii=st+j;best=int(row.argmin());nn.append({'row':ids[ii],'nearest_row':ids[best],'distance':int(row[best])})
            for kk in np.flatnonzero((row<=4)&(np.arange(len(ids))>ii)):
                pairs.append({'row1':ids[ii],'row2':ids[int(kk)],'distance':int(row[kk])})
    return nn,pairs
def main():
    started=time.perf_counter();out=HERE/'image_duplicates_attempt02';out.mkdir(exist_ok=False)
    sources={str(PROTO):sha(PROTO),str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    (HERE/'07_原图重复条件_事前协议.sha256').write_text(sha(PROTO),encoding='utf-8')
    tasks=[];ds={}
    for split in ['train','val']:
        p=DATA/f'VisDrone2019-DET-{split}/annotations/{split}.json';sources[str(p)]=sha(p);ds[split]=read(p)
        for im in sorted(ds[split]['images'],key=lambda x:x['id']):tasks.append((split,im,p.parent.parent/'images'/im['file_name']))
    assert len(tasks)==7019 and len(ds['train']['images'])==6471 and len(ds['val']['annotations'])==38759
    rows=[];med=[];mean=[];dh=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rr,mm,aa,dd in pool.map(descriptor,tasks):
            rows.append(rr);med.append(mm);mean.append(aa);dh.append(dd);sources[rr['path']]=rr['bytes_sha256']
            if len(rows)%700==0:print(json.dumps({'decoded':len(rows),'of':7019,'seconds':time.perf_counter()-started}),flush=True)
    med=np.stack(med);mean=np.stack(mean);dh=np.stack(dh)
    np.savez_compressed(out/'descriptors.npz',DCT_median=med,DCT_mean=mean,dhash=dh)
    save(out/'source_rows.json',rows)
    nt=6471;trainids=list(range(nt));valids=list(range(nt,len(rows)))
    trainmed=med[:nt].reshape(-1,8);traindh=dh[:nt].reshape(-1,8);trainmean=mean[:nt].reshape(-1,8)
    neighbours=[];flagged=set();allstrong=[];mean_collisions=0
    for st in range(nt,len(rows),16):
        md=hamming(med[st:st+16,0],trainmed);dd=hamming(dh[st:st+16,0],traindh);mc=hamming(mean[st:st+16,0],trainmean)
        mean_collisions+=int(np.count_nonzero(mc==0))
        for j in range(len(md)):
            vi=st+j;order=np.lexsort((np.arange(len(trainmed)),dd[j],md[j]))[:3]
            for rank,kk in enumerate(order):
                neighbours.append({'val_row':vi,'train_row':int(kk)//8,'transform':int(kk)%8,'DCT_distance':int(md[j,kk]),'dhash_distance':int(dd[j,kk]),'rank':rank})
            kk=np.flatnonzero((md[j]<=4)&(dd[j]<=8))
            if len(kk):flagged.add(vi)
            for z in kk:allstrong.append({'val_row':vi,'train_row':int(z)//8,'transform':int(z)%8,'DCT_distance':int(md[j,z]),'dhash_distance':int(dd[j,z])})
    tnn,tpairs=within(med,trainids);vnn,vpairs=within(med,valids)
    exact={name:exact_groups(rows,field) for name,field in [('compressed_bytes','bytes_sha256'),('decoded_RGB_pixels','pixels_sha256')]}
    exstats={}
    for typ,groups in exact.items():
        cross=[g for g in groups if any(rows[i]['split']=='train' for i in g) and any(rows[i]['split']=='val' for i in g)]
        exstats[typ]={'all_duplicate_groups':len(groups),'crosssplit_groups':len(cross),'crosssplit_val_images':len({i for g in cross for i in g if rows[i]['split']=='val'}),
                      'withintrain_duplicated_image_members':len({i for g in groups if sum(rows[j]['split']=='train' for j in g)>1 for i in g if rows[i]['split']=='train'}),
                      'withinval_duplicated_image_members':len({i for g in groups if sum(rows[j]['split']=='val' for j in g)>1 for i in g if rows[i]['split']=='val'})}
    prior=ROOT/'具体问题重新筛选_20261027_v1';seal=read(prior/'delivery_manifest.json');sealmap={x['path']:x['sha256'] for x in seal['outputs']}
    hitsets={s:{t:set() for t in [50,75]} for s in [1,2]}
    for s in [1,2]:
        p=prior/f'06_seed{s}_fixed_rank_arrays.npz';sources[str(p)]=sha(p);assert sources[str(p)]==sealmap[str(p)]
        with np.load(p) as z:
            for c in range(10):
                m=z[f'a0_c{c}_matches'];ig=z[f'a0_c{c}_ignored']
                for t,ix in [(50,0),(75,5)]:hitsets[s][t].update(int(x)-1 for x in m[ix][(m[ix]>0)&~ig[ix]])
    flagim={rows[i]['image_id'] for i in flagged};modelref={}
    for name,group in [('flagged',flagim),('rest',{x['id'] for x in ds['val']['images']}-flagim)]:
        aa=[a for a in ds['val']['annotations'] if a['image_id'] in group];ids={a['id'] for a in aa}
        modelref[name]={'images':len(group),'GT':len(ids),'class_counts':dict(collections.Counter(a['category_id'] for a in aa)),
                       'seeds':{str(s):{str(t):{'hit':len(ids&hitsets[s][t]),'denominator':len(ids),'rate':len(ids&hitsets[s][t])/len(ids) if ids else None} for t in [50,75]} for s in [1,2]}}
    sampled=sorted(allstrong,key=lambda z:hashlib.sha256(f'{z["val_row"]}:{z["train_row"]}:{z["transform"]}'.encode()).hexdigest())[:24]
    controls=sorted([z for z in neighbours if z['rank']==0],key=lambda z:(z['DCT_distance'],z['dhash_distance'],z['val_row']))[:8]
    result={'status':'COMPLETED','counts':{'train':6471,'val':548,'train_annotations':len(ds['train']['annotations']),'valGT':38759},'exact':exstats,
            'strong_similarity_proxy':{'valimages':len(flagged),'valimage_denominator':548,'unique_crosssplit_imagepairs':len({(x['val_row'],x['train_row']) for x in allstrong}),'transform_pair_records':len(allstrong)},
            'mean_hash_exact_collision_transform_records_control':mean_collisions,'withintrain_medianHash_le4_unordered_pairs':len(tpairs),'withinval_medianHash_le4_unordered_pairs':len(vpairs),
            'model_reference':modelref,'sources':sources,'seconds':time.perf_counter()-started,'model_or_training_operations':0,
            'limits':'Perceptual flags are not same-image/leakage truth. D4 small-descriptor transforms are retrieval, not exactRGBtransform tests. No inferred originalsequence/domain identities'}
    for name,v in [('exact_groups.json',exact),('crosssplit_nearest3.json',neighbours),('crosssplit_strong_all.json',allstrong),('train_nearest.json',tnn),('val_nearest.json',vnn),('train_close_pairs.json',tpairs),('val_close_pairs.json',vpairs),('pixel_review_selection.json',{'strong':sampled,'closest_controls':controls}),('results.json',result)]:save(out/name,v)
    save(out/'outputs.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:result[k] for k in ['counts','exact','strong_similarity_proxy','mean_hash_exact_collision_transform_records_control','withintrain_medianHash_le4_unordered_pairs','withinval_medianHash_le4_unordered_pairs','model_reference','seconds']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        p=HERE/'image_duplicates_attempt02/FAILED.txt';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc(),encoding='utf-8');raise
