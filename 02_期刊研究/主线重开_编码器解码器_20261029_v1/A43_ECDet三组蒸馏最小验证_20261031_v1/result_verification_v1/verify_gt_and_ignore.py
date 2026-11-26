"""Independent raw-annotation and ignore-policy check; no training-code import."""
import argparse,csv,hashlib,json,pathlib
import numpy as np

def fraction_grid(box,regions):
    box=np.asarray(box,dtype=np.float64)
    if not len(regions):return 0.
    regions=np.asarray(regions,dtype=np.float64)
    lo=np.maximum(regions[:,:2],box[:2]);hi=np.minimum(regions[:,2:],box[2:])
    keep=np.all(hi>lo,axis=1);lo=lo[keep];hi=hi[keep]
    if not len(lo):return 0.
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);ys=np.unique(np.r_[lo[:,1],hi[:,1]])
    mx=(xs[:-1]+xs[1:])/2;my=(ys[:-1]+ys[1:])/2
    coverage=np.any((mx[:,None,None]>=lo[None,None,:,0])&(mx[:,None,None]<hi[None,None,:,0])&(my[None,:,None]>=lo[None,None,:,1])&(my[None,:,None]<hi[None,None,:,1]),axis=2)
    return float(np.sum(np.diff(xs)[:,None]*np.diff(ys)[None,:]*coverage)/np.prod(box[2:]-box[:2]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--annotations',type=pathlib.Path,required=True);ap.add_argument('--manifest',type=pathlib.Path,required=True);ap.add_argument('--gt',type=pathlib.Path,required=True);ap.add_argument('--predictions',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    assert fraction_grid([0,0,10,10],[[0,0,6,10],[4,0,8,10]])==.8
    assert fraction_grid([0,0,10,10],[[0,0,2,10],[8,0,10,10]])==.4
    gt=json.loads(a.gt.read_text(encoding='utf-8'));manifest=json.loads(a.manifest.read_text(encoding='utf-8'));expected=[];images=[];ignored={};removed=0
    for row in manifest['rows']:
        p=a.annotations/(pathlib.Path(row['file']).stem+'.txt')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['annotation_sha256']
        w,h=row['width'],row['height'];boxes=[];regions=[]
        for line in csv.reader(p.read_text(encoding='utf-8').splitlines()):
            x,y,bw,bh,score,cls,*_=map(float,[v.strip()for v in line if v.strip()])
            lo=np.clip([x,y],[0,0],[w,h]);hi=np.clip([x+bw,y+bh],[0,0],[w,h])
            if not np.all(hi>lo):continue
            b=np.r_[lo,hi].tolist()
            if int(cls)==0:regions.append(b)
            if score>0 and 1<=int(cls)<=10:boxes.append((b,int(cls)))
        assert len(boxes)==row['valid_gt']
        ignored[row['id']]=regions
        images.append({'id':row['id'],'width':w,'height':h,'file_name':row['file']})
        for box,cls in boxes:
            if fraction_grid(box,regions)>=.5:removed+=1;continue
            x,y,x1,y1=box;bw=x1-x;bh=y1-y
            expected.append({'id':len(expected)+1,'image_id':row['id'],'category_id':cls,'bbox':[x,y,bw,bh],'area':bw*bh,'iscrowd':0})
    assert gt['images']==images
    assert gt['annotations']==expected
    assert gt['ignored_valid_gt']==removed
    pred=json.loads(a.predictions.read_text(encoding='utf-8'));inside=[]
    for i,v in enumerate(pred):
        x,y,w,h=v['bbox'];coverage=fraction_grid([x,y,x+w,y+h],ignored[v['image_id']])
        if coverage>=.5:inside.append({'index':i,'image_id':v['image_id'],'coverage':coverage})
    assert not inside,inside[:5]
    result={'status':'PASS','raw_annotation_files_verified':len(manifest['rows']),'entire_evaluated_GT_reconstructed_from_raw_annotations':True,'evaluated_GT':len(expected),'ignored_valid_GT':removed,'retained_predictions_checked':len(pred),'retained_predictions_inside_ignore_regions':len(inside),'independent_union_implementation':'orthogonal-cell coverage grid, separate from training evaluator sweep','scope_limit':'Checks saved evaluated predictions. Does not regenerate excluded or raw network predictions. No detector forward or optimizer step.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
