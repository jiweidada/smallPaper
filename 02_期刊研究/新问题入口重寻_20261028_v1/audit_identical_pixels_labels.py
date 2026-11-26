"""Exact-source label disagreement + two frozen annotation-version sensitivities."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,copy,hashlib,collections,time,importlib.util,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DATA=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone')
DROOT=Path('D:/数据集/visdrone2019')
PRE=HERE/'image_duplicates_attempt02'
PROTO=HERE/'10_同像素异标注_事前协议.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    it=wh.prod(-1);return it/np.maximum(a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)[None,:]-it,1e-12)
def matching(I,c1,c2,t):
    edges=[sorted(np.flatnonzero((I[i]>=t)&(c2==c1[i])).tolist(),key=lambda j:(-I[i,j],j)) for i in range(len(c1))]
    owner={}
    def dfs(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or dfs(owner[j],seen):owner[j]=i;return True
        return False
    for i in range(len(c1)):dfs(i,set())
    return sorted([(i,j,float(I[i,j])) for j,i in owner.items()])
def main():
    start=time.perf_counter();out=HERE/'identical_pixels_labels_attempt01';out.mkdir(exist_ok=False)
    sources={str(PROTO):sha(PROTO),str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    (HERE/'10_同像素异标注_事前协议.sha256').write_text(sources[str(PROTO)],encoding='utf-8')
    rr=read(PRE/'source_rows.json');groups=read(PRE/'exact_groups.json')['decoded_RGB_pixels']
    for p in [PRE/'source_rows.json',PRE/'exact_groups.json',PRE/'results.json']:sources[str(p)]=sha(p)
    ds={};anns={};images={}
    for sp in ['train','val']:
        p=DATA/f'VisDrone2019-DET-{sp}/annotations/{sp}.json';sources[str(p)]=sha(p);ds[sp]=read(p)
        bb=collections.defaultdict(list)
        for a in ds[sp]['annotations']:bb[a['image_id']].append(a)
        anns[sp]=bb;images[sp]={x['id']:x for x in ds[sp]['images']}
    result={'status':'RUNNING','protocol_sha256':sha(PROTO),'groups':[],'sources':sources,'CPU_evaluator_calls':0,'new_models_checkpoints_heads_forwards_backwards_training':0}
    palette=[(220,40,30),(240,110,10),(120,50,220),(10,160,20),(20,110,230),(180,50,130),(20,170,170),(130,100,0),(50,50,255),(80,120,80)]
    for gi,g in enumerate(groups):
        entries=[];raws=[];panels=[]
        for ix in g:
            r=rr[ix];sp,iid=r['split'],r['image_id'];p=Path(r['path']);sources[str(p)]=sha(p);assert sources[str(p)]==r['bytes_sha256']
            dimg=DROOT/f'VisDrone2019-DET-{sp}/images'/r['file_name'];sources[str(dimg)]=sha(dimg);assert sources[str(dimg)]==sources[str(p)]
            stem=Path(r['file_name']).stem;raw=DATA/f'VisDrone2019-DET-{sp}/annotations'/(stem+'.txt');dp=DROOT/f'VisDrone2019-DET-{sp}/annotations'/(stem+'.txt')
            sources[str(raw)]=sha(raw);sources[str(dp)]=sha(dp);assert sources[str(raw)]==sources[str(dp)]
            allraw=np.array([[float(x) for x in line.split(',')[:8]] for line in raw.read_text(encoding='utf-8-sig').splitlines() if line.strip()])
            pos=allraw[(allraw[:,4]>0)&(allraw[:,5]>=1)&(allraw[:,5]<=10)];aa=anns[sp][iid];used=set();maxdiff=0.
            for a in aa:
                dd=np.max(np.abs(pos[:,:4]-a['bbox']),axis=1);ixs=[j for j in np.flatnonzero((dd<.01)&(pos[:,5]==a['category_id']+1)) if j not in used]
                assert len(ixs)==1,('rawjoin',sp,iid,a['id'],ixs);used.add(ixs[0]);maxdiff=max(maxdiff,float(dd[ixs[0]]))
            assert len(used)==len(aa)==len(pos)
            entries.append({'split':sp,'image_id':iid,'file_name':r['file_name'],'GT':len(aa),'COCO_class_counts':dict(collections.Counter(a['category_id'] for a in aa)),
                            'raw_lines':len(allraw),'rawTXT_sha256':sha(raw),'raw_positive_to_COCO_max_px_difference':maxdiff,'raw0_or11':int(np.count_nonzero((allraw[:,5]==0)|(allraw[:,5]==11))),
                            'compressed_sha256':sha(p),'RGB_sha256':r['pixels_sha256'],'C_D_images_and_TXT_equal':True})
            raws.append(pos)
            with Image.open(p) as z:panel=z.convert('RGB')
            dr=ImageDraw.Draw(panel)
            for a in aa:
                x,y,w,h=a['bbox'];c=a['category_id'];dr.rectangle((x,y,x+w,y+h),outline=palette[c],width=2);dr.text((x,max(0,y-11)),str(c),fill=palette[c])
            panels.append(panel)
        assert len(entries)==2 and entries[0]['compressed_sha256']==entries[1]['compressed_sha256']
        A,B=raws;I=iou(A[:,:4],B[:,:4]);c1=A[:,5]-1;c2=B[:,5]-1
        agree={}
        for t in [.5,.75]:
            pairs=matching(I,c1,c2,t);agree[str(t)]={'matched_pairs':len(pairs),'A_unmatched':len(A)-len(pairs),'B_unmatched':len(B)-len(pairs),'A_denominator':len(A),'B_denominator':len(B),'pairs':pairs}
        exactA=collections.Counter(tuple(x[:4])+(int(x[5]),) for x in A);exactB=collections.Counter(tuple(x[:4])+(int(x[5]),) for x in B)
        result['groups'].append({'images':entries,'rawTXT_identical':entries[0]['rawTXT_sha256']==entries[1]['rawTXT_sha256'],
                                 'exact_bbox_andclass_agreement':sum((exactA&exactB).values()),'sameclass_IoU_matching_agreement':agree})
        W=sum(x.width for x in panels);H=max(x.height for x in panels);sheet=Image.new('RGB',(W,H+48),'white');off=0;dd=ImageDraw.Draw(sheet)
        for z,e in zip(panels,entries):
            sheet.paste(z,(off,48));dd.text((off+5,5),f'{e["split"]} image_id={e["image_id"]}, GT={e["GT"]}',fill='black');dd.text((off+5,23),e['file_name'],fill='black');off+=z.width
        sheet.save(out/f'pair{gi}_two_label_versions.png')
    # Original native row outcomes from previously sealed fixed-match arrays.
    E=ROOT/'data/00_实验计划与证据索引';BASE=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2';PRIOR=ROOT/'具体问题重新筛选_20261027_v1'
    seal=read(PRIOR/'delivery_manifest.json');sh={z['path']:z['sha256'] for z in seal['outputs']}
    spec=importlib.util.spec_from_file_location('frozen_output_evaluator',HERE/'probe_LQE_effect.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    sources[str(HERE/'probe_LQE_effect.py')]=sha(HERE/'probe_LQE_effect.py')
    lq=read(HERE/'LQE_effect_attempt02/results.json');sources[str(HERE/'LQE_effect_attempt02/results.json')]=sha(HERE/'LQE_effect_attempt02/results.json')
    for name in ['coco.py','cocoeval.py','_mask.cp310-win_amd64.pyd']:
        p=M.ER/'pycocoeval'/name;sources[str(p)]=sha(p);assert sources[str(p)]==lq['sources'][str(p)]
    result['val_prediction_comparison']={};result['annotation_version_sensitivity']={}
    for seed in [1,2]:
        b=BASE/f'seed{seed}';p=b/'predictions.json';sources[str(p)]=sha(p);assert sources[str(p)]==lq['sources'][str(p)];pred=read(p)
        p=b/'prediction_sources.json';sources[str(p)]=sha(p);links=read(p);p=b/'image_manifest.json';sources[str(p)]=sha(p);imhash={x['image_id']:x['sha256'] for x in read(p)}
        for iid in [324,363]:
            r=next(x for x in rr if x['split']=='val' and x['image_id']==iid);assert imhash[iid]==r['bytes_sha256']
        p=PRIOR/f'06_seed{seed}_fixed_rank_arrays.npz';sources[str(p)]=sha(p);assert sources[str(p)]==sh[str(p)]
        labels={t:{} for t in [0,5]}
        with np.load(p) as z:
            for c in range(10):
                ids=z[f'a0_c{c}_row'];mat=z[f'a0_c{c}_matches'];ig=z[f'a0_c{c}_ignored']
                for t in [0,5]:
                    for row,match,ignore in zip(ids,mat[t],ig[t]):
                        if links[int(row)]['image_id'] in [324,363]:labels[t][int(row)]=(bool(match>0 and not ignore),bool(ignore),int(match)-1 if match>0 else None)
        rowkeys={iid:{(x['query'],x['class']):x['row'] for x in links if x['image_id']==iid} for iid in [324,363]}
        A,B=rowkeys[324],rowkeys[363];assert set(A)==set(B)
        common_cap={k for k in A if A[k] in labels[0] and B[k] in labels[0]}
        z1=np.load(b/'arrays/000324.npz');z2=np.load(b/'arrays/000363.npz')
        for pp in [b/'arrays/000324.npz',b/'arrays/000363.npz']:sources[str(pp)]=sha(pp);assert sources[str(pp)]==lq['sources'][str(pp)]
        cmp={'producer_original_image_SHA_equal':True,'full_export_pair_identities':len(A),'common_native_cap_identities':len(common_cap),
             'max_final_box_difference':float(np.max(np.abs(z1['layer_boxes'][-1]-z2['layer_boxes'][-1]))),
             'max_final_logits_difference':float(np.max(np.abs(z1['logits']-z2['logits']))),
             'max_export_score_difference':max(abs(pred[A[k]]['score']-pred[B[k]]['score']) for k in A),'by_IoU':{}}
        assert cmp['max_final_box_difference']==cmp['max_final_logits_difference']==cmp['max_export_score_difference']==0
        for t in [0,5]:
            cc=collections.Counter();detail=[]
            for k in sorted(common_cap):
                aa,ai,ag=labels[t][A[k]];bb,bi,bg=labels[t][B[k]]
                cat='ignored' if ai or bi else 'both_TP' if aa and bb else 'A_only_TP' if aa else 'B_only_TP' if bb else 'neither_TP'
                cc[cat]+=1;detail.append({'query':k[0],'class':k[1],'outcome':cat,'A_GT':ag,'B_GT':bg})
            cmp['by_IoU'][str(.5+t*.05)]={'counts':dict(cc),'denominator_common_cap_pairs':len(common_cap),
                                      'A_TP':sum(labels[t][A[k]][0] for k in common_cap),'A_GT':41,'B_TP':sum(labels[t][B[k]][0] for k in common_cap),'B_GT':40}
            save(out/f'seed{seed}_row_outcomes_IoU{50+t*5}.json',detail)
        result['val_prediction_comparison'][str(seed)]=cmp
        sensitivities={}
        for name,source_iid in [('A_for_both',324),('B_for_both',363)]:
            cf=copy.deepcopy(ds['val']);cf['annotations']=[a for a in cf['annotations'] if a['image_id'] not in [324,363]]
            new_id=max(a['id'] for a in ds['val']['annotations'])+1;newby={}
            for target_iid in [324,363]:
                newby[target_iid]=set()
                for a in anns['val'][source_iid]:
                    x=copy.deepcopy(a);x.update(id=new_id,image_id=target_iid);cf['annotations'].append(x);newby[target_iid].add(new_id);new_id+=1
            assert len(cf['annotations'])=={'A_for_both':38760,'B_for_both':38758}[name]
            for a in cf['annotations']:a['id']+=1
            print(json.dumps({'seed':seed,'annotation_scenario':name,'GT':len(cf['annotations']),'elapsed_s':time.perf_counter()-start}),flush=True)
            mm,hits=M.evaluate(pred,cf,f'seed{seed}_{name}',out);result['CPU_evaluator_calls']+=1
            base=lq['seeds'][str(seed)]['arms']['native']['metrics_pp']
            sensitivities[name]={'fullGT':len(cf['annotations']),'metrics_pp':mm,'delta_pp':{k:mm[k]-base[k] for k in ['AP','AP50','AP75','APs']},
                                 'per_alias':{str(iid):{'GT':len(ids),'TP50':len(ids&hits[0][0]),'TP75':len(ids&hits[0][5])} for iid,ids in newby.items()}}
        result['annotation_version_sensitivity'][str(seed)]={'original_GT':38759,'original_metrics_pp':lq['seeds'][str(seed)]['arms']['native']['metrics_pp'],'scenarios':sensitivities}
    assert all(sha(Path(p))==h for p,h in sources.items())
    result.update(status='COMPLETED',elapsed_seconds=time.perf_counter()-start,inputs_unchanged=True)
    save(out/'results.json',result);save(out/'outputs.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'comparison':result['val_prediction_comparison'],'sensitivity':{s:{k:v['delta_pp'] for k,v in z['scenarios'].items()} for s,z in result['annotation_version_sensitivity'].items()}},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        p=HERE/'identical_pixels_labels_attempt01/FAILED.txt';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc(),encoding='utf-8');raise
