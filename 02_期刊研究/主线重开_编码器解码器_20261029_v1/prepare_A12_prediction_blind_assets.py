"""Annotation-only selection and local source packaging; no image model or prediction."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math
import tarfile
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A12_原生边界分布诊断_v1'
SOURCE=Path('D:/数据集/visdrone2019/VisDrone2019-DET-val')
assert not OUT.exists()
OUT.mkdir()
rows=[]
all_cases={}
for annotation in sorted((SOURCE/'annotations').glob('*.txt')):
    image=SOURCE/'images'/(annotation.stem+'.jpg')
    assert image.exists()
    with Image.open(image) as im: w,h=im.size
    valid=[]
    for line_index,line in enumerate(annotation.read_text(encoding='utf-8-sig').splitlines()):
        x=[float(v.strip()) for v in line.split(',') if v.strip()]
        if len(x)<6 or x[4]<=0 or not 1<=int(x[5])<=10: continue
        left,top,width,height=x[:4]
        x1=max(0.,min(float(w),left));y1=max(0.,min(float(h),top))
        x2=max(0.,min(float(w),left+width));y2=max(0.,min(float(h),top+height))
        if x2<=x1 or y2<=y1: continue
        box=[800*x1/w,800*y1/h,800*x2/w,800*y2/h]
        size=math.sqrt((box[2]-box[0])*(box[3]-box[1]))
        valid.append({'uid':annotation.stem+'_'+str(line_index),'class_id':int(x[5]),'box800':box,'size800':size})
    row={'file':image.name,'prefix':annotation.stem.split('_')[0],'width':w,'height':h,'valid_GT':len(valid),'tiny_GT':sum(v['size800']<8 for v in valid)}
    rows.append(row);all_cases[image.name]=valid
eligible=[r for r in rows if r['tiny_GT']>=10]
selected=[];prefix_counts={}
for limit in (1,2):
    for r in eligible:
        if r in selected or prefix_counts.get(r['prefix'],0)>=limit:continue
        if len(selected)>=12:break
        selected.append(dict(r));prefix_counts[r['prefix']]=prefix_counts.get(r['prefix'],0)+1
    if len(selected)>=12:break
for r in selected:
    r['GT']=all_cases[r['file']]
    image=SOURCE/'images'/r['file'];ann=SOURCE/'annotations'/(Path(r['file']).stem+'.txt')
    r['image_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
    r['annotation_sha256']=hashlib.sha256(ann.read_bytes()).hexdigest()
archive=OUT/'prediction_blind_val_sources.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for r in selected:
        t.add(SOURCE/'images'/r['file'],arcname='images/'+r['file'])
        name=Path(r['file']).stem+'.txt'
        t.add(SOURCE/'annotations'/name,arcname='annotations/'+name)
p={'utc':datetime.now(timezone.utc).isoformat(),'status':'PREDICTION_BLIND_ASSET_SELECTION_COMPLETE_A12_RUNNER_NOT_IMPLEMENTED','source':str(SOURCE),'split':'Official val,not train-derivedD0 dev300. Not claimed to be a blind test or unseen in author checkpoint selection.','source_pool_images':len(rows),'eligible_images':len(eligible),'selection_rule':'Sort filenames lexically;eligible>=10 validGT with sqrt area<8 at800-square. First pass one per prefix;second pass at mosttwo per prefix;stopat12. No detector predictions used.','selected_images':len(selected),'selected_prefixes':len(prefix_counts),'selected_tiny_GT':sum(r['tiny_GT'] for r in selected),'validity_minimum_for_future_gate':{'images_with_pairs':8,'total_tiny_GT':120,'good_bad_edge_pairs':200},'cases':selected,'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'future_forward_cap':len(selected)+1,'future_budget_reason':'One uninstrumented native identity control for first image plusone instrumented native pass per selected image;no warmups or interventions.','executed_full_model_forwards':0,'main_detector_updates':0,'new_fits':0,'mainline_validated':False,'remaining_before_execution':['Finalize source-faithful capture and matching implementation','Freeze complete code/data/protocol manifest','Verify identity and decoder distribution-to-box reconstruction gates'],'current_cumulative_full_model_forwards':200}
(OUT/'asset_selection.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'annotation_pool.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:p[k] for k in ('status','source_pool_images','eligible_images','selected_images','selected_prefixes','selected_tiny_GT','future_forward_cap','archive_bytes','executed_full_model_forwards')},ensure_ascii=False))
