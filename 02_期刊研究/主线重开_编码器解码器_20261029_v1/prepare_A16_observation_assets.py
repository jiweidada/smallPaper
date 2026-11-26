"""Select new images using annotation metadata only; pack already-known native caches."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile
import numpy as np
from PIL import Image
from a16_spatial_operator import pack

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A16_实际特征空间变化审计_v1'
OLD=ROOT/'A12_原生边界分布诊断_v1'
A15=ROOT/'A15_固定query的采样中心干预_v1'
SOURCE=Path('D:/数据集/visdrone2019/VisDrone2019-DET-val')
assert not (OUT/'asset_selection.json').exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pool=json.loads((OLD/'annotation_pool.json').read_text(encoding='utf-8'))
previous=json.loads((OLD/'asset_selection.json').read_text(encoding='utf-8'))
old_prefixes={c['prefix'] for c in previous['cases']}
selected=[];seen=set(old_prefixes)
for case in sorted(pool,key=lambda r:r['file']):
    if case['tiny_GT']<10 or case['prefix'] in seen:continue
    if len(selected)>=32:break
    row=dict(case);image=SOURCE/'images'/case['file'];ann=SOURCE/'annotations'/(image.stem+'.txt')
    with Image.open(image) as im:assert im.size==(case['width'],case['height'])
    w,h=case['width'],case['height'];valid=[]
    for index,line in enumerate(ann.read_text(encoding='utf-8-sig').splitlines()):
        x=[float(v.strip()) for v in line.split(',') if v.strip()]
        if len(x)<6 or x[4]<=0 or not 1<=int(x[5])<=10:continue
        l,t,bw,bh=x[:4];x1=max(0.,min(w,l));y1=max(0.,min(h,t))
        x2=max(0.,min(w,l+bw));y2=max(0.,min(h,t+bh))
        if x2<=x1 or y2<=y1:continue
        box=[800*x1/w,800*y1/h,800*x2/w,800*y2/h]
        size=float(np.sqrt((box[2]-box[0])*(box[3]-box[1])))
        valid.append({'uid':ann.stem+'_'+str(index),'class_id':int(x[5]),'box800':box,'size800':size})
    assert len(valid)==case['valid_GT'] and sum(x['size800']<8 for x in valid)==case['tiny_GT']
    row.update(GT=valid,image_sha256=sha(image),annotation_sha256=sha(ann))
    selected.append(row);seen.add(case['prefix'])
assert len(selected)==25 and len(seen-old_prefixes)==25
archive=OUT/'new_annotation_selected_sources.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for c in selected:
        tar.add(SOURCE/'images'/c['file'],arcname='images/'+c['file'])
        ann=Path(c['file']).stem+'.txt';tar.add(SOURCE/'annotations'/ann,arcname='annotations/'+ann)
write(OUT/'asset_selection.json',{'utc':datetime.now(timezone.utc).isoformat(),
      'status':'ANNOTATION_ONLY_NEW_PREFIX_SELECTION_BEFORE_NEW_MODEL_PREDICTIONS',
      'rule':'Official val sorted filenames;>=10validGT sqrtarea<8at800square;exclude12A12prefixes;onepernewprefix;cap32.25eligibleprefixesexist.',
      'scope':'Exploratory extension,not independent blind test and not claimed unseen by author checkpoint selection.',
      'cases':selected,'selected_images':len(selected),'selected_prefixes':25,
      'selected_tiny_GT':sum(c['tiny_GT'] for c in selected),
      'old_prefixes':sorted(old_prefixes),'new_full_forward_cap':26,
      'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive)})
dest=OUT/'old_patches';dest.mkdir()
with np.load(OUT/'final_sampler_state.npz',allow_pickle=False) as f:state={k:f[k] for k in f.files}
inputs=[]
for index,case in enumerate(previous['cases']):
    capture=A15/'captures'/('case%02d.npz'%index);pred=A15/'predictions'/('case%02d.npz'%index)
    with np.load(capture,allow_pickle=False) as c,np.load(pred,allow_pickle=False) as p:
        mq=p['match_q'];mg=p['match_gt'];use=p['gt_sizes800'][mg]<8;qs=mq[use];gs=mg[use]
        packed=pack(c['ca_query'][qs],c['reference'][qs],tuple(c['value_%d'%l] for l in range(4)),c['spatial_shapes'],state)
        packed.update({'query_indices':qs,'gt_indices':gs,'raw_boxes':p['raw_boxes'],'raw_logits':p['raw_logits'],
                       'gt_boxes800':p['gt_boxes800'],'gt_labels':p['gt_labels'],'gt_sizes800':p['gt_sizes800'],
                       'gt_uids':p['gt_uids'],'prefix':np.asarray(case['prefix']),'image_filename':np.asarray(case['file'])})
    output=dest/('case%02d.npz'%index);np.savez_compressed(output,**packed)
    inputs.append({'case':index,'capture_sha256':sha(capture),'prediction_sha256':sha(pred),
                   'patch_file':str(output.relative_to(OUT)),'patch_sha256':sha(output),'tiny':len(qs)})
write(OUT/'old_patch_packing_receipt.json',{'status':'PACKED_EXISTING_NATIVE_VALUES_NO_RESPONSE_METRICS',
      'new_full_model_forwards':0,'source_operator_calls':0,'main_model_updates':0,'files':inputs})
print(json.dumps({'status':'ASSETS_PREPARED_NO_FEATURE_RESPONSE_METRICS','new_images':25,
      'new_tiny_GT':sum(c['tiny_GT'] for c in selected),'old_packed_tiny':sum(x['tiny'] for x in inputs),
      'archive_bytes':archive.stat().st_size,'new_forward_cap':26}))
