"""Verify raw annotation bytes and GT identities;no prediction metrics."""
from pathlib import Path
import hashlib,json,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;N=R.parent;E=N.parent
OLD=E/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    assert not (R/'raw_annotations_verified.json').exists()
    cfg=read(N/'A30_YOLO26双头原生读出核查_v1'/'native_dual_head_protocol_FROZEN.json');manifest=read(OLD/'dataset_source_manifest.json');lookup={x['file_name']:x for x in manifest['files']};base=Path(manifest['source']);gt=read(Path(cfg['local_raw_root'])/'dataset/diagnostic_gt_orig.json');existing={x['stable_uid']:x for x in gt['annotations'] if not x['iscrowd']}
    rows=[];valid=set()
    for c in cfg['cases']:
        p=base/'annotations'/(Path(c['file_name']).stem+'.txt');data=p.read_bytes();h=hashlib.sha256(data).hexdigest();assert h==lookup[c['file_name']]['annotations']['sha256']
        annotations=[]
        for no,line in enumerate(data.decode('utf-8-sig').splitlines(),1):
            if not line.strip():continue
            values=[float(x.strip()) for x in line.split(',')[:8]];assert len(values)==8
            annotations.append({'source_row':no,'values':values})
            if values[4]>0 and 1<=values[5]<=10:
                uid=c['file_name']+':'+str(no);assert uid in existing
                assert existing[uid]['category_id']==int(values[5])-1 and existing[uid]['bbox']==values[:4]
                valid.add(uid)
        rows.append({'index':c['index'],'image_id':c['image_id'],'file_name':c['file_name'],'width':c['width'],'height':c['height'],'geometry':c['geometry'],'raw_annotation_file':str(p),'raw_annotation_sha256':h,'rows':annotations})
    assert valid==set(existing) and len(valid)==4491
    out={'status':'RAW100_ANNOTATION_BYTES_AND4491_VALID_GT_IDENTITIES_VERIFIED','scope':'Annotation preparation only,no ignored-region filtering or prediction results evaluated yet.','cases':rows,'original_valid_GT':len(valid),'category_mapping':'official1..10 ->YOLO0..9;class0ignore,class11unscored','new_model_calls':0}
    with (R/'raw_annotations_verified.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':out['status'],'cases':len(rows),'valid_GT':len(valid),'new_model_calls':0}))
if __name__=='__main__':main()
