"""Read existing annotations and saved evaluator metadata, never run evaluation."""
from pathlib import Path
import collections, hashlib, json, zipfile, pickletools
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'dataset_eval_audit'; OUT.mkdir(exist_ok=True)
DEIM=Path(r'C:\Users\jiwei\Desktop\DEIM')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={}
for split,fn in [('train','train.json'),('val','val.json'),('test-dev','test.json')]:
    directory=DEIM/'dataset_visdrone'/f'VisDrone2019-DET-{split}'
    path=directory/'annotations'/fn
    data=json.loads(path.read_text(encoding='utf-8'))
    images={x['id']:x for x in data['images']}
    raw=collections.Counter();scores=collections.Counter();labels=collections.Counter()
    raw_by_cat_score=collections.Counter();raw_hashes=[]; raw_count=0
    for p in sorted((directory/'annotations').glob('*.txt')):
        raw_hashes.append((p.name,sha(p)))
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            x=[float(v) for v in line.strip().rstrip(',').split(',')]
            raw[int(x[5])]+=1;scores[int(x[4])]+=1;raw_by_cat_score[f'cat{int(x[5])}_score{int(x[4])}']+=1;raw_count+=1
    label_paths=sorted((directory/'labels').glob('*.txt'))
    label_count=0
    for p in label_paths:
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():labels[int(line.split()[0])]+=1;label_count+=1
    ann=data['annotations'];img_names={x['file_name'] for x in images.values()}
    ann_ids=[x['id'] for x in ann]
    rec={
      'annotation_path':str(path),'annotation_sha256':sha(path),'images':len(images),'annotations':len(ann),
      'categories':data['categories'],'category_counts':dict(collections.Counter(x['category_id'] for x in ann)),
      'iscrowd_counts':dict(collections.Counter(str(x.get('iscrowd','missing')) for x in ann)),
      'ignore_counts':dict(collections.Counter(str(x.get('ignore','missing')) for x in ann)),
      'unknown_category_ids':sorted(set(x['category_id'] for x in ann)-set(x['id'] for x in data['categories'])),
      'duplicate_image_ids':len(data['images'])-len(images),'duplicate_annotation_ids':len(ann_ids)-len(set(ann_ids)),
      'missing_image_files':sum(not (directory/'images'/name).is_file() for name in img_names),
      'nonpositive_bbox_count':sum(x['bbox'][2]<=0 or x['bbox'][3]<=0 for x in ann),
      'original_txt_files':len(raw_hashes),'original_objects':raw_count,'original_category_counts':dict(raw),
      'original_score_counts':dict(scores),'original_category_and_score_counts':dict(raw_by_cat_score),
      'original_files_manifest_sha256':hashlib.sha256(json.dumps(raw_hashes).encode()).hexdigest(),
      'yolo_label_files':len(label_paths),'yolo_labels':label_count,'yolo_category_counts':dict(labels),
      'scope':'Current files at paths retained in run args; historical immutability is unproven. No predictions or accuracy recalculation.'}
    report[split]=rec
    (OUT/f'{split}_original_file_hashes.json').write_text(json.dumps(raw_hashes,indent=2),encoding='utf-8')
    print(split,json.dumps({k:rec[k] for k in ['images','annotations','iscrowd_counts','ignore_counts','original_txt_files','original_objects','original_category_counts','original_score_counts','yolo_labels']},ensure_ascii=True),flush=True)
(OUT/'annotation_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
runs=json.loads((OUT.parent/'run_audit.json').read_text(encoding='utf-8'))
probes=[]
paths=[Path(a['source_dir'])/'eval'/'latest.pth' for a in runs.values()]
paths.append(DEIM/'outputs/deim_hgnetv2_s_MSCA_EnhancedStem_SiLU_320_eval/eval.pth')
for p in paths:
    r={'path':str(p),'exists':p.exists()}
    if p.exists():
        r['sha256']=sha(p)
        with zipfile.ZipFile(p) as z:
            n=next(n for n in z.namelist() if n.endswith('/data.pkl'))
            b=z.read(n)
        r['globals']=sorted(set(str(arg) for op,arg,pos in pickletools.genops(b) if op.name=='GLOBAL'))
        r['short_strings']=list(dict.fromkeys(str(arg) for op,arg,pos in pickletools.genops(b) if op.name in ['BINUNICODE','SHORT_BINUNICODE'] and len(str(arg))<100))
        print('eval_pickle_globals',str(p),r['globals'],flush=True)
    probes.append(r)
(OUT/'saved_eval_probe.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2),encoding='utf-8')
