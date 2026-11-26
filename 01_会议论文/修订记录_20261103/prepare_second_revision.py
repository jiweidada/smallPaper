from pathlib import Path
import hashlib,json,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
doc=ROOT/'01_会议论文/稿件/MPSMA-DEIM_conference_EN_revised.docx'
expected='a0ee9ccc377f22323a4b8a9059da613299a4b464c4d9fad75b14ca5e96ade1c4'
assert hashlib.sha256(doc.read_bytes()).hexdigest()==expected
backup=OUT/'第一轮留档';backup.mkdir(exist_ok=True)
for p in [doc]+[OUT/n for n in ['revise_manuscript.py','revision_patch.json','修订说明与未解决问题.txt','引用核查记录.txt','validation_report.json','visual_qa.json']]:
    q=backup/p.name
    if not q.exists():shutil.copy2(p,q)
    assert hashlib.sha256(q.read_bytes()).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest()

image_root=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\images')
asset_dir=OUT/'source_snapshot/dataset_examples';asset_dir.mkdir(exist_ok=True)
assets=[]
for name in ['0000001_02999_d_0000005.jpg','0000280_02201_d_0000623.jpg']:
    p=image_root/name;q=asset_dir/name
    if not q.exists():shutil.copy2(p,q)
    assets.append({'source':str(p),'snapshot':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'selection':'First and middle filenames of the sorted validation image list; illustrative inputs, no detections, pixels unmodified.'})
(OUT/'dataset_eval_audit/figure6_source_manifest.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2),encoding='utf-8')
toolkit_dir=OUT/'dataset_eval_audit/official_toolkit';toolkit_dir.mkdir(exist_ok=True)
sources=[]
for name,relative in [('evalDET.m','evalDET.m'),('dropObjectsInIgr.m','utils/dropObjectsInIgr.m')]:
    url='https://raw.githubusercontent.com/VisDrone/VisDrone2018-DET-toolkit/refs/heads/master/'+relative
    p=toolkit_dir/name
    if not p.exists():
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=20) as r:b=r.read(100000)
        p.write_bytes(b)
    sources.append({'url':url,'access_date':'2026-11-03','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Read-only source inspection; never executed.'})
(toolkit_dir/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print('First revision preserved; two unmodified source images and official evaluation source snapshots archived.')
