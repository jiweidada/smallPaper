"""Preserve author GT/ignore/evaluation assets and audit existing image coverage."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

root=Path('/24085403014/datasets_verified_20261017/UAVDT_official')
archive_path=root/'raw/UAV-benchmark-MOTD_v1.0.zip'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest()=='1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799'
target=root/'author_toolkit';target.mkdir(exist_ok=False)
manifest=[]
with zipfile.ZipFile(archive_path) as archive:
    names=archive.namelist()
    test_sequences=sorted({Path(name).stem for name in names if '/RES_DET/det_FRCNN/' in name and name.endswith('.txt')})
    assert len(test_sequences)==20
    selected=[name for name in names if ('/GT/' in name and name.endswith('.txt')) or
              ('/det_EVA/' in name and not name.endswith('/')) or name.endswith(('README.md','CalculateDetectionsPR.m','compOas.m'))]
    for name in selected:
        relative=Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        raw=archive.read(name);path=target/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(raw)
        manifest.append({'relative':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
gt_root=target/'UAV-benchmark-MOTD_v1.0/GT'
all_sequences=sorted(path.name.split('_gt_whole')[0] for path in gt_root.glob('*_gt_whole.txt'))
assert len(all_sequences)==50
assert len(list(gt_root.glob('*_gt_ignore.txt')))==50
train_sequences=sorted(set(all_sequences)-set(test_sequences));assert len(train_sequences)==30
coverage=[]
for split,seqs in [('train',train_sequences),('test',test_sequences)]:
    current=Path('/24085403014/DEIM/UAVDT_Datasets')/('UAVDT-'+split)/'images'
    present={}
    for path in current.glob('*.jpg'):
        seq,frame=path.stem.split('_',1);present.setdefault(seq,set()).add(int(frame))
    for seq in seqs:
        native={};flags=Counter();max_frame=0
        for kind in ['whole','ignore']:
            path=gt_root/(seq+'_gt_'+kind+'.txt')
            rows=[line.split(',') for line in path.read_text().splitlines() if line.strip()]
            native[kind]=len(rows)
            max_frame=max([max_frame]+[int(row[0]) for row in rows])
        missing=sorted(set(range(1,max_frame+1))-present.get(seq,set()))
        coverage.append({'split':split,'sequence':seq,'native_annotation_max_frame':max_frame,
                         'available_images':len(present.get(seq,set())),'missing_up_to_annotation_max':missing,
                         'positive_rows':native['whole'],'ignore_rows':native['ignore']})
receipt={'status':'native_annotations_and_official_toolkit_updated_image_package_pending',
         'author_images_link':'https://drive.google.com/open?id=1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc',
         'image_download_blocker':'Author Google Drive quota exceeded, confirmed anonymously; no bypass attempted.',
         'official_detection_evaluation':'VOC-style AP at overlap 0.7 per original paper; do not rename ordinary COCO AP50 as official score. Toolkit retained for exact ignore-rule audit.',
         'train_sequences':train_sequences,'test_sequences':test_sequences,'coverage':coverage,
         'missing_images_lower_bound':sum(len(row['missing_up_to_annotation_max']) for row in coverage),
         'native_whole_files':50,'native_ignore_files':50,'source_manifest':manifest,
         'ready_for_training':False,'old_converted_data_preserved':True}
with (root/'update_receipt.json').open('x') as handle:json.dump(receipt,handle,indent=2)
print(json.dumps(receipt))
