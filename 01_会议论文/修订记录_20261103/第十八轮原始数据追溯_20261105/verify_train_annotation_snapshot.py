"""Independently verify the four range-read label snapshots against existing local data."""
from hashlib import sha256
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
DATA = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations')
AUDIT = HERE/'train_annotation_range_audit.json'
OUTPUT = HERE/'independent_train_annotation_verification.json'
MIRROR = REC/'第十七轮证据表述复核_20261105/public_mirror_pinned_evidence.json'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists()
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    mirror = json.loads(MIRROR.read_text(encoding='utf-8'))
    assert audit['status']=='FOUR_TRAIN_ANNOTATIONS_COMPARED_BY_HTTP_RANGE' and audit['all_four_match_local']
    assert len(audit['annotations'])==4 and not audit['full_archive_downloaded']
    checks = []
    for row in audit['annotations']:
        snapshot = HERE/'public_train_annotation_snapshot'/row['name']
        local = DATA/row['name']
        assert snapshot.read_bytes() == local.read_bytes()
        assert sha(snapshot) == sha(local) == row['sha256'] == row['local_sha256']
        image_name = Path(row['name']).stem+'.jpg'
        matches = [x for x in mirror['records'] if x['file']==image_name]
        assert len(matches)==1 and matches[0]['declared_image_sha256']==matches[0]['local_image_sha256']
        checks.append({'file':row['name'], 'label_sha256':sha(snapshot), 'byte_equal':True,
                       'separate_mirror_image_pointer_sha256':matches[0]['declared_image_sha256']})
    report = {'status':'TRAIN_LABEL_RANGE_SNAPSHOTS_MATCH_LOCAL', 'range_audit_sha256':sha(AUDIT),
              'separate_image_mirror_evidence_sha256':sha(MIRROR), 'labels_checked':len(checks), 'checks':checks,
              'limits':['The four TXT records and four image LFS pointers are from two third-party mirrors; neither is an authenticated official archive.',
                        'The image bytes inside the partially read training ZIP were not checked. Its full archive hash was not recalculated.',
                        'The historical training input identity and effect on model metrics remain unverified.']}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as out:
        json.dump(report,out,ensure_ascii=False,indent=2)
        out.write('\n')
    print(json.dumps({'status':report['status'],'labels_checked':report['labels_checked']},ensure_ascii=False))

if __name__=='__main__':
    main()
