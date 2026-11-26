"""Independent byte-for-byte check of all ZIP validation members and CPU bindings."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
DATA = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')
AUDIT = HERE/'public_val_archive_audit.json'
ZIP = HERE/'public_mirror_VisDrone2019-DET-val.zip'
OUTPUT = HERE/'independent_val_archive_verification.json'
VAL_MANIFEST = PAPER/'本地CPU核验_20261103/validation_image_manifest.json'
TXT_MANIFEST = REC/'dataset_eval_audit/val_original_file_hashes.json'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()

def main():
    assert not OUTPUT.exists()
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    assert audit['status'] == 'THIRD_PARTY_ARCHIVE_MEMBERS_COMPARED' and sha(ZIP) == audit['archive_sha256']
    assert len(audit['images']) == len(audit['labels']) == 548
    assert not audit['image_mismatches'] and not audit['label_mismatches']
    assert all(row['all_equal'] for row in audit['images']+audit['labels'])
    names = {'images': [x['name'] for x in audit['images']], 'annotations': [x['name'] for x in audit['labels']]}
    failures = []
    selected = {}
    with ZipFile(ZIP) as z:
        for section, filenames in names.items():
            assert len(filenames) == len(set(filenames)) == 548
            for name in filenames:
                suffix = '/'+section+'/'+name
                candidates = [item for item in z.namelist() if item.replace('\\', '/').endswith(suffix)]
                assert len(candidates) == 1
                remote_bytes = z.read(candidates[0])
                local_bytes = (DATA/section/name).read_bytes()
                if remote_bytes != local_bytes:
                    failures.append({'section':section, 'name':name})
                if name in {'0000022_00000_d_0000004.jpg','0000023_00000_d_0000008.jpg',
                            '0000022_00000_d_0000004.txt','0000023_00000_d_0000008.txt'}:
                    selected[name] = {'sha256':sha256(remote_bytes).hexdigest(), 'bytes':len(remote_bytes),
                                      'byte_equal':remote_bytes == local_bytes}
    assert not failures and len(selected) == 4
    assert selected['0000022_00000_d_0000004.jpg']['sha256'] == selected['0000023_00000_d_0000008.jpg']['sha256']
    assert selected['0000022_00000_d_0000004.txt']['sha256'] != selected['0000023_00000_d_0000008.txt']['sha256']
    val_sha, ann_sha = sha(VAL_MANIFEST), sha(DATA/'annotations/val.json')
    assert val_sha == audit['validation_image_manifest_sha256'] and sha(TXT_MANIFEST) == audit['original_label_manifest_sha256']
    receipts = []
    for name in ['N0','N4','S1']:
        path = PAPER/f'本地CPU核验_20261103/results/{name}/receipt.json'
        rec = json.loads(path.read_text(encoding='utf-8-sig'))
        assert rec['status']=='COMPLETED' and rec['completed_images']==548
        assert rec['image_manifest_sha256']==val_sha and rec['annotation_sha256']==ann_sha
        receipts.append({'run':name,'receipt_sha256':sha(path)})
    report = {'status':'ALL_CURRENT_VALIDATION_FILES_MATCH_PUBLIC_ARCHIVE_BYTES',
              'source_audit_sha256':sha(AUDIT), 'archive_sha256':sha(ZIP),
              'validation_image_manifest_sha256':val_sha, 'validation_coco_annotation_sha256':ann_sha,
              'original_text_manifest_sha256':sha(TXT_MANIFEST),
              'image_files_checked':548, 'original_text_files_checked':548, 'byte_mismatches':failures,
              'duplicate_pair_files':selected, 'cpu_reevaluation_receipts':receipts,
              'limits':['The archive is a third-party mirror, not an authenticated VisDrone-origin archive.',
                        'Matching current files and CPU receipts does not certify historical run data or quantify metric impact.',
                        'No model, evaluator, training or modification of dataset files was performed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status':report['status'], 'images':report['image_files_checked'],
                      'texts':report['original_text_files_checked'], 'mismatches':len(failures)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
