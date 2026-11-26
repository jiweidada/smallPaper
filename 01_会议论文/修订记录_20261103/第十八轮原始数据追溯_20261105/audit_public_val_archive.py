"""Compare current validation files with a pinned third-party public ZIP, without extraction."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote, urlsplit
from zipfile import ZipFile
import json
import os
import re

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
REC = HERE.parent
DATA = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')
REPO = 'HoangTrungNguyen/VisDrone2019_training'
FILE = 'data/VisDrone2019-DET-val.zip'
MAX_ARCHIVE = 200*1024*1024
OUTPUT = HERE/'public_val_archive_audit.json'
ARCHIVE = HERE/'public_mirror_VisDrone2019-DET-val.zip'
PARTIAL = HERE/'public_mirror_VisDrone2019-DET-val.zip.part'
VAL_MANIFEST = PAPER/'本地CPU核验_20261103/validation_image_manifest.json'
TXT_MANIFEST = REC/'dataset_eval_audit/val_original_file_hashes.json'

def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def get(url, limit=4*1024*1024):
    req = Request(url, headers={'User-Agent': 'Academic data provenance audit'})
    with urlopen(req, timeout=30) as response:
        body = response.read(limit+1)
        if len(body) > limit:
            raise ValueError('metadata response exceeded limit')
        return body

def entries(z, section, suffix):
    output = {}
    for entry in z.infolist():
        parts = entry.filename.replace('\\', '/').split('/')
        if len(parts) >= 2 and parts[-2] == section and parts[-1].lower().endswith(suffix) and not entry.is_dir():
            if parts[-1] in output:
                raise ValueError('duplicate archive basename')
            if entry.file_size > 5*1024*1024:
                raise ValueError('archive member exceeds size limit')
            output[parts[-1]] = entry
    return output

def member_sha(z, item):
    h = sha256()
    with z.open(item) as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    assert not any(p.exists() for p in [OUTPUT, ARCHIVE, PARTIAL])
    record = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'repository': 'https://huggingface.co/datasets/'+REPO,
              'mirror_claim': 'The repository card says the validation ZIP came via the Ultralytics assets mirror; this is not an authenticated official VisDrone archive.',
              'max_download_bytes': MAX_ARCHIVE, 'private_data_uploaded': False,
              'models_run': 0, 'evaluator_runs': 0, 'optimizer_updates': 0}
    try:
        api_body = get('https://huggingface.co/api/datasets/'+REPO+'/revision/main')
        revision = json.loads(api_body)['sha']
        assert re.fullmatch(r'[0-9a-f]{40}', revision)
        raw_url = 'https://huggingface.co/datasets/'+REPO+'/raw/'+revision+'/'+FILE
        pointer = get(raw_url, 2048).decode('utf-8')
        pointer_sha = re.search(r'^oid sha256:([0-9a-f]{64})$', pointer, re.M).group(1)
        pointer_size = int(re.search(r'^size (\d+)$', pointer, re.M).group(1))
        assert 0 < pointer_size <= MAX_ARCHIVE
        record.update({'revision': revision, 'api_response_sha256': sha256(api_body).hexdigest(),
                       'pointer_url': raw_url, 'pointer_body': pointer,
                       'pointer_sha256': sha256(pointer.encode('utf-8')).hexdigest(),
                       'declared_archive_sha256': pointer_sha, 'declared_archive_bytes': pointer_size})
        url = 'https://huggingface.co/datasets/'+REPO+'/resolve/'+revision+'/'+FILE
        req = Request(url, headers={'User-Agent': 'Academic data provenance audit'})
        with urlopen(req, timeout=60) as response, PARTIAL.open('xb') as out:
            host = (urlsplit(response.geturl()).hostname or '').lower()
            assert host == 'huggingface.co' or host.endswith('.hf.co') or host.endswith('.huggingface.co')
            first = response.read(4096)
            assert first.startswith(b'PK\x03\x04')
            count = len(first)
            h = sha256(first)
            out.write(first)
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_ARCHIVE:
                    raise ValueError('archive exceeded size limit')
                out.write(chunk)
                h.update(chunk)
        assert count == pointer_size and h.hexdigest() == pointer_sha
        os.replace(PARTIAL, ARCHIVE)
        image_manifest = json.loads(VAL_MANIFEST.read_text(encoding='utf-8-sig'))
        text_manifest = json.loads(TXT_MANIFEST.read_text(encoding='utf-8-sig'))
        images = {x['file_name']: x['sha256'] for x in image_manifest}
        labels = dict(text_manifest)
        assert len(images) == len(image_manifest) == 548 and len(labels) == len(text_manifest) == 548
        with ZipFile(ARCHIVE) as z:
            archive_images = entries(z, 'images', '.jpg')
            archive_labels = entries(z, 'annotations', '.txt')
            assert set(images) == set(archive_images) and set(labels) == set(archive_labels)
            image_results = []
            for name, archived_hash in images.items():
                local = digest(DATA/'images'/name)
                remote = member_sha(z, archive_images[name])
                image_results.append({'name': name, 'manifest_sha256': archived_hash, 'local_sha256': local,
                                      'archive_sha256': remote, 'all_equal': archived_hash == local == remote})
            label_results = []
            for name, archived_hash in labels.items():
                local = digest(DATA/'annotations'/name)
                remote = member_sha(z, archive_labels[name])
                label_results.append({'name': name, 'manifest_sha256': archived_hash, 'local_sha256': local,
                                      'archive_sha256': remote, 'all_equal': archived_hash == local == remote})
        record.update({'status': 'THIRD_PARTY_ARCHIVE_MEMBERS_COMPARED',
                       'archive_path': str(ARCHIVE), 'archive_bytes': count, 'archive_sha256': digest(ARCHIVE),
                       'validation_image_manifest_sha256': digest(VAL_MANIFEST),
                       'original_label_manifest_sha256': digest(TXT_MANIFEST),
                       'images_compared': len(image_results), 'original_label_files_compared': len(label_results),
                       'image_mismatches': [x for x in image_results if not x['all_equal']],
                       'label_mismatches': [x for x in label_results if not x['all_equal']],
                       'images': image_results, 'labels': label_results,
                       'limits': ['The downloadable ZIP is a third-party mirror, not an authenticated official archive.',
                                  'No data from the original training process, model weights, or predictions were checked here.',
                                  'No image labels were changed and no model or evaluator was run.']})
    except Exception as exc:
        record.update({'status': 'THIRD_PARTY_ARCHIVE_CHECK_INCOMPLETE', 'exception_type': type(exc).__name__,
                       'http_status': getattr(exc, 'code', None),
                       'partial_file_retained': str(PARTIAL) if PARTIAL.exists() else None,
                       'archive_file_retained': str(ARCHIVE) if ARCHIVE.exists() else None})
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(record, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({k: record.get(k) for k in ['status','revision','archive_bytes','archive_sha256','images_compared',
          'original_label_files_compared','exception_type','http_status','partial_file_retained']}, ensure_ascii=False))
    if 'image_mismatches' in record:
        print(json.dumps({'image_mismatches':len(record['image_mismatches']), 'label_mismatches':len(record['label_mismatches'])}))

if __name__ == '__main__':
    main()
