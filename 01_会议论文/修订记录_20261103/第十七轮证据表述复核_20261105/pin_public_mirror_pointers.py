"""Pin the public mirror revision and retain the seven small LFS pointer bodies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
import json
import re

HERE = Path(__file__).resolve().parent
INITIAL = HERE/'public_mirror_pointer_audit.json'
OUTPUT = HERE/'public_mirror_pinned_evidence.json'
API = 'https://huggingface.co/api/datasets/Voxel51/VisDrone2019-DET/revision/main'

def fetch(url, limit):
    req = Request(url, headers={'User-Agent': 'Academic provenance audit (public metadata)'})
    with urlopen(req, timeout=15) as response:
        body = response.read(limit+1)
        assert len(body) <= limit
        return body, response.headers.get('ETag')

def main():
    assert not OUTPUT.exists()
    initial = json.loads(INITIAL.read_text(encoding='utf-8'))
    assert len(initial['records']) == 7 and all(r['status'] == 'GIT_LFS_SHA256' and r['same_as_local'] for r in initial['records'])
    metadata, etag = fetch(API, 4*1024*1024)
    parsed = json.loads(metadata)
    revision = parsed['sha']
    assert re.fullmatch(r'[0-9a-f]{40}', revision)
    def fetch_one(record):
        url = 'https://huggingface.co/datasets/Voxel51/VisDrone2019-DET/raw/' + revision + '/data/' + record['file']
        body, item_etag = fetch(url, 2048)
        text = body.decode('utf-8')
        match = re.search(r'^oid sha256:([0-9a-f]{64})$', text, re.M)
        assert match and match.group(1) == record['current_local_sha256'] == record['remote_sha256']
        assert sha256(body).hexdigest() == record['response_sha256']
        return {'file': record['file'], 'pinned_url': url, 'pointer_body': text,
                'pointer_sha256': sha256(body).hexdigest(), 'etag': item_etag,
                'declared_image_sha256': match.group(1), 'local_image_sha256': record['current_local_sha256']}
    with ThreadPoolExecutor(max_workers=7) as pool:
        records = list(pool.map(fetch_one, initial['records']))
    result = {'status': 'THIRD_PARTY_MIRROR_POINTERS_PINNED_AND_MATCH_LOCAL',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'revision': revision,
              'repository': 'https://huggingface.co/datasets/Voxel51/VisDrone2019-DET',
              'repository_metadata_sha256': sha256(metadata).hexdigest(), 'repository_metadata_etag': etag,
              'initial_audit_sha256': sha256(INITIAL.read_bytes()).hexdigest(), 'records': records,
              'limits': ['This is a third-party mirror; its pointers are metadata, not an authenticated official VisDrone archive.',
                         'No official source image was downloaded, no AP impact was calculated, and historical data identity remains unverified.',
                         'No image, annotation, checkpoint, or local data was uploaded; no model or evaluator was run.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'revision': revision, 'records': len(records)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
