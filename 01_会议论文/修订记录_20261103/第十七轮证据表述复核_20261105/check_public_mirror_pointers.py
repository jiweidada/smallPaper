"""Read tiny public Git-LFS pointers for the six names and one control; no images uploaded."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import re

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'public_mirror_pointer_audit.json'
SOURCE = HERE/'train_val_byte_overlap_audit.json'
BASE = 'https://huggingface.co/datasets/Voxel51/VisDrone2019-DET/raw/main/data/'
CONTROL = '0000001_02999_d_0000005.jpg'

def get_pointer(name):
    url = BASE + name
    try:
        req = Request(url, headers={'User-Agent': 'Academic provenance audit (public file pointers)'})
        with urlopen(req, timeout=12) as response:
            payload = response.read(2049)
            if len(payload) > 2048:
                return {'file': name, 'status': 'RESPONSE_NOT_A_SMALL_POINTER', 'url': url}
            text = payload.decode('utf-8')
            match = re.search(r'^oid sha256:([0-9a-f]{64})$', text, re.M)
            return {'file': name, 'status': 'GIT_LFS_SHA256' if match else 'POINTER_WITHOUT_SHA256',
                    'url': url, 'remote_sha256': match.group(1) if match else None,
                    'response_sha256': sha256(payload).hexdigest(),
                    'etag': response.headers.get('ETag')}
    except (OSError, UnicodeError, ValueError, HTTPError, URLError) as exc:
        return {'file': name, 'status': 'UNAVAILABLE', 'url': url, 'exception_type': type(exc).__name__,
                'http_status': exc.code if isinstance(exc, HTTPError) else None}

def main():
    assert not OUTPUT.exists()
    audit = json.loads(SOURCE.read_text(encoding='utf-8'))
    current = {item['file_name']: item['sha256'] for split in ['train', 'validation'] for item in audit[split]['files']}
    names = [n for split in ['train', 'validation'] for members in audit[split]['within_split_identical_byte_groups'].values() for n in members]
    assert len(names) == 6 and CONTROL in current
    with ThreadPoolExecutor(max_workers=7) as pool:
        records = list(pool.map(get_pointer, names+[CONTROL]))
    for record in records:
        record['current_local_sha256'] = current[record['file']]
        record['same_as_local'] = (record.get('remote_sha256') == record['current_local_sha256']) if record.get('remote_sha256') else None
    output = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'source_audit_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
              'mirror': 'Voxel51/VisDrone2019-DET, a third-party Hugging Face mirror, main branch',
              'official_source': False, 'records': records,
              'limits': ['Only public LFS pointer text was requested, with a 2048-byte response limit; no image, label or user data was uploaded.',
                         'Mirror metadata is not proof of the official archive or the historical training data.',
                         'No model, training or evaluator ran.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for record in records:
        print(json.dumps(record, ensure_ascii=False))

if __name__ == '__main__':
    main()
