"""Bounded acquisition of two source-only TRAIN clips; no model/evaluation."""
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path(__file__).parent / 'smot_source_metadata_20261029'
dest = root / 'source_train_witnesses'
dest.mkdir(exist_ok=True)
candidates = json.loads((root / 'small_train_source_candidates.json').read_text(encoding='utf-8'))
targets = [x for x in candidates if x['video'] in {'0042', '0044'}]

def fetch(item):
    r = dict(item, model_run=False, test_read=False)
    started = time.monotonic()
    path = dest / (item['video'] + '.zip')
    try:
        if path.is_file():
            data = path.read_bytes()
            r['status'] = 'existing_source_preserved'
        else:
            url = 'https://drive.google.com/uc?export=download&id=' + item['public_file_id']
            r['url'] = url
            with urllib.request.urlopen(url, timeout=10) as resp:
                r['content_type'] = resp.headers.get('Content-Type')
                chunks = []
                total = 0
                while True:
                    if time.monotonic() - started > 30:
                        raise TimeoutError('bounded source read deadline exceeded')
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > 20_000_000:
                        raise ValueError('source file exceeds per-clip read cap')
                data = b''.join(chunks)
            if not data.startswith(b'PK\x03\x04'):
                r['status'] = 'stopped_nonzip_no_warning_or_form_followed'
                r['response_bytes'] = len(data)
                return r
            path.write_bytes(data)
            r['status'] = 'source_zip_saved'
        r['bytes'] = len(data)
        r['sha256'] = hashlib.sha256(data).hexdigest()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            r['catalog'] = [{'name': i.filename, 'size': i.file_size} for i in z.infolist()]
    except Exception as exc:
        r['status'] = 'request_failed'
        r['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        r['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return r

with ThreadPoolExecutor(max_workers=2) as pool:
    receipts = list(pool.map(fetch, targets))
(dest / 'source_clip_receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding='utf-8')
for r in receipts:
    print(json.dumps({k: v for k, v in r.items() if k != 'catalog'}, ensure_ascii=False))
