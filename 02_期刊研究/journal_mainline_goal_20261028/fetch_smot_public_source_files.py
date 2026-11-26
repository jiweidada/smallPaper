"""Fetch small public source annotations and license. Never run downloaded code."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

out = Path(__file__).parent / 'smot_source_metadata_20261029'
out.mkdir(exist_ok=True)
targets = [
    ('annotations.zip', '15bgHpeHdT1AOR7pezWZ7YpMse4zvGnCt', 5_000_000),
    ('LICENSE.txt', '1nkqVs7r5_-rgPa4WGw97J47MYRKdfFc9', 16_384),
    ('author_README.txt', '1tHr6ne5wXz88EDW3JpCjhECrWUCOfJzk', 16_384),
]
receipts = []
for name, file_id, cap in targets:
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    r = {'name': name, 'source_file_id': file_id, 'url': url, 'read_cap_bytes': cap,
         'started_unix': time.time(), 'model_execution': False}
    if (out / name).is_file():
        existing = (out / name).read_bytes()
        r['status'] = 'existing_source_preserved'
        r['bytes'] = len(existing)
        r['sha256'] = hashlib.sha256(existing).hexdigest()
        receipts.append(r)
        print(json.dumps(r, ensure_ascii=False))
        continue
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SourceMetadataReview/1.0'})
        with urllib.request.urlopen(req, timeout=12) as resp:
            r['http_status'] = resp.status
            r['content_type'] = resp.headers.get('Content-Type')
            data = resp.read(cap + 1)
            r['bytes_read'] = len(data)
            if len(data) > cap:
                r['status'] = 'stopped_payload_exceeds_cap'
            elif 'text/html' in (r['content_type'] or ''):
                r['status'] = 'stopped_html_response_no_confirmation_followed'
            else:
                (out / name).write_bytes(data)
                r['sha256'] = hashlib.sha256(data).hexdigest()
                r['status'] = 'source_file_saved'
    except Exception as exc:
        r['status'] = 'request_failed'
        r['error'] = type(exc).__name__ + ': ' + str(exc)
    r['elapsed_seconds'] = round(time.time() - r['started_unix'], 3)
    receipts.append(r)
    print(json.dumps(r, ensure_ascii=False))
(out / 'source_files_receipt.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding='utf-8')
