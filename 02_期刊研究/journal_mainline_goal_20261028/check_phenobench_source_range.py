"""Read bounded source metadata only. Does not run any detector or evaluator."""
import json
import time
import urllib.request
from pathlib import Path

out = Path(__file__).parent / 'phenobench_source_metadata_20261029'
out.mkdir(exist_ok=True)
url = 'https://www.phenobench.org/data/PhenoBench-v110.zip'
receipt = {'url': url, 'purpose': 'source archive metadata only',
           'request_range': 'bytes=-65536', 'max_read_bytes': 65537,
           'started_unix': time.time(), 'model_or_evaluation_run': False}
try:
    req = urllib.request.Request(url, headers={'Range': 'bytes=-65536',
                                             'User-Agent': 'SourceMetadataReview/1.0'})
    with urllib.request.urlopen(req, timeout=12) as resp:
        receipt['http_status'] = resp.status
        receipt['headers'] = dict(resp.headers)
        if resp.status != 206:
            receipt['status'] = 'stopped_no_range_support_no_body_read'
        else:
            payload = resp.read(65537)
            receipt['bytes_read'] = len(payload)
            if len(payload) <= 65536:
                (out / 'archive_tail.bin').write_bytes(payload)
                receipt['status'] = 'bounded_tail_saved'
            else:
                receipt['status'] = 'stopped_unexpected_payload_size'
except Exception as exc:
    receipt['status'] = 'request_failed'
    receipt['error'] = type(exc).__name__ + ': ' + str(exc)
receipt['elapsed_seconds'] = round(time.time() - receipt['started_unix'], 3)
(out / 'archive_range_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
