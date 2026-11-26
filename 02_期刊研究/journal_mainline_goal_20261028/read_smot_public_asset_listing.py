"""Read a public source folder listing; no media/model download or execution."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

out = Path(__file__).parent / 'smot_source_metadata_20261029'
out.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--folder-id', default='1Y1J13W6VlgDh-L28n_mVbs7HIfo_Hv5s')
parser.add_argument('--prefix', default='public_folder')
args = parser.parse_args()
if not args.prefix.replace('_', '').isalnum():
    raise ValueError('simple metadata filename prefix required')
url = 'https://drive.google.com/drive/folders/' + args.folder_id + '?usp=sharing'
receipt = {'url': url, 'purpose': 'public train asset listing only',
           'read_cap_bytes': 2097152, 'started_unix': time.time(),
           'model_run': False, 'test_image_analysis': False}
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'SourceMetadataReview/1.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        receipt['http_status'] = resp.status
        receipt['final_url'] = resp.geturl()
        receipt['headers'] = {k: resp.headers[k] for k in ('Content-Type', 'Content-Length', 'Last-Modified') if k in resp.headers}
        payload = resp.read(receipt['read_cap_bytes'] + 1)
        receipt['bytes_read'] = len(payload)
        if len(payload) > receipt['read_cap_bytes']:
            receipt['status'] = 'stopped_body_exceeds_cap'
        else:
            (out / (args.prefix + '_listing.html')).write_bytes(payload)
            receipt['status'] = 'listing_saved_not_yet_interpreted'
except Exception as exc:
    receipt['status'] = 'request_failed'
    receipt['error'] = type(exc).__name__ + ': ' + str(exc)
receipt['elapsed_seconds'] = round(time.time() - receipt['started_unix'], 3)
(out / (args.prefix + '_receipt.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k != 'headers'}, ensure_ascii=False, indent=2))
