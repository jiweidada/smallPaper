"""Probe anonymous public author download endpoints without downloading large data."""
import datetime
import json
from pathlib import Path
import requests
from urllib.parse import urlsplit

sources = {
    'UAVDT_images': '1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc',
    'UAVDT_DET_MOT_toolkit': '19498uJd7T9w4quwnQEy62nibt3uyT9pq',
    'UAVDT_readme': '1FV7zNa_Z7ivir7Kjw3Cpp2SmeQSinij_',
}
result = {'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'probes': [], 'local_metadata': {}}
for label, file_id in sources.items():
    url = 'https://drive.usercontent.google.com/download?id=' + file_id + '&export=download&confirm=t'
    try:
        with requests.get(url, stream=True, timeout=(12, 20)) as response:
            prefix = next(response.iter_content(2048), b'')
            parsed = urlsplit(response.url)
            result['probes'].append({'label': label, 'source_public_id': file_id,
                'status': response.status_code, 'content_type': response.headers.get('Content-Type'),
                'content_length': response.headers.get('Content-Length'),
                'content_disposition': response.headers.get('Content-Disposition'),
                'resolved_origin': parsed.scheme + '://' + parsed.netloc + parsed.path,
                'is_zip_prefix': prefix[:4] == b'PK\x03\x04',
                'text_prefix': prefix.decode(errors='replace')[:1000] if 'text' in response.headers.get('Content-Type', '') else None})
    except Exception as exc:
        result['probes'].append({'label': label, 'error_type': type(exc).__name__, 'error': str(exc)[:400]})
for name in ['README.md', 'metafile.yaml']:
    path = Path('/24085403014/DEIM/datasets/tiny_person/OpenDataLab___TinyPerson') / name
    if path.is_file():
        result['local_metadata'][name] = path.read_text()[:3000]
print(json.dumps(result, ensure_ascii=False))
