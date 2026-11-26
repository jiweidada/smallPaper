"""Compare public release-asset metadata with the already downloaded mirror ZIP."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'ultralytics_release_metadata.json'
AUDIT = HERE/'public_val_archive_audit.json'
API = 'https://api.github.com/repos/ultralytics/assets/releases/tags/v0.0.0'
TARGET = 'VisDrone2019-DET-val.zip'

def get_json(url):
    request = Request(url, headers={'User-Agent':'Academic data source provenance audit', 'Accept':'application/vnd.github+json'})
    with urlopen(request, timeout=20) as response:
        data = response.read(8*1024*1024+1)
        if len(data) > 8*1024*1024:
            raise ValueError('metadata exceeds limit')
        return json.loads(data), sha256(data).hexdigest()

def main():
    assert not OUTPUT.exists()
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    assert audit['status'] == 'THIRD_PARTY_ARCHIVE_MEMBERS_COMPARED'
    result = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'endpoint':API,
              'local_mirror_audit_sha256':sha256(AUDIT.read_bytes()).hexdigest(),
              'mirror_archive_sha256':audit['archive_sha256'], 'mirror_archive_bytes':audit['archive_bytes'],
              'request_scope':'Public GitHub release metadata only; no archive was downloaded from GitHub.',
              'models_run':0, 'evaluator_runs':0}
    try:
        release, response_hash = get_json(API)
        result['release_response_sha256'] = response_hash
        result['release_tag'] = release['tag_name']
        matches = [a for a in release.get('assets', []) if a.get('name') == TARGET]
        if not matches:
            assets_api = release['assets_url']
            for page in range(1, 11):
                assets, _ = get_json(assets_api+'?per_page=100&page='+str(page))
                matches.extend(a for a in assets if a.get('name') == TARGET)
                if matches or len(assets) < 100:
                    break
        if len(matches) != 1:
            result.update({'status':'ASSET_NOT_UNIQUELY_FOUND', 'matching_assets':len(matches)})
        else:
            asset = matches[0]
            declared = asset.get('digest')
            result.update({'status':'RELEASE_ASSET_METADATA_RETRIEVED',
                'asset_name':asset['name'], 'asset_size':asset['size'],
                'asset_digest':declared, 'asset_created_at':asset.get('created_at'),
                'asset_updated_at':asset.get('updated_at'), 'asset_url':asset['browser_download_url'],
                'size_matches_mirror':asset['size'] == audit['archive_bytes'],
                'sha256_matches_mirror':(declared == 'sha256:'+audit['archive_sha256']) if declared else None})
    except Exception as exc:
        result.update({'status':'RELEASE_METADATA_UNAVAILABLE','exception_type':type(exc).__name__,
                       'http_status':exc.code if isinstance(exc, HTTPError) else None})
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
