"""Probe the Google Drive large-file confirmation URL with HEAD only."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
import json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'official_val_confirmed_head_probe.json'
URL = 'https://drive.usercontent.google.com/download?id=1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59&export=download&confirm=t'

def main():
    assert not OUTPUT.exists()
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'method': 'HEAD',
              'source': 'https://github.com/VisDrone/VisDrone-Dataset', 'images_downloaded': False,
              'data_uploaded': False, 'scope': 'Google Drive public large-file confirmation URL metadata'}
    try:
        request = Request(URL, method='HEAD', headers={'User-Agent': 'Academic data source provenance audit'})
        with urlopen(request, timeout=15) as response:
            final = urlsplit(response.geturl())
            result.update({'status': response.status, 'final_host': final.hostname,
                           'content_type': response.headers.get('Content-Type'),
                           'content_length': response.headers.get('Content-Length'),
                           'content_disposition': response.headers.get('Content-Disposition')})
    except (OSError, HTTPError, URLError) as exc:
        result.update({'status': 'UNAVAILABLE', 'exception_type': type(exc).__name__,
                       'http_status': exc.code if isinstance(exc, HTTPError) else None})
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
