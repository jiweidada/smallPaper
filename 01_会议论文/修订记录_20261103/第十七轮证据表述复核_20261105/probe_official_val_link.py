"""Probe the official validation archive link with HEAD only; do not download data."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
import json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'official_val_link_head_probe.json'
URL = 'https://drive.google.com/uc?export=download&id=1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59'

def main():
    assert not OUTPUT.exists()
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'method': 'HEAD',
              'source': 'https://github.com/VisDrone/VisDrone-Dataset',
              'official_archive_page': 'https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view',
              'images_downloaded': False, 'data_uploaded': False}
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
