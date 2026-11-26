"""Probe bounded HTTP range access to a pinned public training ZIP; no full download."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import json
import re

HERE = Path(__file__).resolve().parent
VAL = HERE/'public_val_archive_audit.json'
OUTPUT = HERE/'train_archive_range_probe.json'
REPO = 'HoangTrungNguyen/VisDrone2019_training'
NAME = 'data/VisDrone2019-DET-train.zip'

def main():
    assert not OUTPUT.exists()
    val = json.loads(VAL.read_text(encoding='utf-8'))
    assert val['status']=='THIRD_PARTY_ARCHIVE_MEMBERS_COMPARED'
    revision = val['revision']
    base = 'https://huggingface.co/datasets/'+REPO
    result = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'revision':revision,
              'repository':base, 'requested_suffix_bytes':65536, 'full_archive_downloaded':False,
              'models_run':0, 'evaluator_runs':0, 'private_data_uploaded':False}
    try:
        raw_url = base+'/raw/'+revision+'/'+NAME
        with urlopen(Request(raw_url, headers={'User-Agent':'Academic data source provenance audit'}), timeout=20) as response:
            pointer = response.read(2049)
        assert len(pointer)<=2048
        text = pointer.decode('utf-8')
        match_sha = re.search(r'^oid sha256:([0-9a-f]{64})$', text, re.M)
        match_size = re.search(r'^size (\d+)$', text, re.M)
        assert match_sha and match_size
        result.update({'pointer_url':raw_url,'pointer_body':text,'declared_archive_sha256':match_sha.group(1),
                       'declared_archive_bytes':int(match_size.group(1))})
        url = base+'/resolve/'+revision+'/'+NAME
        request = Request(url, headers={'User-Agent':'Academic data source provenance audit','Range':'bytes=-65536'})
        with urlopen(request, timeout=30) as response:
            result.update({'http_status':response.status, 'content_range':response.headers.get('Content-Range'),
                           'content_length':response.headers.get('Content-Length')})
            if response.status==206:
                data = response.read(65537)
                if len(data)>65536:
                    raise ValueError('suffix exceeded range')
                result.update({'status':'RANGE_SUPPORTED','suffix_bytes_received':len(data),
                               'suffix_sha256':sha256(data).hexdigest(),
                               'zip_end_record_present': b'PK\x05\x06' in data})
            else:
                response.read(1024)
                result.update({'status':'RANGE_NOT_SUPPORTED_OR_NOT_HONORED','suffix_bytes_received':None})
    except Exception as exc:
        result.update({'status':'RANGE_PROBE_INCOMPLETE','exception_type':type(exc).__name__,
                       'http_status':exc.code if isinstance(exc, HTTPError) else result.get('http_status')})
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
