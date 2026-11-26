from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError,URLError
import hashlib,json
ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'RelationDETRplus_metadata_verification_v1.json').read_text(encoding='utf-8'))
url=meta['link'][0]['URL']
assert url.startswith('http://xplorestaging.ieee.org/ielx8/') and '11586014' in url
record={'url':url,'scope':'One ordinary GET to the exact primary-registry linked PDF; no proxy, login, challenge bypass or author contact.','full_pdf_obtained':False}
try:
    with urlopen(url,timeout=25) as response:
        raw=response.read();record.update(http_status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'))
    record.update(bytes=len(raw),response_sha256=hashlib.sha256(raw).hexdigest())
    if raw.startswith(b'%PDF-'):
        dest=ROOT.parents[1]/'文献_相关原始论文_20261004/A52_RelationDETRplus_TPAMI2026.pdf'
        with dest.open('xb') as f:f.write(raw)
        record.update(full_pdf_obtained=True,pdf=str(dest),pdf_sha256=hashlib.sha256(raw).hexdigest())
    else:record['result']='Not a PDF response; stop bounded acquisition.'
except HTTPError as e:
    record.update(http_status=e.code,result='HTTP access failure; no alternate bypass attempted.')
except URLError as e:
    record.update(result='Network access failure',error=str(e.reason))
p=ROOT/'RelationDETRplus_fulltext_access_v1.json';assert not p.exists()
p.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))

