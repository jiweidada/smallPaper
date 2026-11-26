"""Search metadata and publisher-authorized public endpoints; no credentials or access bypass."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,json,hashlib,time
ROOT=Path(__file__).resolve().parent
URLS={
 'crossref':'https://api.crossref.org/works/10.1016%2Fj.measurement.2026.123424',
 'openalex':'https://api.openalex.org/works?filter=doi:https%3A%2F%2Fdoi.org%2F10.1016%2Fj.measurement.2026.123424',
 'publisher_xml':'https://api.elsevier.com/content/article/pii/S0263224126031349?httpAccept=text%2Fxml',
 'publisher_pdf':'https://www.sciencedirect.com/science/article/pii/S0263224126031349/pdfft?isDTMRedir=true&download=true'}
def get(job):
    name,url=job;dest=ROOT/(name+'.response');tic=time.monotonic()
    p=subprocess.run([r'C:\Windows\System32\curl.exe','--location','--silent','--show-error','--max-time','40','--max-filesize','40000000',
       '--user-agent','Academic literature source verification','--output',str(dest),'--write-out','%{http_code}\n%{content_type}\n%{url_effective}',url],capture_output=True,timeout=48)
    info=dict(endpoint=name,url=url,curl_returncode=p.returncode,response_metadata=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace')[-500:],seconds=time.monotonic()-tic)
    if dest.exists():
        data=dest.read_bytes();info.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),is_pdf=data.startswith(b'%PDF'))
        if name=='crossref':
            try:
                m=json.loads(data).get('message',{});info.update(title=m.get('title'),links=m.get('link'),licenses=m.get('license'))
            except Exception:pass
        if name=='openalex':
            try:
                m=json.loads(data);info['locations']=[dict(id=x.get('id'),open_access=x.get('open_access'),best_oa_location=x.get('best_oa_location'),locations=x.get('locations')) for x in m.get('results',[])]
            except Exception:pass
    return info
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(get,URLS.items()))
(ROOT/'retrieval_receipt.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=True))
