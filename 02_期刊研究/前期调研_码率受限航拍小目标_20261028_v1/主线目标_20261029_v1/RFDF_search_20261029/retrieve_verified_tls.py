"""Alternative verified TLS transport for the same public metadata/publisher endpoints."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,ssl,json,hashlib,time
ROOT=Path(__file__).resolve().parent
URLS={
 'crossref':'https://api.crossref.org/works/10.1016%2Fj.measurement.2026.123424',
 'openalex':'https://api.openalex.org/works?filter=doi:https%3A%2F%2Fdoi.org%2F10.1016%2Fj.measurement.2026.123424',
 'semanticscholar':'https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.measurement.2026.123424?fields=title,openAccessPdf,url',
 'publisher_xml':'https://api.elsevier.com/content/article/pii/S0263224126031349?httpAccept=text%2Fxml',
 'publisher_pdf':'https://www.sciencedirect.com/science/article/pii/S0263224126031349/pdfft?isDTMRedir=true&download=true'}
def get(job):
    name,url=job;row=dict(endpoint=name,url=url,TLS_verification=True);tic=time.monotonic()
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Academic literature source verification'})
        with urllib.request.urlopen(req,context=ssl.create_default_context(),timeout=30) as response:
            data=response.read(40000001);assert len(data)<=40000000
            row.update(http_status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),is_pdf=data.startswith(b'%PDF'))
            (ROOT/(name+'_verified.response')).write_bytes(data)
            try:
                m=json.loads(data)
                if name=='crossref':row.update(links=m.get('message',{}).get('link'),license=m.get('message',{}).get('license'),title=m.get('message',{}).get('title'))
                if name=='openalex':row['open_versions']=[dict(open_access=x.get('open_access'),locations=x.get('locations'),best_oa_location=x.get('best_oa_location')) for x in m.get('results',[])]
                if name=='semanticscholar':row['metadata']=m
            except (ValueError,UnicodeError):pass
    except Exception as e:row['error']=repr(e)
    row['seconds']=time.monotonic()-tic;return row
with ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(get,URLS.items()))
(ROOT/'verified_tls_receipt.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=True))
