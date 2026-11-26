from pathlib import Path
import requests,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parent;LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
records=[]
queries=[('mHC_DEIM', 'https://papers.ssrn.com/sol3/Delivery.cfm/a5afe9ba-e759-4520-a729-c6a6e17517e5-MECA.pdf?abstractid=6557880&mirid=1'),
 ('FMRNet_Crossref','https://api.crossref.org/works/10.1109/TGRS.2026.3744667')]
for name,url in queries:
    rec={'id':name,'source_url':url,'attempts':1,'scope':'public primary URL from search; no authenticated access or challenge bypass'}
    try:
        r=requests.get(url,timeout=(12,35))
        rec.update(http_status=r.status_code,content_type=r.headers.get('Content-Type'),bytes=len(r.content),final_url=r.url)
        if name=='mHC_DEIM' and r.status_code==200 and r.content.startswith(b'%PDF'):
            p=LIB/'A51_mHC_DEIM_SSRN6557880.pdf';assert not p.exists();p.write_bytes(r.content)
            text=ROOT/'selected_text'/('A51_mHC_DEIM_SSRN6557880.txt')
            subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe','-layout',str(p),str(text)],check=True)
            rec.update(status='PDF_ACQUIRED_NOT_YET_READ',pdf=str(p),sha256=hashlib.sha256(r.content).hexdigest(),text=str(text))
            p.with_suffix('.pdf.source.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
        elif name=='FMRNet_Crossref' and r.status_code==200:
            j=r.json();p=ROOT/'FMRNet_Crossref_response_v1.json';assert not p.exists();p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
            m=j.get('message',{});rec.update(status='PRIMARY_REGISTRY_METADATA_ONLY',title=m.get('title'),DOI=m.get('DOI'),author=m.get('author'),links=m.get('link'))
        else:
            rec.update(status='NOT_ACQUIRED',body_sha256=hashlib.sha256(r.content).hexdigest())
    except Exception as e:rec.update(status='NOT_ACQUIRED',error=repr(e))
    records.append(rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
p=ROOT/'11_直接检测近邻获取结果_v1.json';assert not p.exists();p.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
