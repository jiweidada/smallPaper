from pathlib import Path
import requests,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parent;LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
records=[]
urls=['https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=11741325',
 'https://xplorestaging.ieee.org/ielx8/36/11333908/11741325.pdf?arnumber=11741325']
for url in urls:
    rec={'source_url':url,'scope':'ordinary public publisher endpoint; no authenticated session or access-control bypass'}
    try:
        r=requests.get(url,timeout=(10,25))
        rec.update(http_status=r.status_code,content_type=r.headers.get('Content-Type'),bytes=len(r.content))
        if r.status_code==200 and r.content.startswith(b'%PDF'):
            p=LIB/'A51_FMRNet_TGRS2026_3744667.pdf';assert not p.exists();p.write_bytes(r.content)
            t=ROOT/'selected_text'/'A51_FMRNet_TGRS2026_3744667.txt'
            subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe','-layout',str(p),str(t)],check=True)
            rec.update(status='PDF_ACQUIRED_NOT_YET_READ',pdf=str(p),text=str(t),sha256=hashlib.sha256(r.content).hexdigest())
            p.with_suffix('.pdf.source.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
            records.append(rec);print(json.dumps(rec),flush=True);break
        rec.update(status='NOT_ACQUIRED',body_sha256=hashlib.sha256(r.content).hexdigest())
    except Exception as e:rec.update(status='NOT_ACQUIRED',error=repr(e))
    records.append(rec);print(json.dumps(rec),flush=True)
p=ROOT/'FMRNet_publisher_access_v1.json';assert not p.exists();p.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
