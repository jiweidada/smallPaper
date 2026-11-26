from pathlib import Path
import requests,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
texts=ROOT/'selected_text';texts.mkdir(exist_ok=True)
records=[]
for name,url,title in [
 ('A51_mHC_arXiv2512.24880v1.pdf','https://arxiv.org/pdf/2512.24880v1','mHC: Manifold-Constrained Hyper-Connections'),
 ('A51_FreqFusion_arXiv2408.12879v1.pdf','https://arxiv.org/pdf/2408.12879v1','Frequency-aware Feature Fusion for Dense Image Prediction')]:
    p=LIB/name
    assert not p.exists(),str(p)
    try:
        r=requests.get(url,timeout=(15,55));r.raise_for_status();assert r.content.startswith(b'%PDF')
        p.write_bytes(r.content)
        text=texts/(p.stem+'.txt')
        subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe','-layout',str(p),str(text)],check=True)
        rec={'title':title,'source_url':url,'version':url.rsplit('/',1)[-1],
             'pdf':str(p),'bytes':len(r.content),'http_status':r.status_code,
             'sha256':hashlib.sha256(r.content).hexdigest(),'text':str(text),
             'reading_scope':'downloaded/extracted only; selected methods will be recorded separately'}
        p.with_suffix('.pdf.source.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as e:
        rec={'title':title,'source_url':url,'status':'NOT_ACQUIRED','error':repr(e)}
    records.append(rec)
(ROOT/'02_原文获取登记_v1.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in x.items() if k in ['title','bytes','sha256','status','error']} for x in records],ensure_ascii=False))
