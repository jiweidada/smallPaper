from pathlib import Path
import requests,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent;LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
entries=[]
for name,arx,title in [('A51_RevBiFPN_arXiv2206.14098v1.pdf','2206.14098v1','RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network'),
 ('A51_NeuralSheafDiffusion_arXiv2202.04579v1.pdf','2202.04579v1','Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs')]:
    p=LIB/name;assert not p.exists()
    url='https://arxiv.org/pdf/'+arx
    r=requests.get(url,timeout=(15,55));r.raise_for_status();assert r.content.startswith(b'%PDF');p.write_bytes(r.content)
    t=ROOT/'selected_text'/(p.stem+'.txt')
    subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe','-layout',str(p),str(t)],check=True)
    e={'title':title,'source_url':url,'version':arx,'pdf':str(p),'text':str(t),
       'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'http_status':r.status_code,
       'reading_scope':'downloaded/extracted only; selected methods reading separately recorded'}
    p.with_suffix('.pdf.source.json').write_text(json.dumps(e,ensure_ascii=False,indent=2),encoding='utf-8');entries.append(e)
(ROOT/'03_补充近邻原文获取_v1.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in e.items() if k in ('title','bytes','sha256')} for e in entries]))
