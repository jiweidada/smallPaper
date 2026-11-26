"""Download primary literature and one author's repository metadata, never execute it."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

P=Path(__file__).resolve().parent
ARCHIVE=P.parents[1]/'文献_相关原始论文_20261004'

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Academic-literature-source-audit/1.0'})
    with urllib.request.urlopen(req,timeout=25) as r:
        return r.read(40*1024*1024),r.url

def save(p,raw):
    with p.open('xb') as f:f.write(raw)
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def main():
    results=[]
    urls=[
      'https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Baty_FlowDet_Unifying_Object_Detection_and_Generative_Transport_Flows_CVPRW_2026_paper.pdf',
      'https://arxiv.org/pdf/2512.16771',
    ]
    path=ARCHIVE/'A60_FlowDet_CVPRW2026.pdf'
    assert not path.exists()
    attempts=[];pdf=None
    for url in urls:
        try:
            raw,actual=fetch(url);assert raw.startswith(b'%PDF-')
            pdf=save(path,raw);pdf.update(source_url=url,final_url=actual)
            attempts.append(dict(url=url,status='DOWNLOADED'));break
        except Exception as e:attempts.append(dict(url=url,status='FAILED',error=type(e).__name__+': '+str(e)))
    src=dict(title='FlowDet: Unifying Object Detection and Generative Transport Flows',
             publication='CVPR2026 LoViF workshop; not main-track claim',attempts=attempts,pdf=pdf,
             recorded_utc=datetime.now(timezone.utc).isoformat(),read_scope='Downloaded only; not yet read')
    save(ARCHIVE/'A60_FlowDet_CVPRW2026.source.json',json.dumps(src,indent=2).encode())
    results.append(src)
    dest=P/'CEOSOD_primary_repo_v1';dest.mkdir(exist_ok=False)
    api='https://api.github.com/repos/BrainPotter/CEOSOD'
    raw,_=fetch(api);repo=json.loads(raw);save(dest/'repo.json',raw)
    raw,_=fetch(api+'/commits/'+repo['default_branch']);commit=json.loads(raw);save(dest/'commit.json',raw)
    pin=commit['sha']
    raw,_=fetch(api+'/git/trees/'+pin+'?recursive=1');tree=json.loads(raw);save(dest/'tree.json',raw)
    assert not tree.get('truncated',False)
    paths=[x['path'] for x in tree['tree'] if x['type']=='blob']
    readme=next(s for s in paths if s.lower()=='readme.md')
    raw,actual=fetch('https://raw.githubusercontent.com/BrainPotter/CEOSOD/'+pin+'/'+readme)
    readme_record=save(dest/'README.md',raw)
    results.append(dict(repo='BrainPotter/CEOSOD',pin=pin,readme=readme_record,
                        code_executed=False,tree_paths=paths))
    save(P/'primary_fetch_receipt_v1.json',json.dumps(results,ensure_ascii=False,indent=2).encode())
    print(json.dumps(dict(flow_pdf_downloaded=pdf is not None,CEOSOD_pin=pin,
        CEOSOD_files=len(paths),CEOSOD_pdf_paths=[s for s in paths if s.lower().endswith('.pdf')],
        readme=raw.decode('utf-8',errors='replace')),ensure_ascii=False))

if __name__=='__main__':main()
