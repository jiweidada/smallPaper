"""One bounded public-source fetch, no training/code execution from sources."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import urllib.request

P=Path(__file__).resolve().parent
OUT=P/'closest_prior_supplement_v1'

def main():
    assert not OUT.exists();OUT.mkdir();rows=[]
    requests=[
        ('CEOSOD_crossref.json','https://api.crossref.org/works/10.1109/TCYB.2026.3704176'),
        ('CEOSOD_default_commit.json','https://api.github.com/repos/BrainPotter/CEOSOD/commits?per_page=1'),
        ('ConsistencyDet_v5.html','https://arxiv.org/html/2404.07773v5')]
    for name,url in requests:
        row=dict(url=url,name=name,recorded_utc=datetime.now(timezone.utc).isoformat())
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'AcademicResearch/1.0','Accept':'*/*'})
            with urllib.request.urlopen(req,timeout=25) as response:
                data=response.read(8*1024*1024+1);assert len(data)<=8*1024*1024
                row.update(status=response.status,content_type=response.headers.get('Content-Type'),final_url=response.geturl())
            if name.endswith('.json'):json.loads(data)
            with (OUT/name).open('xb') as f:f.write(data)
            row.update(saved=str(OUT/name),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),downloaded=True)
        except Exception as e:row.update(downloaded=False,error=repr(e))
        rows.append(row)
    with (OUT/'fetch_receipt_v1.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2,ensure_ascii=False)
    summary=dict(attempts=len(rows),downloaded=sum(x['downloaded'] for x in rows),new_model_forwards=0,new_optimizer_updates=0)
    c=OUT/'CEOSOD_crossref.json'
    if c.exists():
        m=json.loads(c.read_text())['message'];summary['crossref']={k:m.get(k) for k in ['title','URL','link','resource','published-online','published-print']}
    c=OUT/'CEOSOD_default_commit.json'
    if c.exists():summary['CEOSOD_default_pin']=json.loads(c.read_text())[0]['sha']
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
