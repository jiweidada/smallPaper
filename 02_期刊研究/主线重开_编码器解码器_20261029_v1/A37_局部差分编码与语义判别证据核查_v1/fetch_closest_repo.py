from pathlib import Path
import hashlib, json, sys, urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
repo='AngstCJ/Unmasking-the-Tiny-Foreground-Probing-for-Small-Object-Detection'
target=D/'repo_metadata'/'UnmaskingTiny'
target.mkdir(parents=True,exist_ok=True)
rows=[]
def fetch(url,name):
    request=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=30) as response:
        raw=response.read()
    with (target/name).open('xb') as output: output.write(raw)
    rows.append(dict(url=url,name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(raw)
meta=fetch('https://api.github.com/repos/'+repo,'repo.json')
commit=fetch('https://api.github.com/repos/'+repo+'/commits/'+meta['default_branch'],'commit.json')
tree=fetch('https://api.github.com/repos/'+repo+'/git/trees/'+commit['commit']['tree']['sha']+'?recursive=1','tree.json')
assert tree.get('truncated') is False
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),repo=repo,commit=commit['sha'],tree=tree['sha'],files=rows)
with (target/'receipt.json').open('x',encoding='utf-8') as output: json.dump(receipt,output,ensure_ascii=False,indent=2)
paths=[x['path'] for x in tree['tree'] if x['type']=='blob']
print(json.dumps(dict(repo=repo,commit=commit['sha'],tree_files=len(paths),selected_paths=[x for x in paths if x.lower().endswith(('.pdf','.md','.yaml','.yml')) or any(k in x.lower() for k in ['frm','stsm','foreground','attention'])]),ensure_ascii=False))
