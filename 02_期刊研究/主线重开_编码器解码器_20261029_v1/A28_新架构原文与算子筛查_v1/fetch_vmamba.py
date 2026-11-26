"""Pin the author VMamba repository; inspect only a small source subset."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parent
REPO='MzeroMiko/VMamba'

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-literature-source-audit'})
    with urllib.request.urlopen(req,timeout=25) as r:return r.read()

def main():
    dest=ROOT/'vmamba_source';dest.mkdir(exist_ok=False)
    commit=get('https://api.github.com/repos/'+REPO+'/commits/main');(dest/'commit.json').write_bytes(commit)
    c=json.loads(commit);pin=c['sha'];tree=get('https://api.github.com/repos/'+REPO+'/git/trees/'+c['commit']['tree']['sha']+'?recursive=1')
    (dest/'tree.json').write_bytes(tree);t=json.loads(tree);assert not t.get('truncated',False)
    blobs={r['path']:r for r in t['tree'] if r['type']=='blob'}
    names=['README.md','classification/models/vmamba.py','classification/models/csms6s.py',
      'classification/configs/vssm1/vssm_tiny_0230s.yaml']
    # The current author tree is authoritative; do not guess a missing config.
    names=[n for n in names if n in blobs]
    configs=[p for p in blobs if p.endswith('.yaml') and 'tiny' in p and any(x in p for x in ['vssm1','vssm2','vmambav2v'])]
    for name in configs[:6]:
        if name not in names:names.append(name)
    rows=[]
    for name in names:
        url='https://raw.githubusercontent.com/'+REPO+'/'+pin+'/'+name;data=get(url)
        gh=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert gh==blobs[name]['sha']
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        rows.append({'path':name,'bytes':len(data),'git_blob_sha1':gh,'sha256':hashlib.sha256(data).hexdigest(),'url':url})
    out={'repo':REPO,'commit':pin,'tree':c['commit']['tree']['sha'],'commit_date':c['commit']['committer']['date'],
      'files':rows,'scope':'Static source audit only; no fetched model code imported or executed; no weights downloaded.',
      'tree_config_candidates':configs,'state_dim_not_assumed_until_config_read':True}
    (ROOT/'vmamba_source_manifest.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'commit':pin,'files_verified':len(rows),'paths':[r['path'] for r in rows],'config_candidates':configs}))

if __name__=='__main__':main()
