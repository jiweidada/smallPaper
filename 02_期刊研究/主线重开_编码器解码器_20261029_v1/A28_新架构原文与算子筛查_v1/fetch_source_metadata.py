"""Read-only public author-repository metadata; never imports fetched code."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
REPOS=['CQU-ADHRI-Lab/MI-DETR','schnurrd/ICL-Architectures']

def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'UAV-literature-source-audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=25) as r:return r.read()

def main():
    out=[]
    for repo in REPOS:
        dest=ROOT/'repo_metadata'/repo.replace('/','__');dest.mkdir(parents=True,exist_ok=False)
        root='https://api.github.com/repos/'+repo
        meta=get(root);(dest/'repo.json').write_bytes(meta);info=json.loads(meta)
        commit=get(root+'/commits/'+urllib.parse.quote(info['default_branch'],safe=''))
        (dest/'commit.json').write_bytes(commit);pin=json.loads(commit)
        tree=get(root+'/git/trees/'+pin['commit']['tree']['sha']+'?recursive=1')
        (dest/'tree.json').write_bytes(tree);tr=json.loads(tree);assert not tr.get('truncated',False)
        paths=[x['path'] for x in tr['tree'] if x['type']=='blob']
        if 'MI-DETR' in repo:chosen=[p for p in paths if p.startswith('projects/midetr/') and ('transform' in p or p.endswith('midetr.py'))]
        else:chosen=[p for p in paths if p.endswith('.py') and any(s in p.lower() for s in ['delta_net','deltanet','linear_attention','length_general','state_drift'])]
        row={'repo':repo,'commit':pin['sha'],'tree':pin['commit']['tree']['sha'],'commit_date':pin['commit']['committer']['date'],
             'blob_count':len(paths),'selected_operator_candidates':chosen[:35],
             'metadata_files_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir()}}
        (dest/'receipt.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8');out.append(row)
        print(json.dumps(row))
    (ROOT/'pinned_author_repositories.json').write_text(json.dumps({'rows':out},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
