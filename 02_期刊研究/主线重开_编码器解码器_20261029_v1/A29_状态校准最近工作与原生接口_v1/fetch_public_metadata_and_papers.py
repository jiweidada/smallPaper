"""Read-only public download. Archives sources; never imports fetched code."""
from pathlib import Path
import datetime, hashlib, json, sys, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
LIB = ROOT.parent.parent/'文献_相关原始论文_20261004'
REPOS = ['gnepul-ace/mamba_modulation', 'ArminAzizi98/LongContextMamba']
PAPERS = [('A29_Norm_or_Direction_arXiv2607_18625v1', 'https://arxiv.org/pdf/2607.18625v1'), ('A29_LinQuant_arXiv2610_39418v1', 'https://arxiv.org/pdf/2610.39418v1')]
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-source-evidence-audit','Accept':'application/vnd.github+json' if 'api.github.com' in url else 'application/pdf'})
    with urllib.request.urlopen(req,timeout=30) as r: return r.read()
def write_new(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: f.write(data)
def jwrite(path,obj): write_new(path,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def digest(data): return hashlib.sha256(data).hexdigest()
def main():
    rows=[]
    for repo in REPOS:
        try:
            dest=ROOT/'repo_metadata'/repo.replace('/','__')
            dest.mkdir(parents=True,exist_ok=False)
            url='https://api.github.com/repos/'+repo
            meta=get(url);write_new(dest/'repo.json',meta)
            commit=get(url+'/commits/'+json.loads(meta)['default_branch']);write_new(dest/'commit.json',commit);pin=json.loads(commit)
            tree=get(url+'/git/trees/'+pin['commit']['tree']['sha']+'?recursive=1');write_new(dest/'tree.json',tree);tr=json.loads(tree)
            assert not tr.get('truncated')
            paths=[x['path'] for x in tr['tree'] if x['type']=='blob']
            chosen=[p for p in paths if p.endswith(('.py','.md')) and any(x in p.lower() for x in ['mamba_simple','mamba2','calib','scale','readme','extend'])]
            row={'kind':'repo','repo':repo,'commit':pin['sha'],'tree':pin['commit']['tree']['sha'],'commit_utc':pin['commit']['committer']['date'],'blob_count':len(paths),'candidate_paths':chosen[:70],'metadata_sha256':{p.name:digest(p.read_bytes()) for p in dest.iterdir()}}
            jwrite(dest/'receipt.json',row)
        except Exception as e: row={'kind':'repo','repo':repo,'error':repr(e)}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    for stem,url in PAPERS:
        try:
            data=get(url);assert data.startswith(b'%PDF-')
            pdf=LIB/(stem+'.pdf');write_new(pdf,data)
            row={'kind':'paper','file':str(pdf),'url':url,'version':'arXiv v1','bytes':len(data),'sha256':digest(data),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_scope':'Pending inspection; no full-read claim.'}
            jwrite(LIB/(stem+'.source.json'),row)
        except Exception as e: row={'kind':'paper','url':url,'error':repr(e)}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    jwrite(ROOT/'additional_download_receipt.json',{'rows':rows})
if __name__=='__main__': main()
