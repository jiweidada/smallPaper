"""Fetch nine pinned Git blobs for static audit only; no fetched code execution."""
from pathlib import Path
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
SELECT={
 'gnepul-ace/mamba_modulation':['README.md','Mamba/perplexity_eval/modeling/mamba_module.py','Mamba/perplexity_eval/modeling/mamba_torch.py','Mamba/perplexity_eval/eval.py','Mamba2/perplexity_eval/modeling/mamba_module.py'],
 'ArminAzizi98/LongContextMamba':['README.md','perplexity_eval/modeling/mamba_module.py','perplexity_eval/modeling/mamba_torch.py','perplexity_eval/eval.py']}
def main():
    rows=[]
    for repo,paths in SELECT.items():
        meta=ROOT/'repo_metadata'/repo.replace('/','__')
        pin=json.loads((meta/'receipt.json').read_text())['commit']
        tree={x['path']:x for x in json.loads((meta/'tree.json').read_text())['tree'] if x['type']=='blob'}
        for rel in paths:
            url='https://raw.githubusercontent.com/'+repo+'/'+pin+'/'+rel
            req=urllib.request.Request(url,headers={'User-Agent':'UAV-source-evidence-audit'})
            with urllib.request.urlopen(req,timeout=25) as r:data=r.read()
            git_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert git_sha==tree[rel]['sha'],(repo,rel)
            path=ROOT/'source'/repo.replace('/','__')/rel;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(data)
            row={'repo':repo,'commit':pin,'path':rel,'local':str(path),'url':url,'bytes':len(data),'git_blob_sha1':git_sha,'sha256':hashlib.sha256(data).hexdigest(),'imported_or_executed':False}
            rows.append(row);print(json.dumps({'repo':repo,'path':rel,'verified':True}),flush=True)
    with (ROOT/'selected_source_manifest.json').open('x',encoding='utf-8') as f:json.dump({'rows':rows},f,ensure_ascii=False,indent=2);f.write('\n')
if __name__=='__main__':main()
