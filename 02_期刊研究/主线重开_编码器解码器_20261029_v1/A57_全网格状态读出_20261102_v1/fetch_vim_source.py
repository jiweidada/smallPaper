"""Read-only public upstream source retrieval, no downloaded code execution."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request

P=Path(__file__).resolve().parent


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'academic-local-source-audit'}),timeout=30) as f:return f.read()


def main():
    out=P/'upstream_Vim_pinned'
    out.mkdir(exist_ok=False)
    commit=json.loads(fetch('https://api.github.com/repos/hustvl/Vim/commits/main'))['sha']
    tree_bytes=fetch(f'https://api.github.com/repos/hustvl/Vim/git/trees/{commit}?recursive=1')
    tree=json.loads(tree_bytes)
    assert not tree['truncated'] and tree['sha']==commit
    (out/'git_tree.json').write_bytes(tree_bytes)
    names=['mamba-1p1p1/mamba_ssm/modules/mamba_simple.py','mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py','vim/models_mamba.py','vim/rope.py','LICENSE.txt']
    def one(n):
        row=next(r for r in tree['tree'] if r['path']==n)
        b=fetch(f'https://raw.githubusercontent.com/hustvl/Vim/{commit}/{n}')
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha']
        target=out/n
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(b)
        return {'path':n,'sha256':hashlib.sha256(b).hexdigest(),'git_blob':row['sha'],'bytes':len(b)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(one,names))
    weights=json.loads(fetch('https://huggingface.co/api/models/hustvl/Vim-tiny-midclstok/tree/main?recursive=false'))
    receipt={'status':'SOURCE_ONLY_NOT_IMPORTED','repo':'hustvl/Vim','commit':commit,'files':files,
             'official_weight_files':[r for r in weights if r.get('type')=='file'],
             'new_model_forwards':0,'new_optimizer_updates':0}
    with (P/'Vim_source_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
