"""Fetch only public, commit-pinned source for static research; execute no source."""
from pathlib import Path
import hashlib
import json
import urllib.request
import concurrent.futures

P=Path(__file__).resolve().parent/'post_terminal_source_screen'
REPO='LeoHoW6/MambaSOD-Drone-view-small-object-detection-via-a-Mamba-based-query-feature-interaction-framework'
PIN='dc5ca8b18d917a32205bb7e63c8d1882c30f91d9'


def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'local-research-source-audit'})
    with urllib.request.urlopen(req,timeout=25) as f:
        return f.read()


def main():
    P.mkdir(exist_ok=False)
    tree_bytes=fetch(f'https://api.github.com/repos/{REPO}/git/trees/{PIN}?recursive=1')
    (P/'MambaSOD_git_tree.json').write_bytes(tree_bytes)
    tree=json.loads(tree_bytes)
    assert tree['sha']==PIN and not tree['truncated']
    paths=[r['path'] for r in tree['tree'] if r['type']=='blob' and (r['path'].endswith('.py') or r['path'] in ['README.md','LICENSE'])]
    assert len(paths)<=15
    def one(name):
        row=next(r for r in tree['tree'] if r['path']==name)
        data=fetch(f'https://raw.githubusercontent.com/{REPO}/{PIN}/{name}')
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\x00'+data).hexdigest()==row['sha']
        dst=P/'MambaSOD_pinned'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(data)
        return {'path':name,'bytes':len(data),'git_blob':row['sha'],'sha256':hashlib.sha256(data).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(one,paths))
    out={'status':'PUBLIC_SOURCE_ARCHIVED_STATIC_ONLY','repo':REPO,'commit':PIN,
         'files':rows,'new_model_forwards':0,'new_optimizer_updates':0,
         'source_imported_or_executed':False,'paper_fulltext_read':False,
         'scope':'Correctly hyphenated author-linked repository; old failing URL omitted the small-object hyphen. Availability correction is not a new algorithm.'}
    with (P/'MambaSOD_source_receipt.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out))


if __name__=='__main__':main()
