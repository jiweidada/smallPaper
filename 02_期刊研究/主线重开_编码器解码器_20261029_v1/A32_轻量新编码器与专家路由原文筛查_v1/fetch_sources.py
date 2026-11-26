"""Archive public primary papers and author-repository metadata. Never import fetched code."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parent
L = D.parent.parent / '文献_相关原始论文_20261004'
PAPERS = [
 ('A32_SPAR_Det_WACV2026', 'https://openaccess.thecvf.com/content/WACV2026/papers/Kwon_SPAR-Det_Segmentation-guided_and_Prior-Aided_Routing_for_Small_Object_Detection_WACV_2026_paper.pdf', 'WACV2026 official CVF'),
 ('A32_SPAR_Det_WACV2026_supplement', 'https://openaccess.thecvf.com/content/WACV2026/supplemental/Kwon_SPAR-Det_Segmentation-guided_and_WACV_2026_supplemental.pdf', 'WACV2026 official supplement'),
 ('A32_DEIMv2_arXiv2509_20787v5', 'https://arxiv.org/pdf/2509.20787v5', 'arXiv2509.20787v5,27Oct2026'),
 ('A32_EdgeCrafter_arXiv2603_18739v5', 'https://arxiv.org/pdf/2603.18739v5', 'arXiv2603.18739v5,22Oct2026'),
 ('A32_RT_DETRv4_arXiv2510_25257v2', 'https://arxiv.org/pdf/2510.25257v2', 'arXiv2510.25257v2,8Oct2026')
]
REPOS = ['kwonseungchan/SPAR-Det', 'Intellindust-AI-Lab/DEIMv2',
         'Intellindust-AI-Lab/EdgeCrafter', 'RT-DETRs/RT-DETRv4']

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
    with urllib.request.urlopen(req,timeout=30) as f: return f.read()

def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f: f.write(data)

def jsave(p,obj):
    save(p,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    rows=[]
    for stem,url,version in PAPERS:
        try:
            data=get(url)
            assert data.startswith(b'%PDF-'), 'Not a PDF'
            pdf=L/(stem+'.pdf');save(pdf,data)
            txt=L/'_text'/(stem+'.txt')
            subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],check=True)
            row=dict(kind='paper',file=str(pdf),url=url,version=version,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text_file=str(txt),text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_scope='Pending method inspection;not full-read.')
            jsave(L/(stem+'.source.json'),row)
        except Exception as e: row=dict(kind='paper',url=url,error=repr(e))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k in ['kind','file','bytes','sha256','error']},ensure_ascii=False),flush=True)
    for repo in REPOS:
        try:
            d=D/'repo_metadata'/repo.replace('/','__');d.mkdir(parents=True,exist_ok=False)
            url='https://api.github.com/repos/'+repo
            info=get(url);save(d/'repo.json',info);branch=json.loads(info)['default_branch']
            raw=get(url+'/commits/'+branch);save(d/'commit.json',raw);pin=json.loads(raw)
            raw=get(url+'/git/trees/'+pin['commit']['tree']['sha']+'?recursive=1');save(d/'tree.json',raw)
            tree=json.loads(raw);assert not tree.get('truncated')
            candidates=[x['path'] for x in tree['tree'] if x['type']=='blob' and any(s in x['path'].lower() for s in ['gating','moe','cahf','fusion','sta.py','adapter','dinov3','vitt','encoder','distill','inject']) and x['path'].endswith(('.py','.yml','.yaml'))]
            row=dict(kind='repo',repo=repo,commit=pin['sha'],tree=pin['commit']['tree']['sha'],commit_utc=pin['commit']['committer']['date'],candidate_paths=candidates)
            jsave(d/'receipt.json',row)
        except Exception as e: row=dict(kind='repo',repo=repo,error=repr(e))
        rows.append(row);print(json.dumps(row,ensure_ascii=False)[:3500],flush=True)
    jsave(D/'download_receipt.json',dict(rows=rows))

if __name__=='__main__': main()
