"""Archive primary method PDFs and author-repository metadata;no fetched imports."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;L=R.parent.parent/'文献_相关原始论文_20261004'
PAPERS=[('A31_TOOD_ICCV2021','https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf','ICCV2021officialCVF'),('A31_Consistent_Teacher_CVPR2023','https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Consistent-Teacher_Towards_Reducing_Inconsistent_Pseudo-Targets_in_Semi-Supervised_Object_Detection_CVPR_2023_paper.pdf','CVPR2023officialCVF')]
REPOS=['fcjian/TOOD','Megvii-BaseDetection/DeFCN']
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-nearest-method-audit','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
    with urllib.request.urlopen(req,timeout=30) as f:return f.read()
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(data)
def jsave(p,obj):save(p,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def main():
    rows=[]
    for stem,url,version in PAPERS:
        try:
            data=get(url);assert data.startswith(b'%PDF-');pdf=L/(stem+'.pdf');save(pdf,data);txt=L/'_text'/(stem+'.txt');subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],check=True)
            row={'kind':'paper','file':str(pdf),'url':url,'version':version,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text_file':str(txt),'text_sha256':hashlib.sha256(txt.read_bytes()).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_scope':'Pending method inspection;no full-read claim.'};jsave(L/(stem+'.source.json'),row)
        except Exception as e:row={'kind':'paper','url':url,'error':repr(e)}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    for repo in REPOS:
        try:
            d=R/'repo_metadata'/repo.replace('/','__');d.mkdir(parents=True,exist_ok=False);url='https://api.github.com/repos/'+repo
            info=get(url);save(d/'repo.json',info);branch=json.loads(info)['default_branch'];raw=get(url+'/commits/'+branch);save(d/'commit.json',raw);pin=json.loads(raw)
            raw=get(url+'/git/trees/'+pin['commit']['tree']['sha']+'?recursive=1');save(d/'tree.json',raw);tree=json.loads(raw);assert not tree.get('truncated')
            candidates=[x['path'] for x in tree['tree'] if x['type']=='blob' and x['path'].endswith('.py') and any(s in x['path'].lower() for s in ['tood','poto','fcos','maxfil','fcn'])]
            row={'kind':'repo','repo':repo,'commit':pin['sha'],'tree':pin['commit']['tree']['sha'],'commit_utc':pin['commit']['committer']['date'],'candidate_paths':candidates[:40]};jsave(d/'receipt.json',row)
        except Exception as e:row={'kind':'repo','repo':repo,'error':repr(e)}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    jsave(R/'primary_download_receipt.json',{'rows':rows})
if __name__=='__main__':main()
