from pathlib import Path
import datetime, hashlib, json, subprocess, sys, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
L=D.parent.parent/'文献_相关原始论文_20261004'

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
    with urllib.request.urlopen(req,timeout=30) as f:return f.read()

def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(data)

def jsave(p,x):save(p,(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

rows=[]
for stem,url in [
 ('A35_DHQ_DETR_RemoteSensing2025','https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-00514/article_deploy/remotesensing-17-00514.pdf'),
 ('A35_AQSFormer_Electronics2024','https://mdpi-res.com/d_attachment/electronics/electronics-13-04591/article_deploy/electronics-13-04591.pdf')]:
    try:
        raw=get(url);assert raw.startswith(b'%PDF-')
        p=L/(stem+'.pdf');save(p,raw);t=L/'_text'/(stem+'.txt');assert not t.exists()
        subprocess.run(['pdftotext','-layout',str(p),str(t)],check=True)
        row=dict(file=str(p),url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),text_file=str(t),text_sha256=hashlib.sha256(t.read_bytes()).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_scope='Pending selected methods; not full reading.')
        jsave(L/(stem+'.source.json'),row)
    except Exception as e:row=dict(url=url,error=repr(e))
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k in ['file','bytes','error']},ensure_ascii=False),flush=True)
for doi in ['10.3390/app15147686','10.1016/j.imavis.2026.106097','10.3390/rs17030514','10.3390/electronics13234591']:
    try:
        raw=get('https://api.crossref.org/works/'+doi)
        p=D/'metadata'/('Crossref_'+doi.replace('/','_')+'.json');save(p,raw)
        msg=json.loads(raw)['message'];row=dict(doi=doi,file=str(p),title=msg.get('title'),authors=msg.get('author'),sha256=hashlib.sha256(raw).hexdigest())
    except Exception as e:row=dict(doi=doi,error=repr(e))
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k in ['doi','title','error']},ensure_ascii=False),flush=True)
try:
    root='https://api.github.com/repos/zhangyi-magic/BUTR';d=D/'repo_metadata/BUTR'
    raw=get(root);save(d/'repo.json',raw);repo=json.loads(raw)
    raw=get(root+'/commits/'+repo['default_branch']);save(d/'commit.json',raw);commit=json.loads(raw)
    raw=get(root+'/git/trees/'+commit['commit']['tree']['sha']+'?recursive=1');save(d/'tree.json',raw);tree=json.loads(raw);assert not tree.get('truncated')
    candidates=[x['path'] for x in tree['tree'] if x['type']=='blob' and any(v in x['path'].lower() for v in ['decoder','budget','butr','visdrone']) and x['path'].endswith(('.py','.yml','.yaml','.md'))]
    row=dict(repo='zhangyi-magic/BUTR',commit=commit['sha'],tree=commit['commit']['tree']['sha'],candidates=candidates)
    jsave(d/'receipt.json',row)
except Exception as e:row=dict(repo='zhangyi-magic/BUTR',error=repr(e))
rows.append(row);print(json.dumps(row,ensure_ascii=False)[:3500],flush=True)
jsave(D/'additional_download_receipt.json',dict(rows=rows))
