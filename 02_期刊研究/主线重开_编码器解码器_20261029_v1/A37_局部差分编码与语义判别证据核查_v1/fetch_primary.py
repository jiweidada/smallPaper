from pathlib import Path
import datetime,hashlib,json,subprocess,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;L=B.parent/'文献_相关原始论文_20261004'
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
    with urllib.request.urlopen(req,timeout=30) as f:return f.read()
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(data)
def jsave(p,x):save(p,(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
priorpath=B/'83_A36原生分类头与独立指标核验回执.json';prior=json.loads(priorpath.read_text(encoding='utf-8'))
for n,h in prior['files_sha256'].items():assert sha(n)==h,n
assert len(prior['files_sha256'])==82
jsave(D/'handoff_A36_verified.json',dict(previous_goal_turn='progress:A36completed12nativeheadcallsandindependentCOCOverification;qualityoffsetharmnotpromoted;nextactionchangedtoencoder/querysemanticinformation.',
 prior_A36_files_verified=82,prior_receipt=str(priorpath),prior_receipt_sha256=sha(priorpath),
 model_forwards_before=3168,optimizer_updates_before=528,complete_decoder_only_calls_before=61,
 A36_native12cap_remains_closed=True,new_model_call_allocation=0,new_training_allocation=0))
rows=[]
papers=[('A37_PiDiViT_ICCV2025','https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_When_Pixel_Difference_Patterns_Meet_ViT_PiDiViT_for_Few-Shot_Object_ICCV_2025_paper.pdf'),
 ('A37_PiDiViT_ICCV2025_supplement','https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Zhou_When_Pixel_Difference_ICCV_2025_supplemental.pdf')]
for stem,url in papers:
    try:
        data=get(url);assert data.startswith(b'%PDF-')
        p=L/(stem+'.pdf');save(p,data);t=L/'_text'/(stem+'.txt');assert not t.exists()
        subprocess.run(['pdftotext','-layout',str(p),str(t)],check=True)
        x=dict(file=str(p),url=url,bytes=len(data),sha256=sha(p),text_file=str(t),text_sha256=sha(t),
               retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_scope='Pendingselectedmethodinspection;notfullreading.')
        jsave(L/(stem+'.source.json'),x)
    except Exception as e:x=dict(url=url,error=repr(e))
    rows.append(x);print(json.dumps({k:v for k,v in x.items() if k in ['file','bytes','sha256','error']},ensure_ascii=False),flush=True)
try:
    root='https://api.github.com/repos/Seaz9/PiDiViT';d=D/'repo_metadata/PiDiViT'
    raw=get(root);save(d/'repo.json',raw);repo=json.loads(raw)
    raw=get(root+'/commits/'+repo['default_branch']);save(d/'commit.json',raw);pin=json.loads(raw)
    raw=get(root+'/git/trees/'+pin['commit']['tree']['sha']+'?recursive=1');save(d/'tree.json',raw);tree=json.loads(raw);assert not tree.get('truncated')
    candidate=[x['path'] for x in tree['tree'] if x['type']=='blob' and x['path'].endswith(('.py','.yaml','.yml','.md')) and any(q in x['path'].lower() for q in ['pidi','difference','meta_arch','backbone','dino','devit','fusion','conv','attention','readme'])]
    x=dict(repo='Seaz9/PiDiViT',commit=pin['sha'],tree=pin['commit']['tree']['sha'],candidate_paths=candidate)
    jsave(d/'receipt.json',x)
except Exception as e:x=dict(repo='Seaz9/PiDiViT',error=repr(e))
rows.append(x);print(json.dumps(x,ensure_ascii=False)[:3500],flush=True)
jsave(D/'download_receipt.json',dict(rows=rows))
