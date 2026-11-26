from pathlib import Path
import datetime,hashlib,json,subprocess,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;L=D.parent.parent/'文献_相关原始论文_20261004'
FILES={
 'kwonseungchan/SPAR-Det':['configs/SPAR-Det/SPAR_Det_visdrone.py','configs/_base_/models/SPAR_Det_visdrone.py','configs/_base_/datasets/visdrone.py','configs/SPAR-Det/SPAR_Det_aitod.py','configs/_base_/models/SPAR_Det_aitod.py'],
 'Intellindust-AI-Lab/DEIMv2':['configs/base/deimv2.yml'],
 'Intellindust-AI-Lab/EdgeCrafter':['ecdetseg/configs/ecdet/ecdet.yml']}
rows=[]
for repo,paths in FILES.items():
 m=D/'repo_metadata'/repo.replace('/','__');pin=json.loads((m/'receipt.json').read_text(encoding='utf-8'))['commit']
 tree={x['path']:x for x in json.loads((m/'tree.json').read_text(encoding='utf-8'))['tree'] if x['type']=='blob'}
 for path in paths:
  url=f'https://raw.githubusercontent.com/{repo}/{pin}/{path}'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UAV-method-config-audit'}),timeout=30) as f:data=f.read()
  blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==tree[path]['sha']
  p=D/'source'/repo.replace('/','__')/path;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
  rows.append(dict(repo=repo,commit=pin,path=path,url=url,local=str(p),bytes=len(data),git_blob_sha1=blob,sha256=hashlib.sha256(data).hexdigest(),imported_or_executed=False))
(D/'additional_source_manifest.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')
stem='A32_QueryDet_CVPR2022';url='https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf'
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UAV-method-prior-audit'}),timeout=30) as f:data=f.read()
assert data.startswith(b'%PDF-');pdf=L/(stem+'.pdf')
with pdf.open('xb') as f:f.write(data)
txt=L/'_text'/(stem+'.txt');subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],check=True)
r=dict(kind='paper',file=str(pdf),url=url,version='CVPR2022officialCVF',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text_file=str(txt),text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_scope='Pending method inspection;not full-read.')
(L/(stem+'.source.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
(D/'additional_download_receipt.json').write_text(json.dumps(dict(source_blobs=len(rows),paper=r),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(verified_additional_blobs=len(rows),paper=stem,bytes=len(data),sha256=r['sha256'])),flush=True)
