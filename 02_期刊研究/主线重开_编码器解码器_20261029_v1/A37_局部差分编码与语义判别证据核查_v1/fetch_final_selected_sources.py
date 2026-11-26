from pathlib import Path
import hashlib, json, subprocess, sys, urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
rows=[]
def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review'})
    with urllib.request.urlopen(request,timeout=30) as response: return response.read()
jobs=[('PiDiViT','Seaz9/PiDiViT',['tools/train_net.py','detectron2/config/defaults.py',
      'configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml','scripts/eval.sh']),
      ('UnmaskingTiny','AngstCJ/Unmasking-the-Tiny-Foreground-Probing-for-Small-Object-Detection',['README.md'])]
for slug,repo,paths in jobs:
    root=D/'repo_metadata'/slug
    pin=json.loads((root/'receipt.json').read_text(encoding='utf-8'))
    tree=json.loads((root/'tree.json').read_text(encoding='utf-8'))
    blobs={x['path']:x for x in tree['tree'] if x['type']=='blob'}
    for path in paths:
        url='https://raw.githubusercontent.com/'+repo+'/'+pin['commit']+'/'+path
        raw=get(url)
        git_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert git_sha==blobs[path]['sha'],path
        target=D/'source'/slug/path
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output: output.write(raw)
        rows.append(dict(repo=repo,commit=pin['commit'],path=path,url=url,
                         git_blob_sha=git_sha,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
url='https://openreview.net/pdf/707fcb73ebb1d39dd2dfa21c0b43c9b99667f75d.pdf'
preprint=dict(url=url,version='Anonymous ICLR2026 submission, NOT the verified final journal version')
try:
    raw=get(url)
    assert raw.startswith(b'%PDF-'),'Response is not a PDF'
    lib=D.parent.parent/'文献_相关原始论文_20261004'
    pdf=lib/'A37_UnmaskingTiny_OpenReview_preprint.pdf'
    with pdf.open('xb') as output: output.write(raw)
    txt=lib/'_text'/'A37_UnmaskingTiny_OpenReview_preprint.txt'
    assert not txt.exists()
    subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],check=True)
    preprint.update(status='PDF_DOWNLOADED_NOT_YET_READ',file=str(pdf),text_file=str(txt),
                    bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                    text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest())
    with (lib/'A37_UnmaskingTiny_OpenReview_preprint.source.json').open('x',encoding='utf-8') as output:
        json.dump(preprint,output,ensure_ascii=False,indent=2)
except Exception as error:
    preprint.update(status='PUBLIC_PDF_UNAVAILABLE',error=repr(error))
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),files=rows,preprint=preprint,
            author_code_imported_or_executed=False)
with (D/'final_selected_source_verification.json').open('x',encoding='utf-8') as output:
    json.dump(result,output,ensure_ascii=False,indent=2)
print(json.dumps(dict(verified_extra_blobs=len(rows),preprint=preprint),ensure_ascii=False))
