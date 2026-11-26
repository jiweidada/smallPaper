from pathlib import Path
import hashlib, json, subprocess, sys, urllib.error, urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
R=D/'repo_metadata'/'UnmaskingTiny'; R.mkdir(exist_ok=True)
receipt=[]
for name,url in [('openreview_note.json','https://api2.openreview.net/notes?id=IETH5HYvvY'),
                 ('openreview_public_pdf','https://api2.openreview.net/pdf?id=IETH5HYvvY')]:
    row=dict(url=url,name=name)
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review'})
        with urllib.request.urlopen(req,timeout=30) as response:
            raw=response.read(); row['content_type']=response.headers.get('Content-Type')
        row.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if name.endswith('.json'):
            parsed=json.loads(raw)
            with (R/name).open('xb') as output: output.write(raw)
            row.update(status='PUBLIC_NOTE_METADATA_RETRIEVED',note_count=len(parsed.get('notes',[])))
        else:
            assert raw.startswith(b'%PDF-'),'Not a PDF'
            L=D.parent.parent/'文献_相关原始论文_20261004'
            pdf=L/'A37_UnmaskingTiny_OpenReview_API_preprint.pdf'
            txt=L/'_text'/'A37_UnmaskingTiny_OpenReview_API_preprint.txt'
            assert not txt.exists()
            with pdf.open('xb') as output: output.write(raw)
            subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],check=True)
            row.update(status='PUBLIC_PREPRINT_DOWNLOADED_NOT_YET_READ',file=str(pdf),text_file=str(txt),
                       text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),
                       version_scope='Anonymous submission only; final Image and Vision Computing version not acquired')
            with (L/'A37_UnmaskingTiny_OpenReview_API_preprint.source.json').open('x',encoding='utf-8') as output:
                json.dump(row,output,ensure_ascii=False,indent=2)
    except Exception as error:
        row.update(status='UNAVAILABLE',error=repr(error))
    receipt.append(row)
with (D/'openreview_public_api_receipt.json').open('x',encoding='utf-8') as output:
    json.dump(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),rows=receipt),output,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False))
