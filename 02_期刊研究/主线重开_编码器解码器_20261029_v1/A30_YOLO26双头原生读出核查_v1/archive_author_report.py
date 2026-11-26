"""Archive the original Ultralytics author report;no model execution."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;L=R.parent.parent/'文献_相关原始论文_20261004'
def main():
    stem='A30_Ultralytics_YOLO26_arXiv2606_03748v1';url='https://arxiv.org/pdf/2606.03748v1';req=urllib.request.Request(url,headers={'User-Agent':'UAV-author-report-audit'})
    with urllib.request.urlopen(req,timeout=30) as f:data=f.read()
    assert data.startswith(b'%PDF-');pdf=L/(stem+'.pdf')
    with pdf.open('xb') as f:f.write(data)
    txt=L/'_text'/(stem+'.txt');subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],check=True)
    row={'title':'Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models','authors':['Glenn Jocher','Jing Qiu','Mengyu Liu','Shuai Lyu','Fatih Cagatay Akyon','Muhammet Esat Kalfaoglu'],'url':'https://arxiv.org/abs/2606.03748v1','pdf_url':url,'version':'arXivv1,2026-06-02;author report,no acceptance claimed','file':str(pdf),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text_file':str(txt),'text_sha256':hashlib.sha256(txt.read_bytes()).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_scope':'Pending method inspection;no full-read claim.','human_read_attested':False}
    for path in [L/(stem+'.source.json'),R/'author_report_archive.json']:
        with path.open('x',encoding='utf-8') as f:json.dump(row,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'ORIGINAL_AUTHOR_REPORT_ARCHIVED','bytes':len(data),'sha256':row['sha256'],'text_file':str(txt)},ensure_ascii=False))
if __name__=='__main__':main()
