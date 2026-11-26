from pathlib import Path
import concurrent.futures,json,subprocess,sys
from archive_A48_sources import acquire,LIB

ROOT=Path(__file__).resolve().parent
ROWS=[
 ('A48_Filtered_Batch_Normalization_ICPR2020','Filtered Batch Normalization','https://arxiv.org/pdf/2010.08251','10.48550/arXiv.2010.08251','arXiv v1; ICPR2020'),
 ('A48_LearnableBN_AAAI2026','Improving Batch Normalization with Test-Time Adaptation for Robust Object Detection in Self-Driving','https://ojs.aaai.org/index.php/AAAI/article/download/37626/41588','10.1609/aaai.v40i9.37626','AAAI2026 publisher proceedings')]

def bounded(index):
    row=ROWS[index]
    try:
        p=subprocess.run([sys.executable,'-s','-X','utf8',str(__file__),'child',str(index)],capture_output=True,text=True,encoding='utf-8',timeout=55)
        if p.returncode:raise RuntimeError(p.stderr[-2000:])
        return json.loads(p.stdout)
    except Exception as exc:
        result={'title':row[1],'original_url':row[2],'doi':row[3],'version':row[4],
                'status':'ACQUISITION_FAILED','reason':str(exc),'reading_scope':'abstract/primary HTML; not downloaded PDF'}
        path=LIB/(row[0]+'.source.json')
        if not path.exists():path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        return result

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':
        print(json.dumps(acquire(ROWS[int(sys.argv[2])]),ensure_ascii=False))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(bounded,range(2)))
        (ROOT/'source_acquisition_supplement_v1.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps([{k:r.get(k) for k in ['title','status','sha256','reason']} for r in rows],ensure_ascii=False))
