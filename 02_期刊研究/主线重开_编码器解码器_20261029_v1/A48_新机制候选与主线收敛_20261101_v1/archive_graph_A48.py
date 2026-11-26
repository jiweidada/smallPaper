from pathlib import Path
import json,subprocess,sys
from archive_A48_sources import LIB

ROOT=Path(__file__).resolve().parent
ROW=('A48_Graph_WeightedBN_IMMI2024','Material Property Prediction Using Graphs Based on Generically Complete Isometry Invariants',
     'https://link.springer.com/content/pdf/10.1007/s40192-024-00351-9.pdf','10.1007/s40192-024-00351-9','publisher2024')
if __name__=='__main__':
    if len(sys.argv)>1:
        from archive_A48_sources import acquire
        print(json.dumps(acquire(ROW),ensure_ascii=False))
    else:
        try:
            p=subprocess.run([sys.executable,'-s','-X','utf8',str(__file__),'child'],capture_output=True,text=True,encoding='utf-8',timeout=55)
            if p.returncode:raise RuntimeError(p.stderr[-1500:])
            result=json.loads(p.stdout)
        except Exception as exc:
            result={'title':ROW[1],'original_url':ROW[2],'doi':ROW[3],'version':ROW[4],
                    'status':'ACQUISITION_FAILED','reason':str(exc),'reading_scope':'publisher HTML Including PDD Weights eq4-6 and discussion only'}
            meta=LIB/(ROW[0]+'.source.json')
            if not meta.exists():meta.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        (ROOT/'graph_source_acquisition_v1.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps({k:result.get(k) for k in ['title','status','sha256','reason']},ensure_ascii=False))
