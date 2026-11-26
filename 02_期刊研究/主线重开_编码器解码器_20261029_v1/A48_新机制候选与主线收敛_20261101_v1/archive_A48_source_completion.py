from pathlib import Path
import concurrent.futures,json,subprocess,sys
from archive_A48_sources import acquire,LIB

ROOT=Path(__file__).resolve().parent
ROWS=[
 ('A48_LearnableBN_arXiv2411_18860v1','Improving Batch Normalization with TTA for Robust Object Detection in Self-Driving','https://arxiv.org/pdf/2411.18860v1','10.48550/arXiv.2411.18860','arXiv v1 2024-11-28; do not equate with AAAI2026 final'),
 ('A48_CEASC_CVPR2023_Publisher','Adaptive Sparse Convolutional Networks With Global Context Enhancement for Faster Object Detection on Drone Images','https://openaccess.thecvf.com/content/CVPR2023/papers/Du_Adaptive_Sparse_Convolutional_Networks_With_Global_Context_Enhancement_for_Faster_CVPR_2023_paper.pdf',None,'CVPR2023 publisher final'),
 ('A48_Region_Normalization_arXiv1911_10375v1','Region Normalization for Image Inpainting','https://arxiv.org/pdf/1911.10375v1','10.48550/arXiv.1911.10375','arXiv v1, not automatically AAAI2020 final')]

def bounded(index):
    row=ROWS[index]
    try:
        p=subprocess.run([sys.executable,'-s','-X','utf8',str(__file__),'child',str(index)],capture_output=True,text=True,encoding='utf-8',timeout=55)
        if p.returncode:raise RuntimeError(p.stderr[-2000:])
        return json.loads(p.stdout)
    except Exception as exc:
        result={'title':row[1],'original_url':row[2],'doi':row[3],'version':row[4],
                'status':'ACQUISITION_FAILED','reason':str(exc),'reading_scope':'tracked in continuation research record'}
        path=LIB/(row[0]+'.source.json')
        if not path.exists():path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        return result

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':print(json.dumps(acquire(ROWS[int(sys.argv[2])]),ensure_ascii=False))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(bounded,range(3)))
        (ROOT/'source_completion_v1.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps([{k:r.get(k) for k in ['title','status','sha256','reason']} for r in rows],ensure_ascii=False))
