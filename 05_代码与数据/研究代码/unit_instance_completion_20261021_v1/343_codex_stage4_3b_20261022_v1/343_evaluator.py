"""The bound Windows evaluator, fixed to the recorded BASE24 settings."""
import importlib,sys,contextlib,io,time
C=importlib.import_module('343_context')
sys.path.insert(0,str(C.EVAL_ROOT))
from pycocoeval.coco import COCO
from pycocoeval.cocoeval import COCOeval
NAMES=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']

def evaluate(dataset,rows,image_ids=None):
    log=io.StringIO();start=time.perf_counter()
    with contextlib.redirect_stdout(log):
        gt=COCO();gt.dataset=dataset;gt.createIndex()
        if rows:
            dt=gt.loadRes(rows)
        else:
            dt=COCO();dt.dataset={'images':dataset['images'],'categories':dataset['categories'],'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox',using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=100)
        ev.params.imgIds=sorted(gt.getImgIds() if image_ids is None else image_ids)
        ev.evaluate();ev.accumulate();ev.summarize()
    return ev,{k:float(v) for k,v in zip(NAMES,ev.stats)},log.getvalue(),time.perf_counter()-start

def evidence():
    files=[C.EVAL_ROOT/'pycocoeval'/n for n in ('coco.py','cocoeval.py','mask.py','_mask.cp310-win_amd64.pyd')]
    return {'settings':{'using_tiny_metrice':False,'using_vtiny_metrice':False,'max_dets':100},
            'source_sha256':{str(p):C.sha(p) for p in files},
            'binary_limit':'本机_mask.cp310-win_amd64.pyd；与receipt绑定Linux .so不是同一二进制，未验证跨二进制逐位一致。'}
