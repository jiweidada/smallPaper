"""Independent installed pycocotools on the immutable 343 baseline predictions."""
import importlib,json,io,contextlib,time
C=importlib.import_module('344_context')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.coco,pycocotools.cocoeval,pycocotools._mask
NAMES=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']

def main():
    start=time.perf_counter();bound=json.loads((C.S343/'343_result_阶段4全量评估.json').read_text(encoding='utf-8'))
    rows=[]
    for seed in (1,2):
        ann=C.VAL/'annotations/val.json';p=C.S343/f'343_val_seed{seed}_baseline_predictions.json';log=io.StringIO()
        with contextlib.redirect_stdout(log):
            gt=COCO(str(ann));gt.dataset.setdefault('info',{});dt=gt.loadRes(str(p));ev=COCOeval(gt,dt,'bbox')
            assert ev.params.maxDets==[1,10,100]
            ev.params.imgIds=sorted(gt.getImgIds());assert len(ev.params.imgIds)==548
            ev.evaluate();ev.accumulate();ev.summarize()
        stats={k:float(v) for k,v in zip(NAMES,ev.stats)};reference=bound['per_seed'][seed-1]['cases']['baseline']['stats12']
        diff={k:stats[k]-reference[k] for k in NAMES};assert max(abs(v) for v in diff.values())<1e-12
        C.text(f'344_pycocotools_seed{seed}_baseline_COCO_v2.txt',log.getvalue())
        rows.append({'seed':seed,'stats12':stats,'minus_bound343':diff,'prediction_path':str(p),'prediction_sha256':C.sha(p),
            'annotation_path':str(ann),'annotation_sha256':C.sha(ann)})
        print(json.dumps({'seed':seed,'AP':stats['AP'],'max12diff':max(abs(v) for v in diff.values())}),flush=True)
    C.write('344_result_独立pycocotools_baseline_v2.json',{'status':'已验证','per_seed':rows,
        'schema_compatibility':'仅在内存补info={}元数据；标注、预测、阈值和参数不改。', 'installed_evaluator_sha256':{m.__file__:C.sha(m.__file__) for m in (pycocotools.coco,pycocotools.cocoeval,pycocotools._mask)},
        '已执行':{'COCO_evaluate_calls':2,'prediction_JSON_reads':2,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        'wall_seconds':time.perf_counter()-start,'未执行':['真实训练','GPU','EMA预测重评','阶段5'],
        '未核':['与Linux.so二进制逐位相同','主控独立复核'],'source_sha256':C.sources()})

if __name__=='__main__':main()
