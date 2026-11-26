"""Schema-only compatibility: COCO info is optional metadata in source val JSON."""
import importlib,difflib
C=importlib.import_module('344_context')
old=(C.HERE/'344_independent_pycocotools.py').read_text(encoding='utf-8')
new=old.replace("gt=COCO(str(ann));dt=gt.loadRes(str(p));ev=COCOeval(gt,dt,'bbox')", "gt=COCO(str(ann));gt.dataset.setdefault('info',{});dt=gt.loadRes(str(p));ev=COCOeval(gt,dt,'bbox')")
new=new.replace("'installed_evaluator_sha256':", "'schema_compatibility':'仅在内存补info={}元数据；标注、预测、阈值和参数不改。', 'installed_evaluator_sha256':")
new=new.replace('_baseline_COCO.txt','_baseline_COCO_v2.txt').replace('344_result_独立pycocotools_baseline.json','344_result_独立pycocotools_baseline_v2.json')
C.text('344_independent_pycocotools_v2.py',new)
C.text('344_pycocotools兼容修正.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='344_independent_pycocotools.py',tofile='344_independent_pycocotools_v2.py')))
C.write('344_result_pycocotools首次实现失败.json',{'status':'实现失败','exception':"KeyError: 'info' in installed pycocotools.COCO.loadRes",
    '已执行':{'GT_annotation_reads':1,'successful_prediction_loads':0,'COCO_evaluate_calls':0,'global_forwards':0,'optimizer_steps':0},
    'repair':'另存v2，仅在内存增加info={}；不修改磁盘标注。','未执行':['真实训练','GPU','阶段5'],'未核':['独立评估完成'],'source_sha256':C.sources()})
