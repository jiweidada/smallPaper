import os, sys, json
pkg=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\单一候选评价依赖补齐与构建核验_20261008_v1\isolated_runtime"
sys.path.insert(0,pkg)
try:
    import pycocoeval
    print("import OK", pycocoeval.__file__)
    from pycocoeval import COCO, COCOeval
    print("COCO/COCOeval OK")
    import pycocoeval.mask as m
    import pycocoeval._mask as b
    print("mask.iou is backend.iou:", m.iou is b.iou)
except Exception as e:
    import traceback; traceback.print_exc()
