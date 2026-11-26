import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.dont_write_bytecode=True
EXP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
PKG=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\单一候选评价依赖补齐与构建核验_20261008_v1\isolated_runtime\pycocoeval"
sys.path.insert(0,EXP)
import evaluator_binding as eb
# build binding for this package by fingerprint (it will validate the 6 py + one backend)
try:
    b=eb.fingerprint(PKG)
    print("fingerprint ok")
    print("backend:", b["backend"])
    print("py files match receipt:", {k:(b["files"][k]==v) for k,v in json.load(open(os.path.join(EXP,"evaluator_sources.json")))["files"].items()})
    print("extra keys:", set(b["files"])-set(json.load(open(os.path.join(EXP,"evaluator_sources.json")))["files"]))
except Exception as e:
    print("ERR", type(e).__name__, e)
