import os, sys, json
sys.dont_write_bytecode=True
EXP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
PKG=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\单一候选评价依赖补齐与构建核验_20261008_v1\isolated_runtime\pycocoeval"
sys.path.insert(0,EXP)
import evaluator_binding as eb
b=eb.fingerprint(PKG)
exp=json.load(open(os.path.join(EXP,"evaluator_sources.json")))
got=b["files"]
print("all 6 py match:", all(got[k]==v for k,v in exp.items()))
print("extra keys:", [k for k in got if k not in exp])
print("win backend sha:", got.get("_mask.cp310-win_amd64.pyd","")[:16])
