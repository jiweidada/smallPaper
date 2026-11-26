import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
torch.set_num_threads(2)
from uic_pipeline import select_window, development_set, heldout_guard, run_arithmetic, resume_state_keys
res={"item":"任务4 自然数据缓存/训练/续跑/独评管线 预检","executed":[],"tests":{}}

# window selection: put most centers in cell (5,7); ensure row-major tiebreak
cents=[]
for _ in range(10): cents.append((5*10+5, 7*10+5))  # cell (5,7) gets 10
cents.append((1*10+3, 2*10+3))  # cell(1,2)
rc=select_window(cents, grid=64, win_px=640)
res["tests"]["window_max_count"]={"picked":rc,"expect_(5,7)":rc==(5,7)}
# tie: two cells equal count -> row-major first (smaller row then col)
cents2=[(5*10,7*10),(1*10,2*10)]  # cells (5,7),(1,2) each 1 -> row-major picks (1,2)
rc2=select_window(cents2,grid=64)
res["tests"]["window_tie_rowmajor"]={"picked":rc2,"expect_(1,2)":rc2==(1,2)}

# run arithmetic vs 317
res["tests"]["run_arithmetic"]=run_arithmetic()
# heldout guard
res["tests"]["heldout_guard"]={
  "heldout_path_blocked": heldout_guard(r"...\heldout3236\img.png")["safe_to_write"]==False,
  "dev_path_ok": heldout_guard(r"...\development\img.png")["safe_to_write"]==True}
res["tests"]["resume_keys"]={"keys":resume_state_keys()}

# development set SHA: recompute from sample_plan.development ids -> compare fdea615d
import json as J
sp=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
try:
    plan=J.load(open(sp,encoding="utf-8"))
    dev=plan["source_pools"]["development"]
    # 324 development_ids_SHA256 fdea615d: likely sha of json dump of ids; try canonical
    for mode in ["json_compact","json_sorted_noSpace","str_join"]:
        if mode=="json_compact": b=J.dumps(dev,separators=(',',':')).encode()
        elif mode=="json_sorted_noSpace": b=J.dumps(sorted(dev),separators=(',',':')).encode()
        else: b=(",".join(map(str,dev))).encode()
        import hashlib
        h=hashlib.sha256(b).hexdigest()
        if h.startswith("fdea615d"):
            res["tests"]["development_sha"]={"mode":mode,"sha":h,"match":True,"count":len(dev)}
            break
    else:
        res["tests"]["development_sha"]={"note":"未能复现 fdea615d 的序列化方式","sample_plan_tail_sha":plan.get("source_pools",{}).get("development_sha256","(none)")}
except Exception as e:
    res["tests"]["development_sha"]={"error":str(e)}

res["blocker"]={
  "finding":"BASE24 checkpoint 的训练模型类 ResidualGeometryDEIM 不在本地 DEIM 检出中（全仓搜索无该类定义），stock DEIM configs 全部 state_dict size mismatch（decoder.anchors 形状不符）。因此 真实 global 前向 / raw-eval-native parity / global 零grad state不变 / 真实参数10231995 复核 无法本地执行。",
  "affected_preflights":["raw/eval/native 输出 parity","global 零grad、state不变","真实每图 final300 预测与 M0/crop 缓存"],
  "not_executed_reason":"模型类缺失，非本实现过错；需主控提供 ResidualGeometryDEIM 定义或其训练代码。"}
res["executed"]=["window selection 2 cases","run arithmetic","heldout guard","resume keys","development_sha attempt"]
res["not_executed"]=["真实 global 前向与缓存（模型类缺失）","缓存预测/M0/crop 落盘","续跑真实恢复","val548 独评"]
J.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task4_pipeline.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(J.dumps(res,ensure_ascii=False,indent=2))
