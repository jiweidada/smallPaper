import json
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task1_m0.json"
r=json.load(open(p,encoding="utf-8"))
r["note_fix"]="发现并修复：初次实现的 tiebreak 依赖输入行序（镜像等代价候选在重排后会换人）。已改为按候选固有不变量(原fullP索引)加 epsilon 的规范 tiebreak，重排/GT置换均不变。"
r["verdict"]="任务1 全部断言通过（并列稳定、重排不变、UW闭合、空P/空GT/无合法边、maxIoU<.5的GT必在UW、IoU=0.5边界含入、最大覆盖优先）；代价矩阵与原生匹配公式逐元素一致(diff=0)。"
json.dump(r, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("merged verdict")
