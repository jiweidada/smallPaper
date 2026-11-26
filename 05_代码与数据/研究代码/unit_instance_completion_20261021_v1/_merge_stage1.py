import json, os
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\336_result_stage1_subset.json"
d=json.load(open(p,encoding="utf-8"))
re=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\336_result_stage1_raweval.json",encoding="utf-8"))
d["tests"]["raw_eval_consistency"]={"checks":re["checks"],
  "conclusion":"fullP 全部元素都在 PostProcessor top300 内且 score 相同；raw/eval 选择一致",
  "note":"4 张样本 fullP=300，score 均 >0.1，>0.001 阈值未过滤任何项（真实分数远高于阈值）"}
# full-scale cost estimate from measured mean
mean=0.168
d["full_scale_estimate"].update({
  "measured_mean_secs_per_image":mean,
  "estimated_cpu_seconds_6470":round(mean*6470,1),
  "estimated_cpu_hours_6470":round(mean*6470/3600,2),
  "note":"按实测 0.168s/图线性估算 6470 次前向约 %.1f s (%.2f h)，仅为估算，非承诺墙钟；不含图像解码/IO 与 M0/选窗开销" % (mean*6470, mean*6470/3600)})
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("merged; est hours:", d["full_scale_estimate"]["estimated_cpu_hours_6470"])
