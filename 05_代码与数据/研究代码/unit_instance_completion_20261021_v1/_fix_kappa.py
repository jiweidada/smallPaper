import json, os
OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
p = os.path.join(OUT, "340_result_smoke.json")
d = json.load(open(p, encoding="utf-8"))
d["RETRACTION_kappa_gradient_claim"] = {
  "wrong_statement": "上一轮汇报写『κ_c 在零初始化下检测梯度严格为 0（全 10 类 = 0.0）』并把『手动打破零初始化才有梯度』说成必要条件。",
  "correction": "严格为 0 只在第 0 步成立。从第 1 步起，优化器已把 classifier/box_head 末层推离零，10 个 κ_c 的梯度全部非零。",
  "per_step_probe": "340b_kappa_probe.json",
  "probe_summary": [
    {"step": 0, "grad_max_abs": 0.0, "grad_nonzero": 0},
    {"step": 1, "grad_max_abs": 2.76e-09, "grad_nonzero": 10},
    {"step": 5, "grad_max_abs": 1.89e-08, "grad_nonzero": 10},
    {"step": 10, "grad_max_abs": 2.08e-08, "grad_nonzero": 10},
    {"step": 20, "grad_max_abs": 6.99e-08, "grad_nonzero": 10}
  ],
  "consequence": "这不是阻塞项；但暴露真实问题：κ_c 学得极慢，20 步只移动 ~1e-4。"
}
d["kappa_c_slow_update_risk"] = {
  "observed": "20 步后 κ_c max_abs 1.57e-4；梯度 1e-8~1e-7 量级；Adam 步长约 lr 量级（warmup 1e-5→1e-4）。",
  "extrapolation_warning": "按此速率，1218 步总位移估计在 1e-2~1e-1 量级，远小于 O(1)；G 臂的类别 null 校准可能名存实亡。",
  "reference": "317 §4/§7：若 κ_c 未学动，不能声称已充分排除全局 null 校准这一解释。",
  "decision_taken": "主控选择选项 1：保持原样，训练后必须报告每个 κ_c 的实际位移，并在训练前预注册阈值。",
  "threshold_status": "阈值待定（不得训练后看结果再定）。"
}
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated 340_result_smoke.json with retraction")