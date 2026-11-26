import json, os
OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
res = {
  "item": "341 第一节第4条：复现主控 κ→残余测试",
  "target_from_317v4": {"-6.7": 0.9989, "-0.67": 0.7920, "-0.067": 0.7489, "0": 0.7445, "0.067": 0.7402, "0.67": 0.7067, "6.7": 0.6315},
  "reproduced": False,
  "what_matched": {
    "n_windows": 32,
    "mean_candidates_per_window": 45.9,
    "target_mean_candidates": 45.9,
    "direction": "κ 越负 → 残余越接近 mass（与 317 §35 一致）；单候选算子验证 κ=-6.7→0.9999、κ=+6.7→0.9001 正确"
  },
  "what_did_not_match": {
    "level_at_kappa_0": {"target": 0.7445, "mine_intersect": 0.6157, "mine_center": 0.6580, "mine_allP": 0.1282},
    "slope_over_pm6.7": {"target": 0.3674, "mine_intersect": 0.2751, "mine_center": 0.2428, "mine_allP": 0.5977}
  },
  "variants_tried": [
    "候选=与窗正面积相交的 fullP（317 定义）+ 窗内归一化坐标 + mass=1/256/格",
    "候选=中心落在窗内的 fullP + 同上",
    "候选=全部 300 query + 同上",
    "mass=1.0/格（每类整窗质量 256）——数值塌到 0.98 平坦，明显不符"
  ],
  "likely_causes": [
    "317v4 第二节未写明候选集与窗口坐标的具体取法；我按 317 §13 的 P_W 定义（正面积相交）实现，得 0.6157",
    "主控可能用了不同的 mass 归一化（如按类归一化到 1 而非每格 1/256）或不同的 d_out 归一化基准",
    "不排除主控的测试脚本里候选/质量口径与 317 正文有别的读法"
  ],
  "action": "按 341 红线，不凑数、不调参去逼近目标值。请主控给出该测试的候选集口径与 mass/d_out 归一化定义，我再复现。",
  "verdict": "结构量（窗数、候选均值）一致；数值水平不一致；判为『未复现』，待主控澄清口径。"
}
json.dump(res, open(os.path.join(OUT, "341_result_kappa_residual.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("recorded: not reproduced, awaiting clarification")