"""Read saved results and historical logs; no model import or evaluation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent / "修订记录_20261103"
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    execution = read(ROOT / "execution_ledger.json")
    if execution["status"] != "COMPLETED" or [r["run_id"] for r in execution["runs"]] != ["N0", "N4", "S1"]:
        raise ValueError("Expected one complete sequential queue")
    if any(r["exit_code"] != 0 or r["status"] != "COMPLETED" for r in execution["runs"]):
        raise ValueError("Unexpected process status")
    frozen = read(ROOT / "preparation_manifest.json")
    old = read(AUDIT / "run_audit.json")
    image_ids = {row["image_id"] for row in read(ROOT / "validation_image_manifest.json")}
    comparisons = []
    source_manifest = read(AUDIT / "source_manifest.json")
    for run in frozen["prepared_runs"]:
        rid = run["run_id"]
        directory = ROOT / "results" / rid
        receipt, metrics, loading = (read(directory / name) for name in
                                     ("receipt.json", "metrics.json", "loading_diagnostics.json"))
        if receipt["status"] != "COMPLETED" or receipt["completed_images"] != 548:
            raise ValueError("Incomplete evaluation")
        if receipt["checkpoint_sha256"] != run["checkpoint_sha256"] or metrics["checkpoint_sha256"] != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint linkage mismatch")
        if receipt["runner_sha256"] != sha(ROOT / "evaluate_cpu.py"):
            raise ValueError("Runner changed after evaluation")
        if receipt["metrics_sha256"] != sha(directory / "metrics.json"):
            raise ValueError("Metrics changed after evaluation")
        if metrics["predictions_sha256"] != sha(directory / "predictions.json"):
            raise ValueError("Predictions changed after evaluation")
        if any(loading[name] for name in ("missing_keys", "unexpected_keys", "shape_mismatches")) or not loading["strict"]:
            raise ValueError("Strict loading failed")
        predictions = read(directory / "predictions.json")
        counts = Counter(row["image_id"] for row in predictions)
        if set(counts) != image_ids or set(counts.values()) != {300}:
            raise ValueError("Unexpected prediction/image coverage")
        if any(row["category_id"] not in range(10) or not all(math.isfinite(x) for x in row["bbox"] + [row["score"]])
               for row in predictions):
            raise ValueError("Invalid category or nonfinite prediction")
        log_path = AUDIT / "source_snapshot" / rid / "log.txt"
        entry = next(row for row in source_manifest if Path(row["snapshot"]) == log_path)
        if sha(log_path) != entry["sha256"]:
            raise ValueError("Historical log snapshot changed")
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        historical = max(rows, key=lambda row: row["test_coco_eval_bbox"][0])
        if historical["epoch"] != old[rid]["best_epoch"]:
            raise ValueError("Historical best row changed")
        table = []
        for metric, expected in zip(METRICS, historical["test_coco_eval_bbox"]):
            actual = metrics["metrics"][metric]
            table.append({"metric": metric, "historical_best_row": expected, "cpu_re_evaluation": actual,
                          "delta_percentage_points": 100 * (actual - expected),
                          "exact_numeric_match": actual == expected})
        largest = max(table, key=lambda row: abs(row["delta_percentage_points"]))
        comparisons.append({"run_id": rid, "checkpoint_sha256": run["checkpoint_sha256"],
                            "historical_best_epoch": historical["epoch"],
                            "validation_batch_size": receipt["validation_batch_size"],
                            "model_trainable_parameters": receipt["model_trainable_parameters"],
                            "completed_images": 548, "predictions_count": len(predictions),
                            "full_metric_vector_exact_match": all(row["exact_numeric_match"] for row in table),
                            "largest_absolute_metric_difference": largest, "metrics": table,
                            "metric_file_sha256": sha(directory / "metrics.json"),
                            "receipt_file_sha256": sha(directory / "receipt.json"),
                            "loading_file_sha256": sha(directory / "loading_diagnostics.json")})
    result = {"status": "THREE_AUTHORIZED_EVALUATIONS_COMPLETED_AND_ARTIFACTS_CHECKED",
              "runs": comparisons,
              "max_absolute_AP_difference_percentage_points": max(abs(r["metrics"][0]["delta_percentage_points"]) for r in comparisons),
              "no_equivalence_threshold_preregistered": True,
              "conclusion": "The recorded current implementation loads all three retained EMA checkpoints strictly and produces new hash-bound results. Numeric differences from the historical best rows remain; similarity of overall AP does not prove exact historical code/environment/checkpoint-to-log identity.",
              "unresolved": ["Historical runtime and source identity", "Causes of metric differences",
                             "Fair training controls for the baseline and ablations", "N1/N2/N3 checkpoint re-evaluation", "Deployment accuracy/timing linkage"],
              "no_training_or_tuning": True, "submission_ready": False}
    (ROOT / "comparison_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["本地 CPU 重评结果 2026-11-03", "", "已完成 N0、N4、S1 各一次完整 548 图验证；均严格加载 EMA 和后处理器，无缺失键、多余键或形状不匹配。每组保存 164400 条预测（每图300条），COCO 汇总使用 maxDets=[1,10,100]。没有训练、调参或新性能测试。", "", "运行环境：CPU/FP32，PyTorch 2.6.0+cpu、TorchVision 0.21.0+cpu、faster-coco-eval 1.6.5、NumPy 1.26.4。faster-coco-eval 版本取原依赖清单下限，不代表已查明历史版本。", "", "ID | 历史最佳行AP | 本次CPU AP | 差值/百分点 | 历史APs | 本次APs | 差值/百分点"]
    for run in comparisons:
        ap, aps = run["metrics"][0], run["metrics"][3]
        lines.append(f"{run['run_id']} | {ap['historical_best_row']:.9f} | {ap['cpu_re_evaluation']:.9f} | {ap['delta_percentage_points']:+.9f} | {aps['historical_best_row']:.9f} | {aps['cpu_re_evaluation']:.9f} | {aps['delta_percentage_points']:+.9f}")
    lines.extend(["", "完整12指标差值", "单位均为百分点；不是相对百分比。"])
    for run in comparisons:
        lines.append(run["run_id"] + "：" + "; ".join(f"{r['metric']} {r['delta_percentage_points']:+.9f}" for r in run["metrics"]))
    lines.extend(["", "解释与边界", "这些是本次当前代码/环境下与指定 SHA 权重绑定的新结果。三组均不是完整12指标逐项数值相等；总体 AP 接近不能单独证明历史版本或最佳日志归属完全一致。此次没有预先设定等价性容差，因此不追加事后容差宣布复现通过。差异的原因尚未确定，不直接归因于 CPU/GPU 浮点或评价器版本。", "原论文的历史日志表保留原口径，本次CPU结果独立报告，不将两种条件的行混在同一消融排名。", "权重可严格加载及当前评价链已补证；原N0 batch16、其他batch4以及聚合结构混杂仍未解决。公平训练对照、历史代码/数据不可变关联和部署测量链仍缺证据，尚不能投稿。", "", "资料入口：comparison_report.json 为完整数值与哈希；results/N0,N4,S1 分别含预测、指标、评价数组及回执；execution_ledger.json 含进程命令与退出码。", "", "环境准备说明：N4/S1首次兼容性检查仅因终端GBK不能打印源码中的特殊字符而退出，统一UTF-8后严格加载检查通过；失败和修正后的检查记录均保留。该阶段没有前向或准确率计算。实际完整评测每组仅运行一次。", "四个注意力源文件仅在独立演示中使用 calflops，评测入口提供遇调用即抛错的占位导入以禁用profiling；模型源文件字节未改。跳过无关包级自动导入，所选模块使用冻结副本。", "", "公开软件来源：https://pypi.org/project/faster-coco-eval/1.6.5/；https://pytorch.org/get-started/previous-versions/。依赖下载已获单独授权，未上传代码、数据或权重。"])
    (ROOT / "本地重评结果与限制.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": [
        {"run_id": r["run_id"], "AP": r["metrics"][0]["cpu_re_evaluation"],
         "AP_delta_pp": r["metrics"][0]["delta_percentage_points"],
         "largest_delta": r["largest_absolute_metric_difference"]} for r in comparisons]}))

if __name__ == "__main__":
    main()
