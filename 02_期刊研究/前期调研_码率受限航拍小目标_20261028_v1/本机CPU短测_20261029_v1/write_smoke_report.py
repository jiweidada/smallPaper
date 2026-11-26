from pathlib import Path
import hashlib
import json
import statistics

root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/name).read_text(encoding="utf-8"))
model=load("cpu_model_smoke.json")
data=load("annotation_audit.json")
images=load("image_checks.json")
fixtures=load("offline_fixture_checks.json")
receipt=load("model_cpu_accessfixed_receipt.json")
assert model["status"]==images["status"]==fixtures["status"]=="PASS" and receipt["exit_code"]==0
times=[t["seconds"] for t in model["timings_not_a_benchmark"]]
bn=model["bn_modules_checked"]
ledger={
    "user_date":"2026-10-29","timezone":"America/Los_Angeles",
    "authorization":"User: 先把本机能运行的测试; local bounded CPU functional tests only",
    "status":"COMPLETED_CPU_FUNCTIONAL_CHECKS",
    "scope":"software and data-contract checks; no scientific effect study",
    "train_annotation_images_checked":data["images"],"train_annotation_boxes_checked":data["annotations"],
    "real_train_fit_images_used":3,"generated_unique_jpeg_files":9,"jpeg_encode_calls":18,
    "torch_import_successful_processes":1,"ultralytics_import_successful_processes":1,
    "constructor_detect_head_forwards":model["construction_detect_head_calls"],
    "explicit_detector_forwards":model["explicit_detector_forwards"],
    "fixed_detection_loss_calls":19,"optimizer_steps":0,"backward_calls":0,"training_epochs":0,
    "pretrained_detector_checkpoints_downloaded":0,"pretrained_detector_checkpoints_loaded":0,
    "restorer_calls":0,"restorer_weights_downloaded":0,"proposed_policy_fits":0,
    "gpu_jobs":0,"remote_accesses":0,"test_or_test_dev_used":False,"ap_computed":False,
    "scientific_hypothesis_validated":False,"bn_unchanged":True,"parameter_state_unchanged":True,
    "commands_and_timestamps":"*_receipt.json; model success: model_cpu_accessfixed_receipt.json",
    "earlier_model_start_failure":"model_cpu.log: Pillow import failed due sandbox access to installed files; 0 model forwards in that attempt",
    "successful_environment":"portable Python base + two isolated --target dependency directories; no global install",
    "historical_counters":"08 and earlier research artifacts are design-time snapshots; this ledger records current local execution"
}
(root/"execution_ledger.json").write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding="utf-8")
versions=model["versions"]
report=f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run（本机 CPU 功能短测）
- Origin Date: 2026-10-29（用户端日期，America/Los_Angeles）
- Verification Status: UNVERIFIED（科研假设尚未验证；以下功能测试已执行并通过）
- Version Label: local_cpu_smoke_v1

本机 CPU 短测结果

结论：本机已跑通 VisDrone 训练样本的 JPEG 输入、区域组合和 YOLO26s 固定检测损失流程，可承担这部分程序开发与小规模功能检查。尚无检测精度或区域联合恢复收益的实验证据。本轮训练 0 轮，没有访问远端。

1. 已观察到的结果

数据：读取本地转换训练标注 {data['images']} 张、{data['annotations']} 个框、10 类。图像 ID、文件名和标注 ID 无重复；图像/类别引用有效；未发现非有限坐标。存在 1 个零高度框（annotation_id=76696，9999985_00000_d_0000020.jpg）。另有 2597 个微小越界框，最大越界 {images['outside_box_followup']['max_pixels']:.8f} 像素，全部小于 0.001 像素。原标注、原图片均未修改。

处理规则：参与短测的图片将 xyxy 裁剪到原图边界，再丢弃非正面积框，再转换为640输入的归一化xywh。正式训练前仍须统一原始标注转换和忽略区域规则。微小越界来自转换精度是推断，未逐框回查全部原始文本。

划分：按既定 SHA256 规则干算 dev300 / fit6171，交集为0；未创建正式划分清单，也未证明场景独立。本轮仅取 fit 候选中按文件名排序的前3张做功能检查，均为960×540。3张图片的原始文本类别计数与转换标注一致。本轮未读取 val/test/test-dev 图像或标签。

JPEG：3张真实训练图片，在原分辨率下分别进行Q15、Q20、Q70编码。Pillow {images['pillow']}，libjpeg {images['libjpeg']}，libjpeg-turbo {images['libjpeg_turbo']}；RGB、subsampling=2、optimize=True、progressive=False，不写可选EXIF/ICC元数据。18次编码产生9个唯一文件；每种配置重复编码的字节一致，已记录每文件SHA256、字节数及bpp。源图片本身为JPEG。见 image_checks.json / payloads。

区域组合：4区域→16种子集；离线小型人工数据检查硬分区与平滑重叠分区、非负权重、有效区域权重和为1、空/全端点、padding不变、零残差恒等和混合不越界，全部通过。人工二次效用函数验证11个系数构造组合与5个组合留出索引及系数恢复，误差小于1e-12；这些人工函数没有提供真实图像非加性的证据。见 offline_fixture_checks.json。

检测器：实际构建 YOLO26s，10类，随机种子0、随机权重，参数量 {model['parameters']:,}。没有下载/加载预训练检测权重。输入 [1,3,640,640]，CPU，4线程，FP32，eval，保留未融合双头；P3/P4/P5步长8/16/32。Ultralytics {versions['ultralytics']} 的模型配置、默认配置、head、tasks、loss、tal 共6个源码文件与既有固定源码存档逐项核对，换行规范化后均一致。固定源码标签v8.4.191，commit=2004cba021c68a46429be460e26954ab171b2862。

实际前向：19次显式前向及19次固定损失计算全部通过。其中第一张图用人工亮度残差生成16个组合，另做1次原输入重复，再对另外2张图各做1次Q20输入。构造器另有{model['construction_detect_head_calls']}次用于初始化的Detect头前向，单独记录。输出框张量[1,4,8400]、类别张量[1,10,8400]、解码输出[1,14,8400]。损失取 raw['one2many']，调用 v8DetectionLoss(tal_topk=10).loss(raw,batch)[0].sum()；没有使用动态E2ELoss作为效用函数。

一致性：全部损失及输出有限；重复输入的损失和解码输出逐值完全一致。{bn}个BatchNorm模块的运行均值、方差和计数均未变化，模型state_dict前后SHA256相同。无优化器、反向传播或参数更新。

耗时：19次前向及损失计算合计 {sum(times):.4f} 秒，均值 {statistics.mean(times):.4f} 秒/次，范围 {min(times):.4f}–{max(times):.4f} 秒。包括构建在内、从模型构建开始计时 {model['elapsed_seconds_since_model_construction']:.4f} 秒；完整成功子进程（含导入等）约 {receipt['elapsed_seconds']:.3f} 秒。均为本轮短测实测，不是规范测速，不能直接外推完整训练或原分辨率恢复的耗时；依赖首次配置耗时另计。

STAL原生几何：用人工2×2 GT框[29,29,31,31]检查640输入网格。原框覆盖0个候选中心，原生16像素最小边扩展后覆盖5个；大框候选几何保持不变，输入GT未被修改。这只测试几何候选池，未测试这些候选最终能否成为正样本，也未证明小目标精度改善。

2. 环境和异常记录

成功运行环境：Python3.12.14，torch={versions['torch']}，torchvision={versions['torchvision']}，ultralytics={versions['ultralytics']}，numpy={versions['numpy']}，Pillow={versions['pillow']}，opencv-python={versions['opencv-python']}。本轮torch.cuda.is_available()返回False。全依赖版本见 environment_versions.txt。

依赖隔离目录：
{root.parents[2] / 'tmp/cpu_smoke_runtime_20261029'}
{root.parents[2] / 'tmp/yolo_smoke_runtime_20261029'}

没有修改全局Anaconda、应用捆绑依赖或驱动。最初venv/ensurepip初始化失败，最终使用两份工作区独立依赖目录和专用入口。普通沙箱首次模型启动无法读取已安装的Pillow文件，0次模型前向；使用已获自动审批的普通主机执行上下文后通过。两次日志分开保留，没有覆盖失败记录。

公开依赖下载较慢。pip的官方CPU与PyPI依赖安装最终成功；备用原生下载缓存有未完成项目：Torch备用副本续传后仍不完整，已标为.whl.partial；polars备用缓存超时；3个CPU备用缓存条目缺少预期hashes字段，未通过缓存脚本。未将这些不完整备用文件用于安装或模型运行。保留原始下载日志和JSON记录；这些异常不改变最终CPU功能检查通过的观察。

3. 本轮不能回答的问题

随机权重不是VisDrone检测基线。人工亮度残差不是FBCNN或任何学到的恢复结果。损失在本轮仅用于检查接口、数值和一致性，没有解释其组合差异、计算AP或选择最优策略。

因此尚未证明：JPEG适配后仍有独立恢复收益；区域效用存在值得建模的联合效应；所提策略优于全图恢复、全局混合或简单局部规则；算法具有可发表的新颖性。D0/DA训练、冻结FBCNN恢复、真实TP/FN/FP/AP及留出组合验证均未执行。当前约4000次科学判别方案也未执行。

4. 后续最小缺口

先准备同一协议的10类YOLO26s D0/DA已训练权重及经核验的冻结恢复权重，再用训练dev中的小样本执行“原压缩 / 全图恢复 / 全局混合 / 16个区域组合”判别。以12号文件的阶段门槛判断是否继续；从真实检测证据决定是否需要策略学习。本轮已经完成运行路径检查，下一步的关键缺口是合适的模型权重和科学对照。

5. 复查与复现入口

成功命令、PID、UTC起止和退出码：model_cpu_accessfixed_receipt.json；成功日志：model_cpu_accessfixed.log；模型结构化结果：cpu_model_smoke.json。失败启动：model_cpu.log / model_cpu_receipt.json。所有本轮执行计数见 execution_ledger.json；08及更早文件的0计数为当时设计快照。

本机复现脚本：run_cpu_smoke.ps1。它依次执行标注/程序检查、3图JPEG检查和最多300秒CPU模型短测；不安装依赖、不训练、不访问远端。宜在本机普通PowerShell运行；Codex沙箱可能需要与本轮相同的主机权限读取依赖。再次执行会更新同名结果JSON，带时间戳的新日志另存。

测试目录：{root}

文件证据：annotation_audit.json / offline_fixture_checks.json / image_checks.json / cpu_model_smoke.json / execution_ledger.json / *_receipt.json。
"""
(root.parent/"14_本机CPU短测结果_20261029.txt").write_text(report,encoding="utf-8")
manifest=[]
for p in sorted(root.glob("*.py")):
    manifest.append({"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
(root/"script_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"report":str(root.parent/"14_本机CPU短测结果_20261029.txt"),"functional_status":"PASS", "explicit_forwards":19,
                  "forward_loss_seconds":sum(times),"training_epochs":0,"scientific_hypothesis_validated":False},ensure_ascii=True))
