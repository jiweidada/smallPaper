# 中断的 FULL 实验本地备份核查

## 已确认的材料

本地 `secondary_evidence/evidence_audits/FULL_best_epoch0000` 中的首轮检查点、预测、评价数组、指标和回执共 5 个文件，与已保存的下载回执逐项按大小及 SHA256 一致。这证明首轮保存证据仍在，不能代替预先指定的第 320 轮终点。

断线前最后一份本地状态快照报告 FULL 完成 128 轮，最后完整轮的零基索引为 127；其中记录该轮的 EMA 张量指纹 `17df8ef8e5cdea764027320c7c769bc59ce7a7a53bc10da3e0fe225772d6b257`。这份状态快照保存的是指标及哈希等元数据，不包含该轮实际权重。索引 127 之后是否曾继续运行、是否存在外部备份均未知。

两个用于转移准备材料的本地压缩包已仅按**成员名称**检查：`input_bundle.zip` 有 237 项，`secondary_frozen_inputs.tar.gz` 有 250 项，未发现 `runs/FULL/`、`FULL_best_epoch`、`epoch0127` 或 `secondary_queue.json` 对应名称。后者包含 `preflight/FULL.initial.pt`，它是训练前初始化材料，不是索引 127 或 319 的训练检查点。未解压或运行压缩包中的文件。

另外，在 `04_实验记录`、`05_代码与数据`、`90_历史与工具` 的限定文本范围内，以冻结协议 SHA、索引 127 的 EMA 张量 SHA 和原实验目录名为关键词搜索，检查 68,278 个可搜索文本文件，未发现匹配。范围只含 32 MiB 以下的 JSON、JSONL、TXT 和 Markdown，排除代码依赖目录、压缩包内容及远程存储；因此不证明其他位置不存在备份。具体文件哈希、搜索参数和输出见 `full_local_backup_audit.json`。

## 如找到原实例或备份

第二台原运行根目录记录为 `/25085400120/mpsma_fair_20261103_v1`。若原实例或快照仍在，优先保留该目录的原始结构，并定位 `secondary_queue.json`、训练及监督日志、`runs/FULL/`、`execution_configs/FULL.yml`、`preflight/`、协议与环境文件。检查文件清单和 SHA256 后再判断内容；不要覆盖已保留的本地证据。

若运行确曾完成，预定终点核查尤其需要 `runs/FULL/epochs.jsonl`、`receipt.json`、`terminal_epoch0319.pt`、`best.pt`，与两个端点相应的 `*.metrics.json`、`*.predictions.json`、`*.evaluation_arrays.npz`，以及恢复事件、逐步学习率和首 100 批记录。`evidence_audits/` 若存在也应原样保留；缺少其中某一项时，只能按实际可核查的范围报告。是否真的有完整 320 轮及上述文件，要由找到的原始材料决定。

即使找回完整 FULL，BASE 目前也只保留 225 个完整轮次，且末期 AP 骤降原因未明；仅找回 FULL 不能自动补齐预定的双组公平主对照。是否能够续跑、是否需要新训练和最少预算，必须根据实际恢复材料重新判断并取得明确授权。

本轮只检索及核对已有文件，没有构建模型、执行推理、评价或训练，也没有改动原实验产物。
