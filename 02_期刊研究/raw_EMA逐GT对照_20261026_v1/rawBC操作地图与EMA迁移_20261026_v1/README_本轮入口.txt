本轮入口：raw B/C操作地图→同语义CPU EMA A/B/C迁移→既有两对话讨论。

先读00_结果与执行核验.txt，再读01_两对话讨论记录.txt、02_更新问题卡与下一证据.txt。
完整冻结协议：protocol.json；SHA在protocol.sha256。
原始执行：analyze.py；raw_operations_v1和EMA_transfer_v1内的results.json、verification.json及input_manifest.json。
raw两张完整操作地图为GT_operation_map.csv.gz与pair_operation_map.csv.gz；EMA完整逐GT记录为per_GT.csv。
完整指标、GT ID集合及1000次图配对TP率区间以原始JSON为准，报告中的小数仅为显示舍入。
discussion_record.json保存可见往返原话与检索来源，不含内部推理。
本轮所有新增文件保存在此新目录，未覆盖旧报告。旧父目录README及05报告是本轮前快照。
