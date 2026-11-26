本轮：冻结协议→全GT raw×EMA 4×4→固定raw A阶段→既有对话讨论。

先读00_逐GT响应表_全体与固定rawA阶段.txt，内含八张全GT矩阵及所有固定raw A阶段汇总。
01_两对话讨论与记录.txt：两对话的分工、实际往返、证据边界和下一待冻结事项。
02_结果后的问题卡与待冻结证据.txt：两张问题卡与未执行的后续记录级问题。
discussion_record.json：本轮人类请求、可见往返原话及消息ID；partner_feedback_09.txt、partner_feedback_10.txt是对方原记录的同字节快照。
protocol.json与protocol.sha256：执行前冻结口径与SHA。
attempt01/all_GT_4x4.csv：每seed/阈值、主对照与参照对照的全部4×4格。
attempt01/fixed_raw_A_stage_4x4.csv：固定raw A阶段的完整4×4格。
attempt01/fixed_class_area_size_4x4.csv：所有固定类别、面积、短边组及其与raw A阶段交叉的4×4格。
attempt01/joined_GT.csv：全部77518行同GT追踪；all_GT_results.json及fixed_raw_A_stage_results.json为完整汇总。
attempt01/verification.json、input_manifest.json：32组历史集合复现、176项分层重构和10份输入SHA核验。
independent_crosscheck.json：从两份原始逐GT表独立按身份集合交集复核128个全GT格子的记录。
independent_stage_check.json：从原始阶段表及两份hit表独立按身份集合交集复核768个固定raw A阶段格子的记录。
本轮没有新增模型前向、评价或训练。未来解释性拆分只是待冻结提议，本轮没有执行。
