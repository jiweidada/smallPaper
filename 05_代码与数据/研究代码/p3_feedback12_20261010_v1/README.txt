P3 final-only feedback：隔离12轮实现，尚未训练

来源：同级p3_readout_screen24_20261009_v1；逐文件来源、SHA及关键改动见source_manifest.json。
只在本新目录写入。原P3、native训练引擎、既有权重、结果与计划未修改。

候选final_only_feedback与aux_only具有完全相同SPATIAL参数及推理。
训练新增reference=z1.detach().requires_grad_(True)，DN拼接前形状[B,300,4]。
shadow_delta使用同一个local_readout，输入semantic/P3/z0全部detach，未新增参数。
原native criterion先计算全部原损失/GO匹配；仅对已核对的加权主项
loss_mal、loss_bbox、loss_giou、loss_fgl先乘native当前GradScaler scale求VJP，retain_graph=True。
先将g_scaled转FP32，再除scale并detach，交给ZeroValueSurrogate。
该Function前向严格返回FP32零，反向对shadow_delta.float返回grad_output*g；有限时与点积减detach的梯度完全相同。
整个loss随后仍由原native scaler统一缩放反向，损失数值不变。
solver绑定当前scaler引用；读取原forward捕获的autocast状态，实际FP32 fallback强制scale1。
AMP额外VJP非有限时不清洗/替代/截断梯度，零前向避免inf-inf污染loss；实际非有限头梯度交原生GradScaler检测并skip/backoff。
不静默修改正式scale，不新增重试；非AMP额外VJP非有限仍fail-fast，failure.json保留已消耗attempt/step。
该项只给local_readout专属参数梯度；原encoder辅助学习路径仍保留。
原native内部的reference/pre/FDR detach不变；不把它称为完整无detach解码器。
推理不建leaf、不算shadow、不附加context，每次仍只运行一次原head。

预算：原132轮学习率/增广配方前12轮，epoch_zero_based=11的last EMA；无resume。
train命令仅接受final_only_feedback；aux_only用于零更新工程等价验收，不自动重训对照。
BASE/SPATIAL历史checkpoint0011的存在、SHA、EMA及正式重评由独立调用方验收。
results.capture/load_endpoint接受baseline/spatial/aux_only/final_only_feedback，严格epoch11。
不接受24轮权重替代，不选best，不自动追加轮数，不内置远端启动器或自动科学结论。

工程验收（调用方另行执行；此交付只完成AST/配置静态检查）：
python preflight.py --native-root <原绑定native> --device cpu --output <新JSON>
python preflight.py --native-root <原绑定native> --device cuda:0 --amp --output <新JSON>
preflight仅合成features/GT，真实native decoder及criterion；无optimizer构造/step，无真实训练数据。
检查：eval严格等价、主损失逐项等值、proxy标量零、共享输入/参数proxy梯度为None、
有限fixture的新增头梯度差与proxy一致、共享裁剪前梯度一致、梯度有限；两分支参数不变、未知主loss拒绝。
AMP preflight分两项：默认65536检查loss有限且零增量，完整scaled backward确实暴露非有限头梯度；
预先固定128的独立工程fixture检查有限且非零的reference-VJP→shadow oracle和隔离梯度。128不配置正式训练scaler。
两项均零optimizer构造/step/update，不能据此声称已实测native skip/backoff；正式训练审计才记录真实结果。
另含FP16微梯度下溢/缩放恢复及有限Function与旧点积梯度严格等价夹具。
AMP原生头梯度以预先既有的未缩放atol/rtol=2e-3比较；CPU oracle及纯代数保持strict0。
逐参数记录applied/oracle差、同oracle重复差、main-reference VJP重复差的未缩放maxabs/L2与信号L2；不据此预断数值差原因。
验收失败保存回执，不自动调scale或按失败扩大容差；之前strict0失败历史由调用方保留。
委托reset_model/set_model覆盖原checkpoint保存接口。

审计：前8次criterion调用记录g_ref、proxy头梯度、encoder辅助头梯度范数及比值；
额外grad审计只限这8次。TrainingAudit按原optimizer实际posthook计步，另记attempt/EMA/AMP scale。
临时包装原clip_grad_norm_，原调用参数与返回值不变，前8实际step前记录当前attempt的pre/postclip。
最早8次clip另行保存；AMP跳步不会把旧attempt的clip记录挂到后续成功step；finally恢复。
feedback_audit.json记录影子前向与主VJP累计GPU事件时间、host提交时间和首8次审计时间。
实际attempt在下次训练forward或epoch完成时结算，关联额外VJP是否非有限、optimizer posthook是否执行、scale前后值；失败中断的未完成attempt不冒称跳步。
epoch训练审计记录真实壁钟时间；所有增加的审计/同步也属于实际成本。
梯度裁剪、AMP溢出/跳步、训练共同适应会影响整体更新；不声称主干优化器更新保持不变。

本包未声称有效或首创。外部正式验收需同时比较12轮APs/AP、原成功损伤与成本。
