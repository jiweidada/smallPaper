from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
load=lambda name:json.loads((ROOT/'server_receipts'/name).read_bytes())
environment=load('environment.json');data=load('data_preparation.json');pair=load('preflight_pair.json')
budget=load('formal_run_budget.json');common=load('common_initialization.json');queue=load('queue_live.json')
start=load('D0_start_receipt.json');live=load('D0_live_status.json');bundle=json.loads((ROOT/'offline_bundle_archive.json').read_bytes())
assert pair['status']=='PASS' and pair['geometric_batch_hashes_identical'] and pair['jpeg_pixels_differ']
assert start['initial_state_sha256']==common['state_sha256'] and start['train_count']==6171 and start['dev_count']==300
assert all(x['actual_optimizer_updates']==8 and x['amp_skips']==0 for x in pair['raw_results'])
assert live['actual_optimizer_updates']>0 and not live['preflight']
try:
    from zoneinfo import ZoneInfo
    local_stamp=datetime.datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()
except Exception:
    local_stamp='Unix '+str(live['updated_unix'])+'（原始技术时间戳）'
ledger={'client_date':'2026-10-29','timezone':'America/Los_Angeles','status':'FORMAL_D0_RUNNING_DA_QUEUED',
        'authorization':'User explicitly supplied this server and allowed experiments; then instructed local download -> upload',
        'download_upload_completed':True,'offline_install_completed':True,'dependency_graph_packages':54,
        'server_data_file_hashes_verified':data['source_file_hashes_verified'],'fit_dev_images':[6171,300],
        'evaluator_contract_fixture':'PASS on local CPU and server','public_initialization_unfused':'PASS',
        'capacity_preflight':{'D0_updates':8,'DA_updates':8,'amp_skips_both':0,'discarded_before_formal':True,
                             'same_first8_geometry_hashes':True,'pixel_intervention_observed':True},
        'formal_D0_observed':live,'formal_DA_started':False,'training_budget':budget,
        'scientific_effect_evaluated':False,'mainline_validated':False,'new_policy_training':0,'test_or_test_dev_used':False,
        'password_written_to_files':False,'credential_handling':'getpass; only SSH controller process memory during authorized work',
        'server_root':'/root/c10_region_restore_20261029_v1','snapshot_time':local_stamp}
if (ROOT/'evaluator_amendment_v1.json').exists():
    ledger['evaluation_implementation_amendment']='evaluator_amendment_v1.json; native clipping aligned before scientific evaluation; training code SHA unchanged'
(ROOT/'execution_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
peak=pair['raw_results'][0]['peak_allocated_bytes']/1024**3
reserved=pair['raw_results'][0]['peak_reserved_bytes']/1024**3
report=f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run（远端两组检测基线及初始问题判别）
- Origin Date: 2026-10-29；America/Los_Angeles
- Verification Status: UNVERIFIED（准备与预检通过；正式训练未完成，科研假设未验证）
- Version Label: remote_c10_v1

远端实验：运行状态与冻结方案

本地整理时刻（America/Los_Angeles）：{local_stamp}；远端状态原始更新时间戳：{live['updated_unix']}。机器时钟可能有偏差，进度和耗时按epoch/批次计数及monotonic计时核验。
已完成本地下载、文件校验、上传、服务器离线安装和两组GPU预检。D0已正式训练；最新已下载快照为第{live['epoch']}/100轮、累计{live['batches']}批、{live['actual_optimizer_updates']}次真实优化器更新、AMP跳步{live['amp_skips']}次。DA在D0之后按同一队列运行，当前尚未正式开始。

一、已实测和已完成的工作

1. 服务器：用户本次指定主机，RTX3090单卡，nvidia-smi报告24576MiB，驱动550.100；PyTorch报告可用设备总内存{environment['vram']}字节。CPU可见48逻辑线程，/dev/shm16GiB。只使用GPU0；没有访问或恢复旧DEIM/相位项目。

2. 环境：Python3.9.7、torch {environment['torch']}、torchvision {environment['torchvision']}、ultralytics {environment['ultralytics']}、numpy {environment['numpy']}、Pillow {environment['pillow']}、libjpeg {environment['libjpeg']}、libjpeg-turbo {environment['libjpeg_turbo']}。54个发行包的Linux/Python3.9依赖闭包通过；pip check未发现缺项。

   安装过程遵循“本地下载→校验→SFTP上传→--no-index离线安装”。主要51个大/小依赖逐文件匹配官方PyPI SHA256；之后补齐Python3.9的importlib-resources、zipp及该版Ultralytics实际要求的thop2.2.3，排除旧thop2.0.18。总归档{bundle['bytes']}字节，SHA256：{bundle['sha256']}。公开镜像只作为传输来源，主要依赖的完整文件必须匹配官方PyPI摘要才可入包。没有在服务器继续访问外网安装，也未改动基础Anaconda或驱动。

3. 权重：YOLO26s、YOLO26n、FBCNN color均在本机下载并校验后上传。YOLO26s/n与官方GitHub release摘要一致；FBCNN记录下载来源及本地SHA256，作者release未提供独立摘要。YOLO26s预训练包保留未融合双头，已在本地和服务器确认。FBCNN本轮仅取得权重，尚未用于恢复实验。

4. 数据：本地VisDrone训练副本6471图及6471份原始txt已转移，12942个源文件逐一SHA验证。使用原始score=1、类别1–10的合法框，裁剪边界并映射类别0–9；全train有效正框343204。3个非正面积原始框被排除，其中1个属于有效类别、2个属于ignore区域。原图片和原始标注保留；派生YOLO标签单独存放。图片JPEG结束标记及EXIF方向检查通过，避免原生加载器触发图片修复。

   固定SHA256('compress-dev-v1:'+filename)划分fit6171/dev300；dev依哈希顺序分为诊断100、调参100、确认100。训练不使用dev图片作参数更新；按既定配方监测完整dev300 reference，最终不用它选择best checkpoint，后100因此也不能宣称“从未接触过的盲测”。本轮未读取val548或test/test-dev。

5. 评估器：人工样例通过crowd忽略、空图FP、空预测、严格面积<1024、图像重复抽样bootstrap与完整COCO结果一致性。另检查横图、竖图、放大图的单次LetterBox坐标往返。均为程序口径检查，不是检测效果证据；服务器同样复验通过。

6. 公共初始化：官方COCO YOLO26s采用原生形状兼容权重迁移构造10类模型，未匹配的10类分类输出按seed0初始化，生成两组共用的完整初始化文件。文件SHA256：{common['checkpoint_sha256']}；模型状态SHA256：{common['state_sha256']}。六个关键原生源码文件与固定v8.4.191/commit2004cba021c68a46429be460e26954ab171b2862存档一致。D0实际启动时的参数状态与公共初始化完全相同。

二、配对预检实测

D0和DA各运行8个完整训练微批次，各8次实际优化器更新，AMP跳步均0；模型及EMA状态有限。每组还运行dev300 reference验证以检查训练/验证接口。前8批图像名单、类别、框和batch索引的几何SHA完全一致，JPEG组像素确有不同。两组分配显存峰值约{peak:.3f}GiB、保留显存峰值约{reserved:.3f}GiB；这是该次预检峰值，不是全训练或FBCNN的显存上界。

预检权重不进入正式训练。两组正式训练各自从公共初始化重新开始。当前D0已观察到真实参数更新；早期训练/预检AP不用于主线判断。

三、冻结的正式预算与比较

D0：source reference训练。DA：每次访问原始图片时以0.5概率追加JPEG，quality从15/20/35/50/70均匀抽取；放在首次resize和mosaic前，使用独立每worker RNG。两组均每次重新解码原图，mosaic缓存不绕过这一干预；其他几何/颜色增强、原生STAL、双头训练与Progressive Loss保持共同配方。

两组各100轮、seed0、640输入、batch8、nbs64、AdamW lr0=0.000714，最后10轮关闭mosaic；详细固定超参数见experiment_protocol.json和实际D0_start_receipt.json。单GPU顺序执行D0→DA；固定第100轮EMA为主要checkpoint。未新增网络模块、教师或策略训练，未安排多seed搜索。

每组24小时硬上限，整队列48小时上限；超时或失败保留日志并停止。OOM只允许正式开训前按既定8/4/2两组共同选择；本次batch8已通过。关闭原生OOM自动改batch和NaN自动恢复，避免在未知状态下改变配方或重复训练。记录真实优化器更新、AMP跳步、EMA更新及epoch日志。

四、这轮完成后能回答什么

两组固定checkpoint完成后，队列自动在预选诊断100图上比较reference/Q20。使用同一份Q20实际文件、固定640单次letterbox、未融合one2many加multi-label NMS（conf0.001、IoU0.7、maxDet500）。报告内部COCO式AP/AP50、严格输入面积<1024的AP_small_input_500、原图AP_small以及多阈值TP/FN/FP和source→JPEG同一GT命中转移；差值提供2000次配对图像bootstrap。

这是初始P1判别。类0/11 ignore区域扩展为各类别COCO crowd，score0有效类别在本类别作为crowd；该口径不等于官方挑战分数。图像bootstrap不覆盖训练seed方差，相关帧可能低估不确定性，不声称独立航次泛化。

尚不能证明区域联合恢复有效。若DA后仍有有意义损失，还要补简单codec/缩放对照、冻结恢复与全局混合、独立/联合区域选择的真实检测对照，才能判断P2；部署可见选择的可学习性P3更未验证。本队列在初始诊断后停止，不自动追加新模块或更长训练。若基线未收敛、预算/协议失效，结果只能记为无法判定。

五、可核查入口

远端工作目录：/root/c10_region_restore_20261029_v1
队列PID：{queue.get('queue_pid')}；当前D0 PID：{live['pid']}。PID只用于本次快照，后续须结合命令行、状态文件与更新时间核实。
远端状态：receipts/queue_live.json；训练状态：results/D0/live_status.json及results/DA/live_status.json；逐轮真实计数：各自epoch_audit.jsonl；日志：logs/train_D0.log / logs/train_DA.log。完成后结果：evaluation/initial_diagnostic_result.json。失败不自动重启。

评价实现校正：首次科学评价前补充了原生边界裁剪一致性检查；原图AP与输入面积分组共享同一组原生scale_boxes裁剪后的预测，后者仅将坐标重投影。人工跨padding样例在本地及服务器通过。训练源码、初始化、数据划分、预算和超参数均未改变，四个训练相关源码SHA已逐项核对。原评价源码和修正前后SHA保留在source_reference和evaluator_amendment_v1.json；启动时的冻结清单仍保留原值，修正单独记录，未覆盖历史。

本地回执：{ROOT/'server_receipts'}
预检：server_receipts/preflight_pair.json；数据：data_preparation.json、split_manifest.json；环境：environment.json、pip_freeze.txt；初始化：common_initialization.json；开训：D0_start_receipt.json；代码冻结：code_frozen_at_launch.json；本轮计数：execution_ledger.json。

准备过程中保留了网络中断、未完整分段、元数据检查器、数据回执变量名冲突和重复缩放测试的失败日志。相关问题均在正式开训前修正并通过相应检查；没有将失败样例或早期AP作为算法收益。凭据未写入报告、脚本、参数文件或磁盘凭据文件。
"""
(ROOT.parent/'15_远端实验_运行状态与冻结方案_20261029.txt').write_text(report,encoding='utf-8')
print(json.dumps({'status':ledger['status'],'D0':{'epoch':live['epoch'],'batches':live['batches'],'updates':live['actual_optimizer_updates']},
                  'report':str(ROOT.parent/'15_远端实验_运行状态与冻结方案_20261029.txt')},ensure_ascii=True))
