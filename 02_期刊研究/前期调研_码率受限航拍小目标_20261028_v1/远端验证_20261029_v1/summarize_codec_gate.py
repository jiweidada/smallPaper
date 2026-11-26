"""Verify codec-control backup and write a self-contained exploratory result note."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,tarfile
local=Path(__file__).resolve().parent;base=local.parent;folder=local/'完成结果_20261029_v1'
receipt=json.loads((folder/'codec_archive_receipt.json').read_text());archive=folder/'codec_gate_completed.tar.gz'
assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
target=(folder/'codec_raw').resolve();target.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz') as tar:
    for member in tar.getmembers():
        name=PurePosixPath(member.name);assert not name.is_absolute() and '..' not in name.parts
        assert ':' not in member.name and '\\' not in member.name and (member.isfile() or member.isdir())
        path=target.joinpath(*name.parts).resolve();assert target in path.parents
        if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
        path.parent.mkdir(parents=True,exist_ok=True)
        with tar.extractfile(member) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
r=target/'simple_codec_gate_v1';complete=json.loads((r/'completed.json').read_text());boot=json.loads((r/'bootstrap_completed.json').read_text())
frozen=complete['frozen'];assert hashlib.sha256((r/'run_gate.py').read_bytes()).hexdigest()==frozen['script_sha256']
assert hashlib.sha256((r/'protocol.json').read_bytes()).hexdigest()==frozen['protocol_sha256']
obs=json.loads((r/'observations.json').read_text())
for p in obs['payloads']:
    remote=Path(p['file']);relative=PurePosixPath(p['file']).relative_to('/root/c10_region_restore_20261029_v1')
    path=target.joinpath(*relative.parts);assert hashlib.sha256(path.read_bytes()).hexdigest()==p['sha256']
    if p['kind']=='half_jpeg':assert path.with_suffix('.dims').stat().st_size==8
now=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-7))).isoformat()
original=json.loads((folder/'raw/evaluation/initial_diagnostic_result.json').read_text())
lines=['## Material Passport','','- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run + exploratory statistical interpretation',
       '- Origin Date: 2026-10-29；America/Los_Angeles','- Verification Status: 归档/源码/全部200个新载荷SHA核验；cap500前缀完全复现；未进行独立硬件/seed复现','- Version Label: codec_gate_v1','',
       '简单编码与评价上限：完成结果及主线判断','',f'整理时间：{now}。本轮新增训练更新0，检测前向800；bootstrap仅重用预测，不做模型前向。','',
       '一、实际检索与验证策略','',
       '先冻结方案，检查1000框上限；用既定调参100的总实际字节量选择公共质量，再在固定诊断100上评价，不按诊断AP选质量。',
       'WebP原尺寸、lossless=False、method4；半尺寸JPEG先以INTER_AREA缩小，再JPEG，另计8字节原尺寸元数据；接收端直接重建既定letterbox有效区。',
       '质量网格为1及5:5:95，以最小总字节绝对偏差选定。校准无检测器查询或GT效用。固定点为WebP25/半尺寸JPEG80。',
       '环境Pillow11.3.0、libwebp1.5.0；源图本身为JPEG。本轮没有改训练、数据划分、主要maxDet500口径或原始科学预测。','',
       '二、真实传输字节','',
       '条件 | 调参集相对JPEG Q20总字节差 | 诊断集总字节 | 诊断集相对差',
       'JPEG Q20 | 基准 | 8003692 | 基准']
cal=json.loads((r/'calibration.json').read_text())
for cond in ['webp','half_jpeg']:
    rate=complete['rates'][cond];lines.append(f"{cond} | {100*cal['selected_relative_rate_drift'][cond]:+.4f}% | {rate['total_bytes']} | {100*rate['relative_byte_drift']:+.4f}%")
lines+=['','【限制】这是聚合字节量的相近比较，不是每图等字节或精确固定码率；WebP诊断偏差接近预定5%上沿。半尺寸JPEG调参偏差5.59%超5%，诊断为2.95%，两者都保留，未根据AP重新细调。',
        '单图字节漂移分布和编码/解码墙钟均保存在observations.json；未做专门CPU测速控制、重复或端到端无线时延测量，不能把这些墙钟当部署延迟保证。','',
        '三、主要小目标AP结果','',
        '以下均为输入坐标GT面积<32²的AP50:95、maxDet500、内部COCO式crowd口径，不是官方挑战成绩。source是参考JPEG。',
        '组别 | source(%) | JPEG Q20(%) | WebP25(%) | 半尺寸JPEG80(%)']
for arm in ['D0','DA']:
    vals=[100*original[arm]['source']['AP_small_input_500'],100*original[arm]['q20']['AP_small_input_500']]+[100*complete['metrics'][arm][c]['AP_small_input'] for c in ['webp','half_jpeg']]
    lines.append(arm+' | '+' | '.join(f'{x:.4f}' for x in vals))
lines+=['','DA的配对图像bootstrap（2000次，seed20261029；单位pp）：','比较 | 点差 | 95%区间']
for key,v in boot['comparisons']['DA'].items():
    ci=v['paired_image_bootstrap95_pp'];lines.append(f"{key} | {v['point_pp']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}]")
lines+=['','【观测】DA半尺寸JPEG相对Q20仅+0.2273pp，区间跨0；WebP点估计−0.5588pp，区间也跨0。不能说缩图稳定改善，也不能说JPEG普遍优于WebP。',
        '【观测】DA半尺寸JPEG相对source仍低0.8583pp，区间[0.1357,1.8684]pp；这弱化了“剩余损失至少1pp”的动机，但并未证明差距已消失。',
        '【推断】简单对照没有给出足够确定的淘汰或正面算法结论。只允许继续有限冻结恢复诊断，不据此训练新策略。',
        '【统计范围】两个arm共10个探索性点差区间，未校正为同时置信覆盖，不据它们宣称确认性显著发现。单seed、相关帧、有限样本、训练期覆盖dev300 source仍是限制。','',
        '四、评价框数敏感性','',
        '组别 | 预定cap500的source−Q20差(pp) | cap1000差(pp)']
for arm in ['D0','DA']:lines.append(f"{arm} | {original[arm]['remaining_small_AP_loss_pp']:.4f} | {complete['metrics'][arm]['source_q20_cap1000_gap_pp']:.4f}")
lines+=['','四个source/Q20的cap1000预测前500框均精确复现原先主要小目标AP；confidence0.25下TP/FN/FP也一致。主差距并未因放宽框数消失。保留cap500为主结果。','',
        '五、最接近工作及当前增量边界','',
        'TaskGuard已用冻结恢复器/检测器、区域反事实和伪梯度，最终作整图restore/preserve。本文候选只可能落在真实单/双区干预标签与联合子集选择，而不是“恢复风险”或“冻结任务反馈”。',
        'RFDF出版商预览已明确采用识别相关区域与多代理融合，完整算法尚未取得；不能声称排除了机制重合。',
        'RTScale已按图像尺度敏感性与时间约束选择缩放；ROI-Packing已有ROI提取、缩放、打包/解包与VVC。若缩图有效，也不把它改名包装为新的检测算法。',
        '阅读范围：TaskGuard方法§3、附录B/C/E选读；RFDF仅出版商摘要/片段；RTScale仅出版商摘要/元数据；ROI-Packing作者arXiv v1的§II方法与实验表选读，未通读PDF或核作者代码。','',
        '六、下一步及可证伪条件','',
        '冻结FBCNN原尺寸容量已通过。正式区域判别比较不恢复、全恢复、3档额外全局alpha和四区16子集，以及GT定义的独立/二阶/有限损失最优选择。另固定JPEG分配结果重算损失，以区分分配变化。',
        'GT选择只作诊断。若DA联合选区相对独立和最强固定全局alpha没有至少0.5pp小目标AP潜力，或仅loss变化而无真实检测收益，则不进入策略训练。该筛选阈值在区域结果出现前冻结。',
        '若联合空间明确，也还须查量化/羽化伪影、跨图学习、同监督容量对照、成本与最近方法，不将GT诊断当已找到可部署创新。','',
        '七、核验入口与参考链接','',
        f'已核验归档：{archive}',f'归档SHA256：{receipt["sha256"]}',f'展开目录：{r}',
        '原始结果：completed.json；校准：calibration.json；字节/图像SHA：observations.json；区间：bootstrap_completed.json；预测：evaluation/。',
        '方案与程序SHA见frozen.json；人工几何/cap接口检查见fixture.json。','',
        'TaskGuard：https://arxiv.org/html/2609.08011v1',
        'RFDF：https://www.sciencedirect.com/science/article/pii/S0263224126031349',
        'RTScale：https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2022.2',
        'ROI-Packing：https://arxiv.org/html/2512.09258v1',
        'FBCNN官方源码：https://github.com/jiaxi-jiang/FBCNN/tree/54d1831927506b3247e2d4d245abb4f4dab1a1cd','',
        '统计解释检查11/11已逐项覆盖：分组方向不混同、总体不推个体、样本选择边界、无结果后筛图、GT/FP基数、非极端筛选前后比较、100图无掉样、全部10个对比保留、探索路径和实现修正披露、固定checkpoint干预不泛化为真实飞行因果、无反向因果主张。未建立独立场景抽样或多重比较保证，检查覆盖不等于这些假设已满足。']
(base/'17_简单编码对照与主线门槛_20261029.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(folder/'codec_backup_verification.json').write_text(json.dumps(dict(status='ARCHIVE_CODE_AND_200_PAYLOADS_VERIFIED',sha256=receipt['sha256'],bytes=receipt['bytes'],completed_detector_forwards=800),indent=2))
print(json.dumps(dict(status='CODEC_REPORT_WRITTEN',archive_verified=True,report=str(base/'17_简单编码对照与主线门槛_20261029.txt'))))
