from source_fetch import ROOT, ARCHIVE, fetch
import json

path=ROOT/'00_调研报告.txt'
text=path.read_text(encoding='utf-8')
old='开发检测器：暂建议一个固定版本的YOLOv8s、固定640输入，作为训练侧方法的开发对象；公开实现与近邻方法的接口较易对接。它是工程选择，未被证明是压缩条件下的最优架构。版本/预训练权重/完整训练配方需在未来启动前冻结。'
new='开发检测器（本轮更正）：优先建议YOLO26s、固定640输入；YOLOv8s作为可选跨版本参照。YOLO26已有公开代码/模型，STAL直接面向小目标正样本覆盖，JCRR也使用YOLO26s。此前偏向YOLOv8s的工程理由不足以排除YOLO26。具体版本/权重/训练配方/推理路径需冻结，尚未实测航拍压缩效果和训练成本，见05_YOLO26选择与选题来源说明.txt。'
if old in text: text=text.replace(old,new)
text=text.replace('数据与开发基线可以暂定VisDrone2019-DET + YOLOv8s','数据与开发基线现更正为VisDrone2019-DET + YOLO26s')
text=text.replace('文献编号R1–R18对应第7、9部分','文献编号R1–R19对应第7、9、10部分')
text=text.replace('01_核心文献矩阵.txt：18条','01_核心文献矩阵.txt：19条')
if '[W3]' not in text:
 text+='\n10. YOLO26基线选择更正与问题来源\n\n当前优先开发基线更正为YOLO26s。其STAL等已有小目标监督设计应进入问题成立条件；此前选YOLOv8s偏保守，没有充分比较依据。直接近邻JCRR也已经使用YOLO26s/n。关于选题初筛、用户确认、近邻查重后降级的实际过程，见05_YOLO26选择与选题来源说明.txt。当前仍是文献筛出的候选问题，没有新模型诊断或已确认的算法主线。\n\n[W3] Ultralytics官方YOLO26文档：https://docs.ultralytics.com/models/yolo26/\n[R19] Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., & Kalfaoglu, M. E. (2026). Ultralytics YOLO26: Unified real-time end-to-end vision models. arXiv:2606.03748v1. https://arxiv.org/abs/2606.03748v1\n范围：论文仅摘要/元数据；官方文档选读Overview、Key Features、性能表、架构和推理路径。没有据COCO结果推定VisDrone压缩性能。\n'
path.write_text(text,encoding='utf-8')

path=ROOT/'04_主线收敛与数据算法选择.txt'
text=path.read_text(encoding='utf-8')
text=text.replace('开发基线暂定：YOLOv8s，固定640输入和明确的代码版本/预训练权重。','开发基线更正为：YOLO26s，固定640输入和明确的代码版本/预训练权重。')
text=text.replace('理由：公开实现与数据处理入口成熟；与直接JPEG研究的检测器家族容易比较；单模型训练侧修改较容易控制额外推理成本。','理由：官方已有公开代码/模型与训练入口；STAL已有小目标正样本覆盖设计，应在此基础上核实问题是否仍存在；直接近邻JCRR也使用YOLO26s/n。之前优先YOLOv8s的工程理由不足以排除YOLO26。')
text=text.replace('YOLO11/YOLO26可以成为后续近邻复核对象；当前没有证据表明“更新版本”能替代对压缩问题的机制研究。','YOLOv8s可作为后续跨版本参照；架构对照数量按预算取舍，不自动增加完整矩阵。YOLO26的COCO结果不能证明航拍压缩鲁棒性。具体解释见05_YOLO26选择与选题来源说明.txt。')
text=text.replace('数据/基线：VisDrone2019-DET + YOLOv8s。','数据/基线：VisDrone2019-DET + YOLO26s。')
path.write_text(text,encoding='utf-8')

records=json.loads((ROOT/'literature_records.json').read_text(encoding='utf-8'))
record={'id':'R19','tier':'基线选择依据','short':'YOLO26','authors':'Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., & Kalfaoglu, M. E.','year':2026,'title':'Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models','venue':'arXiv:2606.03748v1','doi':'10.48550/arXiv.2606.03748','url':'https://arxiv.org/abs/2606.03748v1','data':'摘要主要COCO/LVIS等；不是UAV压缩专项','algorithm':'双头、DFL-free、MuSGD、Progressive Loss、STAL等','mutable':'基础检测器与训练配方','rate':'没有在本轮阅读范围得到JPEG实际bpp结果','size':'STAL针对小目标正样本覆盖；不是已确认的压缩漏检解决方案','evidence':'公开代码模型和小目标监督机制支持将其作为优先基线；不据COCO性能外推VisDrone压缩效果','limit':'未运行/未通读完整论文；官方文档的NMS/端到端两种路径需分别冻结','reading':'论文仅官方摘要/元数据；另读官方文档Overview、Key Features、支持任务/架构、Detection性能表和推理路径选段。','pdf_pattern':'20261028_码率调研_YOLO26_arxiv_v1.pdf','abstract_only':True}
if not any(r['id']=='R19' for r in records): records.append(record)
(ROOT/'literature_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'model_choice_followup.json').write_text(json.dumps({'date':'2026-10-28','user_questions':['为什么不用最新的yolo26呢','另外 这个主线你是怎么找出来的？'],'decision':'YOLO26s prioritized; YOLOv8s optional cross-version reference','sources':['https://docs.ultralytics.com/models/yolo26/','https://github.com/ultralytics/ultralytics','https://arxiv.org/abs/2606.03748v1'],'experiments':0,'reading_scope':record['reading']},ensure_ascii=False,indent=2),encoding='utf-8')
receipt=fetch('YOLO26_arxiv_v1','https://arxiv.org/pdf/2606.03748v1')
(ROOT/'retrieval_yolo26.json').write_text(json.dumps([receipt],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'baseline':'YOLO26s','archive_status':receipt['status'],'error':receipt.get('error')},ensure_ascii=True))
