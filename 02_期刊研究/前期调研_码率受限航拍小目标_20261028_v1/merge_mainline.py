from source_fetch import ROOT
import json

path=ROOT/'literature_records.json'
records=json.loads(path.read_text(encoding='utf-8'))
for r in json.loads((ROOT/'literature_mainline_supplement.json').read_text(encoding='utf-8')):
 if r['id'] not in {p['id'] for p in records}: records.append(r)
path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

fields=[('研究位置','tier'),('文献','title'),('作者','authors'),('年份','year'),('发表载体/版本','venue'),('原始链接','url'),('数据与任务','data'),('方法','algorithm'),('可变部分','mutable'),('码率口径','rate'),('小目标口径','size'),('原文证据','evidence'),('限制','limit'),('实际阅读范围','reading')]
text='核心文献矩阵与阅读范围\n日期：2026-10-28（America/Los_Angeles）\nR1–R8为原主干，R9–R14为方法/边界/数据依据，R15–R18为主线追加查重，R19为YOLO26基线依据。不是系统综述的完整收录集。\n非仅摘要仍可能只是出版商预览选段，不能理解为全文阅读；PDF下载/自动提取不等于全文已读。\n原文证据是对原文的转述，不是本项目实验结果。\n\n'
for r in records:
 text+=f"{r['id']} | {r['short']} | {r['year']}\n"
 for label,key in fields: text+=f'{label}：{r[key]}\n'
 text+=f"DOI：{r['doi'] or '本轮未核定，保留官方链接，不猜填'}\n仅摘要：{'是' if r['abstract_only'] else '否；以实际阅读范围为准'}\n\n"
(ROOT/'01_核心文献矩阵.txt').write_text(text,encoding='utf-8')

logpath=ROOT/'02_检索与证据记录.json'
log=json.loads(logpath.read_text(encoding='utf-8'))
for row in json.loads((ROOT/'search_mainline_supplement.json').read_text(encoding='utf-8')):
 if row['id'] not in {b['id'] for b in log['batches']}: log['batches'].append(row)
log['resource_clarification']={'question':'后续验证的训练资源上限大致是什么？本轮仍只做调研，这会影响算法和验证路线的选择。','answer':'单卡约24GB，训练时间可接受数天','meaning':'Planning constraint; no experiment authorization added.'}
log['scope_updates']=['继续这个方向找主线','看看什么数据集 什么算法','当前的问题 找到了是什么吗？']
log['access_failures']=[{'targets':['https://www.nature.com/articles/s41598-026-63984-0','https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70597','https://docs.ultralytics.com/reference/utils/tal/'],'status':'one batch fatal connection failure; no page content returned'}]
if (ROOT/'model_choice_followup.json').exists(): log['model_choice_followup']=json.loads((ROOT/'model_choice_followup.json').read_text(encoding='utf-8'))
logpath.write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'references':len(records),'query_batches_including_failed_preliminary':len(log['batches'])},ensure_ascii=True))
