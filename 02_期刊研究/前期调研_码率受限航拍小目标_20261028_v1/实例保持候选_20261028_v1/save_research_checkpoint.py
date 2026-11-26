"""Persist source provenance and research decisions only. No experiment code."""
from pathlib import Path
import json,hashlib,datetime

root=Path(__file__).resolve().parent
base=root.parent
prior=base/'主线目标_20261028_v1'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
records=read(root/'supplementary_literature_records_R35_R45.json')['records']
receipts=read(root/'document_reading_receipts.json')
assert len(records)==11 and len({x['id'] for x in records})==11
assert len(receipts)==8
manifest=[]
for row in receipts:
    p=Path(row['path'])
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==row['sha256']
    rid=next(x['id'] for x in records if row['key'].startswith(x['id']+'_'))
    source=next(x for x in records if x['id']==rid)
    manifest.append({**row,'reading_scope':source['reading_scope'],'reference_id':rid,
                     'hash_verified_utc':now,'note':'Structure PASS is not full reading or an independent reproduction.'})
write(root/'source_manifest_and_actual_reading.json',manifest)

history=prior/'历史版本'
history.mkdir(exist_ok=True)
board_path=prior/'candidate_board.json'
old_board=read(board_path)
backup=history/'candidate_board_C5优先阶段_20261028.json'
if not backup.exists():write(backup,old_board)
board=old_board
board.update(updated_at_utc=now,goal_status='active',mainline_found=False,
             local_empirical_claims=False,
             qualification='No independent algorithm mainline approved; bounded second batch completed.')
for c in board['candidates']:
    if c['id']=='C5':
        c['status']='downgraded_no_distinct_algorithm_or_observed_mechanism'
        c['reason']='JPEG+分配的独立观测及不同动作逻辑均未闭合；LAD/CrossKD等约束明确。不是证明现象不存在；R30仍仅摘要。'
        c['algorithm_increment']=None
new=[
 {'id':'C7','question':'恢复前后对象身份得失是否被整图净指标掩盖，并影响小目标标准检测表现？',
  'status':'lead_only_not_promoted','prior':['R35','R36','R37','R38','R39','R40','R41','R45'],
  'reason':'净计数身份交换只是逻辑例；一般选择/相容蒸馏/风险控制/局部失败图/原始恢复融合已有先例。无认可算法差异。',
  'algorithm_increment':None,'forbidden_inheritance':'不继承Q01零新增TP损失闸门，不以该要求制造创新。'},
 {'id':'C8','question':'UAV复合退化恢复后，已适配检测器的小目标剩余缺口是什么？',
  'status':'direct_UAV_lead_not_mainline','prior':['R42','R44','R45'],
  'reason':'冻结检测器总体结果不能证明适配后问题；MDUR专属资产/补充协议未取得，小目标分组未核。',
  'algorithm_increment':None},
 {'id':'C9','question':'多模块在航拍检测中负组合的原因是否超过初始化/容量/实现混杂？',
  'status':'deprioritized_engineering_and_protocol_confounds','prior':['R43'],
  'reason':'主表基线取自文献、消融是不同重实现结构、初始化解释未受控验证；不作为算法首选。',
  'algorithm_increment':None}
]
board['candidates']=[c for c in board['candidates'] if c['id'] not in {x['id'] for x in new}]+new
board['next_steps']=[
 '已完成两批候选裁决；不无界扩展，不重做同卡审计。',
 '仅能改变裁决的原始证据或具体不同的算法逻辑才算后续推进；比较缺口本身不是算法贡献。',
 '07/08仅为未来条件性问题筛选方案，实验仍须用户另行授权。'
]
board['latest_report']=str(base/'10_候选降级与第二批UAV证据.txt')
write(board_path,board)
write(root/'candidate_board_updated.json',board)

ledger_path=prior/'research_run_ledger.json'
ledger=read(ledger_path)
backup=history/'research_run_ledger_R21_R34阶段_20261028.json'
if not backup.exists():write(backup,ledger)
ledger.update(updated_at_utc=now,goal_status='active',mainline_found=False,
              bibliography='此前R1–R34；本轮补充R35–R45，明确选读/摘要/日期边界。',
              current_deliverables=[str(base/'10_候选降级与第二批UAV证据.txt'),
                                    str(root/'01_新增文献矩阵_R35_R45.txt'),
                                    str(root/'02_候选逐项裁决与可证伪问题卡.txt'),
                                    str(board_path),str(root/'query_ledger_continuation.json'),
                                    str(root/'source_manifest_and_actual_reading.json')],
              next_step='两批未通过；仅推进能改变裁决的原始证据或实质算法逻辑。不得强凑主线，亦无新增实验授权。',
              evidence_limits=[
               '没有本项目新模型结果，所有文献数字均为作者报告，未复现。',
               'R35–R45均为选段或摘要/预览，不冒充全文通读。',
               'R41出版商所示卷期晚于研究日期，online日期未独立核实。',
               'MDUR原始资源入口未定位到专属资产，不宣称数据全球不可取。',
               'LAF主表跨论文基线不能解释为受控负效果。',
               '208个文件名前缀未经证实为独立场景，不提供形式化CRC保证。',
               '候选C7仅逻辑盲点，没有适配后的UAV实例级现象或认可算法。'])
assert all(v==0 for v in ledger['execution'].values())
write(ledger_path,ledger)
write(root/'research_run_ledger.json',ledger)

report=base/'00_调研报告.txt'
text=report.read_text(encoding='utf-8-sig')
marker='码率受限的无人机航拍小目标检测：'
assert marker in text
old=history/'00_调研报告_补充R35前.txt'
if not old.exists():old.write_text(text,encoding='utf-8')
prefix=('最新判断入口（2026-10-28）：10_候选降级与第二批UAV证据.txt。\n'
        '此前候选排序已更新：JPEG+分配降级；目前未认可独立算法主线。持续目标active，未运行实验。\n'
        '以下v3正文保留为历史阶段记录；当前判断、R35–R45及阅读边界以最新入口和实例保持候选_20261028_v1为准。\n\n')
report.write_text(prefix+text[text.index(marker):],encoding='utf-8')

required=[Path(p) for p in ledger['current_deliverables']]
for p in required:assert p.is_file() and p.stat().st_size>0
checks={'checked_at_utc':now,'research_date_client':'2026-10-28',
        'artifact_integrity':'PASS','source_pdfs_hash_verified':len(manifest),
        'references_added':len(records),'structural_PASS_not_full_reading':True,
        'mainline_found':False,'goal_status':'active','experiments_run':False,
        'scope':'Only source provenance, document links, and decision consistency. No scientific result validation.',
        'artifacts':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in required]}
write(root/'checkpoint_verification.json',checks)
print(json.dumps({k:checks[k] for k in ['artifact_integrity','source_pdfs_hash_verified','references_added','mainline_found','goal_status','experiments_run']},ensure_ascii=False))
