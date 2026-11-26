"""Archive and verify research notes only; never execute models or datasets."""
from pathlib import Path
import ast, hashlib, json, shutil
from datetime import datetime, timezone

root=Path(__file__).resolve().parent
base=root.parent
now=datetime.now(timezone.utc).isoformat()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
supp=json.loads((root/'supplementary_literature_records.json').read_text(encoding='utf-8'))
board=json.loads((root/'candidate_board.json').read_text(encoding='utf-8'))
query=json.loads((root/'query_ledger.json').read_text(encoding='utf-8'))
audit=json.loads((root/'annotation_size_metadata_audit.json').read_text(encoding='utf-8'))
assert [r['id'] for r in supp['records']]==['R'+str(i) for i in range(21,35)]
assert board['mainline_found'] is False and board['goal_status']=='active'
assert query['new_exact_queries_logged']==38
assert audit['overall']['gt']==343204 and audit['invalid_boxes']==1
assert sha(Path(audit['source_path']))==audit['source_sha256']
assert all(v==0 for v in audit['execution'].values())

inventory=[]
receipts=json.loads((root/'document_reading_receipts.json').read_text(encoding='utf-8'))
for row in receipts:
    path=Path(row['path'])
    assert path.exists() and sha(path)==row['sha256']
    text_path=Path(row['text_path']); preflight_path=Path(row['preflight_path'])
    assert text_path.exists() and preflight_path.exists()
    evidence_id='R15' if row['key'].startswith('R15') else row['key'].split('_')[0]
    scope=supp['existing_record_reading_updates'][0]['reading'] if evidence_id=='R15' else next(x['reading'] for x in supp['records'] if x['id']==evidence_id)
    if row['key']=='R21_TTAC_arxiv_v2':
        scope='结构UNAVAILABLE；只保留原件，不以该PDF页码支撑核心方法。使用CVF版及HTML选读。'
    inventory.append({**row,'reading_scope':scope,'preservation_hash_verified':True})
assert len(inventory)==9
(root/'source_manifest_and_reading.json').write_text(json.dumps({'verified_at_utc':now,'documents':inventory,
    'rule':'结构PASS只说明结构可读；实际阅读范围按每条reading_scope，未独立复现论文。'},ensure_ascii=False,indent=2),encoding='utf-8')

history=root/'历史版本'
history.mkdir(exist_ok=True)
main=base/'00_调研报告.txt'
backup=history/'00_调研报告_v3_补检前.txt'
prefix=('后续补检入口（2026-10-28）：09_持续找主线_证据淘汰与当前判断.txt。\n'
        '以下v3正文保留为前阶段记录；新增R21–R34及最新阅读范围在主线目标_20261028_v1/。'
        '持续目标尚未完成，尚未认可独立算法增量；已有实验方案仍未执行。\n\n')
text=main.read_text(encoding='utf-8')
if not text.startswith('后续补检入口（2026-10-28）'):
    assert not backup.exists(), 'Preserve existing backup rather than overwrite'
    shutil.copy2(main,backup)
    main.write_text(prefix+text,encoding='utf-8')

scripts=[]
for path in sorted(root.glob('*.py')):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.append(node.module)
    assert not any(x.split('.')[0] in {'torch','ultralytics','tensorflow','onnxruntime'} for x in imports)
    scripts.append({'path':str(path),'sha256':sha(path),'syntax':'AST parsed only; script not imported','imports':sorted(set(imports))})

ledger={'updated_at_utc':now,'research_date_client':'2026-10-28','workflow':'academic-research-suite/deep-research/targeted lit review',
 'single_assistant':True,'subagents':0,'goal_status':'active','mainline_found':False,
 'prior_run_ledger':str(base/'深入核查_20261028_v1'/'research_run_ledger.json'),
 'goal_objective':'找到具体问题、最接近机制、实质算法增量及24GB/数天预算内的验证淘汰路线；允许否定JPEG方向；暂不实验。',
 'current_deliverables':[str(base/'09_持续找主线_证据淘汰与当前判断.txt'),str(root/'01_新增核心文献矩阵.txt'),str(root/'candidate_board.json'),str(root/'query_ledger.json'),str(root/'annotation_size_metadata_audit.json')],
 'bibliography':'此前R1–R20加补充R21–R34；R15补读单列，未重复计为新论文。',
 'evidence_limits':['R30仅摘要，尚未完成近邻全文排重','R31–R34仅选段或元数据，不能冒充全文已读','TTAC arxiv PDF结构UNAVAILABLE，核心使用CVF PASS版/HTML','静态标注尺寸统计不能证明精度影响','新增候选无认可算法增量'],
 'next_step':'优先完成C5与LAD/退化分配/目标冲突/跨域匹配的机制比较；若只是旧方法迁移即淘汰JPEG+分配作为算法主线，转查其他UAV具体瓶颈。',
 'execution':{'model_construction':0,'model_weight_loads':0,'model_imports':0,'model_forwards':0,'training_runs':0,'dataset_image_reads':0,'dataset_image_encodes':0,'augmentations_run':0,'random_simulations':0,'GPU_jobs':0,'remote_jobs':0,'test_or_test_dev_reads':0,'test_or_test_dev_evaluations':0},
 'nonexperimental_operations':['公开论文和源码下载','论文PDF解析/局部页渲染阅读','静态源码阅读','既有训练标注数字的确定性几何计算','研究记录写入和哈希核验'],
 'approval':'用户授权持续前期调研；没有新增实验授权，目标不构成实验启动指令。'}
(root/'research_run_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')

required=['00_目标与验收标准.txt','01_新增核心文献矩阵.txt','supplementary_literature_records.json','candidate_board.json','query_ledger.json','web_evidence_goal_increment.json','annotation_size_metadata_audit.json','document_reading_receipts.json','source_manifest_and_reading.json','research_run_ledger.json']
files=[]
for name in required:
    path=root/name
    assert path.exists() and path.stat().st_size>0
    if path.suffix=='.json': json.loads(path.read_text(encoding='utf-8'))
    files.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
for path in [main,backup,base/'09_持续找主线_证据淘汰与当前判断.txt']:
    assert path.exists()
    files.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
receipt={'verified_at_utc':now,'verdict':'PASS for artifact integrity and boundary bookkeeping only; mainline remains unachieved',
 'new_literature_records':len(supp['records']),'existing_record_reading_updates':len(supp['existing_record_reading_updates']),
 'PDF_receipts_verified':len(inventory),'structural_PASS':sum(x['structural_verdict']=='PASS' for x in inventory),
 'structural_UNAVAILABLE':sum(x['structural_verdict']=='UNAVAILABLE' for x in inventory),
 'abstract_only_ids':[x['id'] for x in supp['records'] if x['abstract_only']],
 'annotation_source_unchanged':True,'new_scripts_static_syntax_audit':scripts,'files':files,
 'execution':ledger['execution'],'mainline_found':False,'goal_status':'active','independent_reviewer':False}
(root/'goal_progress_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['verdict','new_literature_records','PDF_receipts_verified','structural_PASS','structural_UNAVAILABLE','abstract_only_ids','annotation_source_unchanged','mainline_found','goal_status','execution']},ensure_ascii=True))
