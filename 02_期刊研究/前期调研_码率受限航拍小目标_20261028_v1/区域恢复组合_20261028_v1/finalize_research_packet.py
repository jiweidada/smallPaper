"""Persist a proposed research line, provenance, and scope. No models or experiments."""
from pathlib import Path
import json,hashlib,datetime
root=Path(__file__).resolve().parent
base=root.parent
prior=base/'主线目标_20261028_v1'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

card_path=root/'02_主线问题卡_C10.json'
card=read(card_path)
card['algorithm_hypothesis']['utility_definition']=read(root/'03_效用定义与源码依据.json')
card['status']='selected_preexperimental_mainline_pending_completion_audit'
card['report']=str(base/'11_区域联合恢复_研究主线与最小判别.txt')
write(card_path,card)
refs=read(root/'literature_records_R46_R53.json')['records']
reference_ids={r['id'] for r in refs}
assert reference_ids=={f'R{i}' for i in range(46,54)}
docs=read(root/'document_receipts.json')
manifest=[]
for row in docs:
    p=Path(row['path']);assert sha(p)==row['sha256']
    rid={'PixelRL_arxiv':'R47','FBCNN':'R48'}[row['key']]
    scope=next(r['reading_scope'] for r in refs if r['id']==rid)
    manifest.append({**row,'reference_id':rid,'actual_reading_scope':scope,'hash_rechecked_utc':now})
archive=base.parent/'文献_相关原始论文_20261004'
for rid,name,scope in [
 ('R1','20261028_码率调研_JCRR_publisher.pdf','Previously read original tables; this pass reread pp4–6 and text of Tables2/3 (pp8–9), not full paper.'),
 ('R35','20261028_实例保持_TaskGuard_arxiv_v1.pdf','Earlier method/table visual reading retained; this pass reread discussion and AppendixB selected; not full 15 pages.'),
 ('R5','20261028_码率调研_RD_MIM_official_CDN.pdf','This pass read §4.1/§4.2 and Tables4–7/selected method text; corruptions test-only clarified, not full reading.')]:
    p=archive/name
    assert p.is_file()
    manifest.append({'reference_id':rid,'path':str(p),'sha256':sha(p),'actual_reading_scope':scope,'hash_rechecked_utc':now})

code_root=base/'深入核查_20261028_v1'/'source_code'/'v8.4.191_2004cba021c6'
pin=read(code_root/'pin.json')
assert pin['commit']=='2004cba021c68a46429be460e26954ab171b2862'
code_receipts=read(base/'深入核查_20261028_v1'/'retrieval_receipts.json')
for rel,locators in [('ultralytics/utils/loss.py','355–390,423–532,1319–1351'),('ultralytics/nn/modules/head.py','169–204')]:
    p=code_root/rel
    expected=[r for r in code_receipts if r.get('status')=='downloaded' and r.get('path')==str(p)]
    assert expected and sha(p)==expected[-1]['sha256']
    manifest.append({'reference_id':'R19/static-utility-audit','path':str(p),'sha256':sha(p),'commit':pin['commit'],'locators_read':locators,'execution':'text only; no import/model construction'})
write(root/'source_manifest_and_reading.json',manifest)

history=prior/'历史版本';history.mkdir(exist_ok=True)
bp=prior/'candidate_board.json';board=read(bp)
backup=history/'candidate_board_C5至C9裁决后_20261028.json'
if not backup.exists():write(backup,board)
entry={'id':'C10','question':card['target_question'],'status':'selected_preexperimental_mainline',
       'prior':['R1','R3','R35','R46','R47','R48','R49','R52','R53'],
       'reason':'从公开区域效用异质性形成联合恢复问题；已给出单/双区干预监督、联合效用决策和强简单反证。UAV JPEG交互及算法有效性待验证。',
       'algorithm_increment':'拟学习显式成对条件收益并联合选择固定恢复残差；不以门控/MLP/二阶展开本身为新颖性。',
       'evidence_status':'published/preprint external observations + explicitly separated inference and hypothesis',
       'empirically_validated':False,'novelty_exhaustively_proven':False,
       'card':str(card_path),'report':card['report']}
board['candidates']=[x for x in board['candidates'] if x['id']!='C10']+[entry]
board.update(updated_at_utc=now,mainline_found=True,mainline_stage='preexperimental research hypothesis',
             goal_status='active_pending_completion_audit',selected_candidate='C10',
             algorithm_effect_confirmed=False,local_empirical_claims=False,
             qualification='Found means a literature-supported falsifiable research line, not an empirically verified innovation.',
             latest_report=card['report'])
board['next_steps']=['No further broad candidate expansion in this report.',
                     'Future minimum validation requires separate explicit experiment authorization.',
                     'If any stopping condition in C10 holds, adopt simple prior methods or retire C10; do not stack modules.']
write(bp,board);write(root/'candidate_board.json',board)

lp=prior/'research_run_ledger.json';ledger=read(lp)
backup=history/'research_run_ledger_R35_R45阶段_20261028.json'
if not backup.exists():write(backup,ledger)
ledger.update(updated_at_utc=now,mainline_found=True,mainline_stage='preexperimental research hypothesis',
              goal_status='active_pending_completion_audit',selected_candidate='C10',
              bibliography='R1–R45 plus R46–R53; exact reading scopes and primary links retained.',
              current_deliverables=[card['report'],str(root/'01_文献矩阵_R46_R53.txt'),str(card_path),
                                    str(root/'03_效用定义与源码依据.json'),str(root/'query_ledger.json'),
                                    str(root/'source_manifest_and_reading.json'),str(root/'static_resource_receipt.json')],
              next_step='Research-stage completion audit. No model/encoding/training authorization exists.',
              evidence_limits=['Spatial heterogeneity observed externally; non-additivity in UAV JPEG is a hypothesis.',
                               'Ordinary adaptation may eliminate the remaining issue; must be tested first.',
                               'R49 full method unavailable; no exhaustive novelty claim.',
                               'FBCNN weights not downloaded or verified; JCRR supplement unavailable.',
                               'GPU capacity and timing not measured; static storage arithmetic is not peak memory.',
                               'No new empirical model or algorithm result has been produced.'])
assert all(v==0 for v in ledger['execution'].values())
write(lp,ledger);write(root/'research_run_ledger.json',ledger)

main=base/'00_调研报告.txt';text=main.read_text(encoding='utf-8-sig')
marker='码率受限的无人机航拍小目标检测：';assert marker in text
backup=history/'00_调研报告_C10前.txt'
if not backup.exists():backup.write_text(text,encoding='utf-8')
prefix=('最新研究主线入口（2026-10-28）：11_区域联合恢复_研究主线与最小判别.txt。\n'
        'C10为文献支撑的可证伪研究主线/算法假设；UAV JPEG区域交互、策略效果和新颖性均未由本项目实验确认。\n'
        '以下v3正文和04–10保留为历史阶段记录；暂不跑实验的边界不变。\n\n')
main.write_text(prefix+text[text.index(marker):],encoding='utf-8')

for p in ledger['current_deliverables']:assert Path(p).is_file() and Path(p).stat().st_size>0
checks={'checked_at_utc':now,'artifact_integrity':'PASS','source_documents_and_code_verified':len(manifest),
        'new_references':len(refs),'card_status':card['status'],'goal_status':'active_pending_completion_audit',
        'scientific_outcome_verified':False,'experiments_run':False,
        'verification_scope':'provenance, current source hashes, protocol/card consistency, links and historical preservation; not scientific efficacy or novelty',
        'artifacts':[{'path':p,'bytes':Path(p).stat().st_size,'sha256':sha(Path(p))} for p in ledger['current_deliverables']]}
write(root/'artifact_verification.json',checks)
print(json.dumps({k:checks[k] for k in ['artifact_integrity','source_documents_and_code_verified','new_references','scientific_outcome_verified','experiments_run']},ensure_ascii=False))
