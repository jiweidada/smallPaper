"""Final bibliography/protocol metadata reconciliation; no model execution."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
base = root.parent
doi = '10.1007/978-3-031-20077-9_28'
for name in ['literature_records.json', '03_来源核验与原件存档.json']:
    path = base / name
    value = json.loads(path.read_text(encoding='utf-8'))
    records = value if isinstance(value, list) else value['records']
    item = next(row for row in records if row['id'] == 'R20')
    item['doi'] = doi
    item['publisher_url'] = 'https://link.springer.com/chapter/' + doi
    item['metadata_verification'] = 'Author list and title verified at ECVA; DOI and pp473–491 verified at publisher'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

path = base / '01_核心文献矩阵.txt'
text = path.read_text(encoding='utf-8')
start = text.index('R20 | AERIS')
text = text[:start] + text[start:].replace('仅摘要：否；以实际阅读范围为准',
    'DOI：'+doi+'（出版商与ECVA链接交叉核对）\n仅摘要：否；以实际阅读范围为准')
path.write_text(text, encoding='utf-8')

path = base / '08_最小验证参数.json'
value = json.loads(path.read_text(encoding='utf-8'))
value['evaluation']['strict_small_area_upper_bound_implementation'] = 'math.nextafter(1024.0, -math.inf), or equivalent strict area filtering; COCO default upper area boundary is inclusive'
value['evaluation']['rule_source_read_not_executed'] = 'https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py'
path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

path = base / '02_检索与证据记录.json'
value = json.loads(path.read_text(encoding='utf-8'))
value['batches'][-1]['additional_primary_snapshots'] = [str(p) for p in sorted((root/'sources').glob('S_*.json'))]
value['batches'][-1]['primary_metadata_followup'] = ['https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2829_ECCV_2022_paper.php', 'https://link.springer.com/chapter/'+doi, 'https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py']
path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

for name in ['00_调研报告.txt','06_FSIC_PDAD_YOLO26核查与主线判断.txt']:
    path = base / name
    text = path.read_text(encoding='utf-8')
    text += '\nR20出版元数据补核：ECVA链接指向Springer，页码473–491及DOI '+doi+' 已由出版商页面确认；全文阅读范围不变。\n'
    path.write_text(text, encoding='utf-8')

path = root / 'research_run_ledger.json'
value = json.loads(path.read_text(encoding='utf-8'))
value['artifact_verification_receipt'] = str(root / 'artifact_verification_receipt.json')
value['bibliography_count'] = 20
value['reporting_stage'] = 'research report and future protocol completed; experiments not started'
path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('Final bibliography and metric boundary metadata reconciled; no model execution.')
