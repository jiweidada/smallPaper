"""Document, JSON, hash, and static syntax checks; no model or dataset experiment."""
from pathlib import Path
import ast, hashlib, json

root = Path(__file__).resolve().parent
base = root.parent
read_json = lambda p: json.loads(p.read_text(encoding='utf-8'))
checks = []

records = read_json(base / 'literature_records.json')
assert len(records) == 20 and len({r['id'] for r in records}) == 20
checks.append('20 unique literature records')
for ref in ['R2','R16']:
    item = next(r for r in records if r['id']==ref)
    assert not item['abstract_only'] and item['full_body_read'] and item['human_read_attestation'] is False
checks.append('New body-reading scope is explicit and not a human attestation')

sources = read_json(base / '03_来源核验与原件存档.json')
assert len(sources['records']) == 20 and len(sources['pdf_inventory']) == 9
for item in sources['pdf_inventory']:
    path = Path(item['file'])
    assert path.exists(), item['ref_id']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['ref_id']
checks.append('Nine archived PDF files exist and match SHA256 records')
for name in ['FSIC','PDAD']:
    item = read_json(root / f'{name}_user_source_receipt.json')
    assert Path(item['text_path']).exists() and Path(item['preflight_path']).exists()
    assert item['structural_verdict'] == 'PASS' and item['external_upload'] is False
    for page in item['visual_pages']:
        assert (root / 'reading_pages' / f'{name}_p{page}.png').exists()
checks.append('Both supplied-paper text/preflight/render receipts exist')

design = read_json(base / '08_最小验证参数.json')
assert design['stage_A']['detector_fits'] == 2
assert design['stage_B_conditional']['restorer_fits'] == 1
assert design['detector']['commit'] == '2004cba021c68a46429be460e26954ab171b2862'
assert design['inference']['nms'] is None and design['inference']['head_end2end'] is False
assert design['inference']['max_det'] == 500
assert not design['execution_authorized_by_current_request']
assert all(value == 0 for value in design['execution_receipt'].values())
assert all(value == 0 for value in sources['execution'].values())
checks.append('Design and zero-execution boundaries agree')

manifest = []
for version_root in sorted((root / 'source_code').iterdir()):
    if not version_root.is_dir():
        continue
    pin = read_json(version_root / 'pin.json')
    for file in sorted(version_root.rglob('*')):
        if file.is_file() and file.name != 'pin.json':
            data = file.read_bytes()
            if file.suffix == '.py':
                ast.parse(data.decode('utf-8-sig'), filename=file.name)
            manifest.append({'tag':pin['tag'], 'commit':pin['commit'], 'path':str(file),
                             'repository_path':file.relative_to(version_root).as_posix(),
                             'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest(),
                             'static_syntax_parsed':file.suffix=='.py', 'executed':False})
assert manifest
(root / 'source_code_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
checks.append(f'{len(manifest)} downloaded static source/config files hashed; Python syntax parsed without importing')

new_report = (base / '06_FSIC_PDAD_YOLO26核查与主线判断.txt').read_text(encoding='utf-8')
protocol = (base / '07_VisDrone_YOLO26s最小验证方案.txt').read_text(encoding='utf-8')
for name in ['FSIC','PDAD']:
    assert read_json(root / f'{name}_user_source_receipt.json')['sha256'] in new_report
assert '预训练优化器未在该段明确' in new_report
assert '不是已测出的普适码率分界' in protocol
assert '不称为未见质量' in protocol
assert '不称VisDrone官方挑战AP' in protocol
main = (base / '00_调研报告.txt').read_text(encoding='utf-8')
for obsolete in ['R16只有摘要', '范围：仅出版商摘要与元数据；PDF下载失败。', '未取得全文确认']:
    assert obsolete not in main, obsolete
assert '定向前期调研报告 v3' in main
assert (base / '历史版本_本次全文核查前' / '00_调研报告.txt').exists()
checks.append('Main report reconciled; historical version preserved; new text has explicit source/metric limits')

all_json = [base/'02_检索与证据记录.json',base/'03_来源核验与原件存档.json',base/'08_最小验证参数.json',
            root/'research_run_ledger.json',root/'source_code_manifest.json']
for path in all_json:
    read_json(path)
receipt = {'date':'2026-10-28','scope':'artifact integrity and static text only',
           'checks':checks,'model_imports':0,'model_execution':0,'dataset_reencoding':0,
           'test_dev_access':0,'subagents':0,'result':'PASS'}
(root / 'artifact_verification_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=True))
