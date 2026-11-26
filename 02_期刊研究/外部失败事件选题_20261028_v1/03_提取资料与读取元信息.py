"""Text extraction and metadata inspection only; no evaluation or model operations."""
from pathlib import Path
import json
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parent
library = root.parent / '文献_相关原始论文_20261004'
textdir = library / '_text'
textdir.mkdir(exist_ok=True)
for name in ['Constellation_IJCV2026_s11263-026-03029-1.pdf', 'FDD_CVPR2023_原论文.pdf']:
    pdf = library / name
    pages = PdfReader(pdf).pages
    output = textdir / (pdf.stem + '.txt')
    extracted = '\n\n'.join(f'--- PAGE {i+1} ---\n' + (page.extract_text() or '') for i, page in enumerate(pages))
    output.write_text(extracted.encode('utf-8', errors='replace').decode('utf-8'), encoding='utf-8')
    print(json.dumps({'source': name, 'pages': len(pages), 'text_path': str(output)}, ensure_ascii=False))
commit = json.loads((root / 'Constellation_GitHub_main元信息.json').read_text(encoding='utf-8'))
tree = json.loads((root / 'Constellation_GitHub_文件树.json').read_text(encoding='utf-8'))
dataset = json.loads((root / 'Constellation_HF数据集元信息.json').read_text(encoding='utf-8'))
print(json.dumps({'github_commit': commit.get('sha'), 'commit_date': commit.get('commit', {}).get('committer', {}).get('date'), 'tree_sha': tree.get('sha'), 'files': [{'path': x['path'], 'size': x.get('size')} for x in tree.get('tree', []) if x.get('type') == 'blob']}, ensure_ascii=False))
print(json.dumps({'dataset_id': dataset.get('id'), 'sha': dataset.get('sha'), 'lastModified': dataset.get('lastModified'), 'usedStorage': dataset.get('usedStorage'), 'siblings': dataset.get('siblings'), 'cardData': dataset.get('cardData')}, ensure_ascii=False))
