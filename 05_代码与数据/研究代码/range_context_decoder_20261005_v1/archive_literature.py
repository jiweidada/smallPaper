"""Validate the newly obtained original PDF and append its library hash."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2] / '文献_相关原始论文_20261004'
name = '26_QEDetr_RemoteSensing2025.pdf'
path = ROOT / name
digest = hashlib.sha256(path.read_bytes()).hexdigest()
reader = PdfReader(path)
text = '\n\n'.join(f'\n--- PDF PAGE {i + 1} ---\n' + (page.extract_text() or '')
                  for i, page in enumerate(reader.pages))
target = ROOT / '_text' / (path.stem + '.txt')
if target.exists():
    raise RuntimeError('Preserving existing extracted text')
target.write_text(text, encoding='utf8')
manifest_path = ROOT / '_hashes.json'
manifest = json.loads(manifest_path.read_text(encoding='utf8'))
backup = ROOT / '_hashes_before_range_candidate_20261005.json'
if backup.exists():
    raise RuntimeError('Preserving existing manifest backup')
backup.write_bytes(manifest_path.read_bytes())
for row in manifest['files']:
    assert hashlib.sha256((ROOT / row['file']).read_bytes()).hexdigest() == row['sha256']
assert name not in {x['file'] for x in manifest['files']}
manifest['files'].append({'file': name, 'bytes': path.stat().st_size, 'sha256': digest})
manifest['count'] = len(manifest['files'])
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
receipt = {'file': name, 'sha256': digest, 'pages': len(reader.pages), 'text_characters': len(text),
           'source': 'https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-00893/article_deploy/remotesensing-17-00893.pdf',
           'status': 'original_PDF_validated_text_extracted', 'all_library_hashes_verified': manifest['count']}
(ROOT / '_sources_range_candidate_20261005_validated.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps(receipt))
