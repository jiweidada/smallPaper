"""Archive only the sources used in a bounded backup-method screen; no GPU work."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess
import time
import requests

ROOT = Path(__file__).resolve().parent
LIB = ROOT.parents[1] / '文献_相关原始论文_20261004'
SOURCES = [
    {'stem': 'A48_OGMN_arxiv2304_11805v1', 'kind': 'pdf',
     'title': 'OGMN: Occlusion-guided Multi-task Network for Object Detection in UAV Images',
     'url': 'https://arxiv.org/pdf/2304.11805v1', 'version': 'arXiv v1, 2023-04-24; final journal identity not checked',
     'reading_scope': 'Original arXiv HTML abstract and selected methods 3.1-3.4, equations1-9 and start of Algorithm1; not full-paper or experiments reading.'},
    {'stem': 'A48_VE_DINO_SmartCities_8_1_12', 'kind': 'pdf',
     'title': 'Enhancing Human Detection in Occlusion-Heavy Disaster Scenarios: A Visibility-Enhanced DINO (VE-DINO) Model with Reassembled Occlusion Dataset',
     'url': 'https://mdpi-res.com/d_attachment/smartcities/smartcities-08-00012/article_deploy/smartcities-08-00012.pdf',
     'version': 'Publisher PDF URL; acquisition and firstpage identity tracked separately',
     'reading_scope': 'Publisher search-result abstract/contribution snippet only; HTML429 and web PDF access failed. No method-equivalence claim; downloading does not upgrade reading scope.'},
    {'stem': 'A48_grid_sample_PyTorch_2_7', 'kind': 'html',
     'title': 'torch.nn.functional.grid_sample, PyTorch2.7 official documentation',
     'url': 'https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.grid_sample.html',
     'version': 'PyTorch2.7 pinned documentation, matching remote torch major/minor',
     'reading_scope': 'Signature, padding_mode options and align_corners sections. Software documentation, not a paper.'}
]


def acquire(item):
    row = dict(item, accessed_local_unix=time.time())
    dest = LIB / (item['stem'] + '.' + item['kind'])
    row['path'] = str(dest)
    try:
        assert not dest.exists(), 'Preserve existing source; inspect before any retry.'
        response = requests.get(item['url'], timeout=(8, 22))
        response.raise_for_status()
        if item['kind'] == 'pdf':
            assert response.content.startswith(b'%PDF'), 'Not a PDF response'
        else:
            assert b'<html' in response.content.lower(), 'Not an HTML response'
        with dest.open('xb') as f:
            f.write(response.content)
        row.update(status='ACQUIRED', bytes=dest.stat().st_size,
                   sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), resolved_url=response.url)
        if item['kind'] == 'pdf':
            txt = LIB / '_text' / (item['stem'] + '.txt')
            assert not txt.exists()
            subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe', '-layout', str(dest), str(txt)], check=True)
            row.update(text_path=str(txt), text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest())
    except Exception as exc:
        row.update(status='ACQUISITION_FAILED', reason=str(exc))
    meta = LIB / (item['stem'] + '.source.json')
    with meta.open('x', encoding='utf-8') as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    return row


if __name__ == '__main__':
    target = ROOT / 'backup_source_acquisition_v1.json'
    assert not target.exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(acquire, SOURCES))
    with target.open('x', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(json.dumps([{k: r.get(k) for k in ['title', 'status', 'bytes', 'sha256', 'reason']} for r in records], ensure_ascii=False))
