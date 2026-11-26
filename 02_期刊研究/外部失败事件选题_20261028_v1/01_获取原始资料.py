"""Only download and archive primary literature and public source metadata. No model execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
library = root.parent / '文献_相关原始论文_20261004'
sources = [
    ('Constellation_IJCV2026_s11263-026-03029-1.pdf', 'https://link.springer.com/content/pdf/10.1007/s11263-026-03029-1.pdf', library),
    ('LL_YOLO_EnlightenGAN_MeasurementControl2025_原文.html', 'https://sage.cnpereading.com/doi/10.1177/00202940251356201', library),
    ('Constellation_作者项目页_20261028.html', 'https://mehmetkeremturkcan.com/constellation-web/', root),
    ('Constellation_GitHub_main元信息.json', 'https://api.github.com/repos/zk2172-columbia/constellation-dataset/commits/main', root),
    ('Constellation_GitHub_文件树.json', 'https://api.github.com/repos/zk2172-columbia/constellation-dataset/git/trees/main?recursive=1', root),
    ('Constellation_HF数据集元信息.json', 'https://huggingface.co/api/datasets/mehmetkeremturkcan/constellation_urban_intersection_dataset', root),
    ('FDD_CVPR2023_原论文.pdf', 'https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Generalized_UAV_Object_Detection_via_Frequency_Domain_Disentanglement_CVPR_2023_paper.pdf', library),
    ('Constellation_固定版本README.txt', 'https://raw.githubusercontent.com/zk2172-columbia/constellation-dataset/6447e579ec2d16dd7485ed24c1e4ab7f3c402487/README.md', root),
    ('Constellation_作者train_script_仅读不执行.txt', 'https://raw.githubusercontent.com/zk2172-columbia/constellation-dataset/6447e579ec2d16dd7485ed24c1e4ab7f3c402487/training/ultralytics/train_script.py', root),
    ('Constellation_作者evaluation_仅读不执行.txt', 'https://raw.githubusercontent.com/zk2172-columbia/constellation-dataset/6447e579ec2d16dd7485ed24c1e4ab7f3c402487/evaluation/ultralytics/evaluation.py', root),
    ('Constellation_HF固定版本数据卡.txt', 'https://huggingface.co/datasets/mehmetkeremturkcan/constellation_urban_intersection_dataset/raw/59bc53b19ab4ee372c650bb7107bc979cfc2fd7a/README.md', root),
    ('Constellation_HF文件大小元信息.json', 'https://huggingface.co/api/datasets/mehmetkeremturkcan/constellation_urban_intersection_dataset/tree/59bc53b19ab4ee372c650bb7107bc979cfc2fd7a?recursive=true&expand=false', root),
]

def fetch(item):
    name, url, folder = item
    target = folder / name
    record = {'name': name, 'url': url, 'path': str(target), 'accessed_utc': datetime.now(timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 literature source verification'})
        with urllib.request.urlopen(req, timeout=45) as response:
            payload = response.read()
            record['final_url'] = response.url
            record['content_type'] = response.headers.get('Content-Type')
        if name.endswith('.pdf') and not payload.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF; no PDF file saved')
        if target.exists():
            existing = target.read_bytes()
            if existing != payload:
                target = target.with_name(target.stem + '_20261028' + target.suffix)
                record['path'] = str(target)
        target.write_bytes(payload)
        record.update(ok=True, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    except Exception as exc:
        record.update(ok=False, error=str(exc))
    return record

with ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(fetch, sources))
(root / '02_原始资料获取回执.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
for record in records:
    print(json.dumps({k: record[k] for k in ['name', 'ok', 'bytes', 'error'] if k in record}, ensure_ascii=False))
