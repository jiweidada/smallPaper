from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, time
import requests

ROOT = Path(__file__).resolve().parent
LIB = ROOT.parents[1] / '文献_相关原始论文_20261004'
TEXT = LIB / '_text'
SOURCES = [
    ('A48_WBN_Yasuda_2020', 'Consistent Batch Normalization for Weighted Loss in Imbalanced-Data Environment',
     'https://www.jstage.jst.go.jp/article/nolta/11/4/11_454/_pdf', '10.1587/nolta.11.454', 'journal 2020'),
    ('A48_CEASC_CVPR2023', 'Adaptive Sparse Convolutional Networks with Global Context Enhancement for Faster Object Detection on Drone Images',
     'https://arxiv.org/pdf/2303.14488', '10.48550/arXiv.2303.14488', 'arXiv v1; CVPR 2023 accepted'),
    ('A48_Region_Normalization_AAAI2020', 'Region Normalization for Image Inpainting',
     'https://ojs.aaai.org/index.php/AAAI/article/download/6967/6821', '10.1609/aaai.v34i07.6967', 'AAAI 2020 proceedings'),
    ('A48_CateNorm_2022', 'CateNorm: Categorical Normalization for Robust Medical Image Segmentation',
     'https://www.cs.jhu.edu/~alanlab/Pubs22/xiao2022catenorm.pdf', '10.48550/arXiv.2103.15858', 'author-hosted published PDF; verify venue in PDF'),
    ('A48_Marker_Normalization_VISAPP2024', 'Understanding Marker-Based Normalization for FLIM Networks',
     'https://www.scitepress.org/Papers/2024/123859/123859.pdf', '10.5220/0012385900003660', 'VISAPP 2024 proceedings'),
]

def acquire(row):
    stem, title, url, doi, version = row
    pdf = LIB / (stem + '.pdf')
    record = dict(title=title, original_url=url, doi=doi, version=version,
                  path=str(pdf), accessed_unix=time.time(), reading_scope='acquired; method reading tracked separately')
    try:
        if not pdf.exists():
            response = requests.get(url, timeout=(12, 35))
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError('Response is not a PDF')
            with pdf.open('xb') as stream:
                stream.write(response.content)
            record['resolved_url'] = response.url
        record.update(status='PDF_ACQUIRED', bytes=pdf.stat().st_size,
                      sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
        txt = TEXT / (stem + '.txt')
        if not txt.exists():
            subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe', '-layout', str(pdf), str(txt)], check=True)
        record['text_path'] = str(txt)
        record['text_sha256'] = hashlib.sha256(txt.read_bytes()).hexdigest()
    except Exception as exc:
        record.update(status='ACQUISITION_FAILED', reason=str(exc))
    meta = LIB / (stem + '.source.json')
    if not meta.exists():
        with meta.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
    return record

if __name__ == '__main__':
    TEXT.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(acquire, SOURCES))
    (ROOT / 'source_acquisition_v1.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps([{k:r.get(k) for k in ['title','status','bytes','sha256','reason']} for r in records], ensure_ascii=False))
