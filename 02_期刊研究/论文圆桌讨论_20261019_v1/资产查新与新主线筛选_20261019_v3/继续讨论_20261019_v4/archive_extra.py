import hashlib, json, pathlib, urllib.request
from pypdf import PdfReader

root = pathlib.Path(__file__).resolve().parent
library = pathlib.Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\文献_相关原始论文_20261004')
sources = [
    ('P26', 'H-DETR', 'https://www.microsoft.com/en-us/research/uploads/prod/2023/06/DETRs-with-Hybrid-Matching.pdf'),
    ('P27', 'Unbalanced_Optimal_Transport', 'https://openaccess.thecvf.com/content/CVPR2023/papers/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.pdf'),
]
records = []
for sid, name, url in sources:
    dest = library / f'圆桌续议_20261019_{sid}_{name}.pdf'
    rec = dict(id=sid, name=name, url=url)
    try:
        if not dest.exists():
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            raw = urllib.request.urlopen(req, timeout=30).read()
            if not raw.startswith(b'%PDF'): raise ValueError('response is not a PDF')
            dest.write_bytes(raw)
        raw = dest.read_bytes()
        pdf = PdfReader(dest)
        textpath = library / '_text' / f'圆桌续议_20261019_{sid}.txt'
        if not textpath.exists():
            textpath.write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{p.extract_text()}' for i,p in enumerate(pdf.pages)), encoding='utf-8')
        rec.update(status='downloaded_extracted', path=str(dest), text_path=str(textpath), pages=len(pdf.pages), sha256=hashlib.sha256(raw).hexdigest())
    except Exception as exc:
        rec.update(status='not_downloaded', error=str(exc), no_tls_bypass=True)
    records.append(rec)
(root/'补充匹配近邻_下载记录.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
