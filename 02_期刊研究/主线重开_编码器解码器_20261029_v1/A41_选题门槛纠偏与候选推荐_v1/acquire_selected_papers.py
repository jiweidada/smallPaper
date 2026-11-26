import concurrent.futures, datetime, hashlib, importlib.util, json, pathlib, shutil, subprocess, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
LIB = ROOT.parent.parent / '文献_相关原始论文_20261004'
LIB.mkdir(exist_ok=True)
(LIB / '_text').mkdir(exist_ok=True)
PAPERS = [
    ('SD_DETR_arxiv2310_15955v1', 'https://arxiv.org/pdf/2310.15955v1'),
    ('DESTR_CVPR2022', 'https://openaccess.thecvf.com/content/CVPR2022/papers/He_DESTR_Object_Detection_With_Split_Transformer_CVPR_2022_paper.pdf'),
    ('DETRDistill_ICCV2023', 'https://openaccess.thecvf.com/content/ICCV2023/papers/Chang_DETRDistill_A_Universal_Knowledge_Distillation_Framework_for_DETR-families_ICCV_2023_paper.pdf'),
]

def acquire(item):
    key, url = item
    path = LIB / ('A41_' + key + '.pdf')
    row = {'id': key, 'url': url, 'local': str(path), 'pdf_read_scope': 'not_yet_read', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        if path.exists():
            data = path.read_bytes()
            row['download_status'] = 'existing_file_reused'
        else:
            request = urllib.request.Request(url, headers={'User-Agent':'Academic research source archiving'})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read(15_000_001)
                row['final_url'] = response.url
            if not data.startswith(b'%PDF') or len(data) > 15_000_000:
                raise ValueError('Response is not an accepted-size PDF')
            path.write_bytes(data)
            row['download_status'] = 'downloaded'
        row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        textpath = LIB / '_text' / (path.stem + '.txt')
        exe = shutil.which('pdftotext')
        if exe:
            subprocess.run([exe, '-layout', str(path), str(textpath)], check=True, capture_output=True, timeout=20)
            row['text_extraction'] = 'pdftotext_layout'
        elif importlib.util.find_spec('fitz'):
            import fitz
            doc = fitz.open(path)
            textpath.write_text('\n'.join(page.get_text() for page in doc), encoding='utf-8')
            row['text_extraction'] = 'pymupdf'
        else:
            row['text_extraction'] = 'unavailable'
        if textpath.exists():
            row['text_local'] = str(textpath)
            row['text_sha256'] = hashlib.sha256(textpath.read_bytes()).hexdigest()
    except Exception as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
        row.setdefault('download_status', 'unavailable')
    return row

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(acquire, PAPERS))
    (ROOT / 'paper_archive_receipts.json').write_text(json.dumps({'rows':rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in rows:
        print(row['id'], row['download_status'], row.get('bytes'), row.get('text_extraction'), row.get('error',''))
