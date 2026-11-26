"""Finite primary-source acquisition; no training or model calls."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess
import time
import requests

ROOT = Path(__file__).resolve().parent
LIB = ROOT.parents[1] / '文献_相关原始论文_20261004'
PAPERS = [
    ('A49_SparseDet_ICCV2023_Publisher', 'SparseDet: Improving Sparsely Annotated Object Detection with Pseudo-positive Mining',
     'https://openaccess.thecvf.com/content/ICCV2023/papers/Suri_SparseDet_Improving_Sparsely_Annotated_Object_Detection_with_Pseudo-positive_Mining_ICCV_2023_paper.pdf',
     'ICCV2023 publisher PDF; author-hosted PDF selected methods3.1-3.5 read via web, publisher pages require cross-check.'),
    ('A49_BRL_arxiv2002_05274v1', 'Solving Missing-Annotation Object Detection with Background Recalibration Loss',
     'https://arxiv.org/pdf/2002.05274v1', 'arXivv1,2020-02-12; abstract only so far, not full method.'),
    ('A49_Gated_DeltaNet2_arxiv2605_22791v1', 'Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention',
     'https://arxiv.org/pdf/2605.22791v1', 'arXivv1 technical report,2026-05-21; abstract only, not a verified vision host or detection method.'),
    ('A49_Deep_Delta_Learning_arxiv2601_00417v3', 'Deep Delta Learning',
     'https://arxiv.org/pdf/2601.00417v3', 'arXivv3,2026-05-12; abstract only. v4 dated2026-07-27 exists and is not method-read here; do not call v3 latest.')
]
REPOS = [('kjh0902/pud_detr', ['README.md', 'train_pud_detr.py']),
         ('AISLAB-PUD-DETR/PUD-DETR', ['README.md', 'train_pu.py'])]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire_paper(row):
    stem, title, url, scope = row
    record = dict(kind='paper', title=title, url=url, version_and_reading_scope=scope,
                  accessed_local_unix=time.time())
    pdf = LIB / (stem + '.pdf')
    try:
        assert not pdf.exists(), 'Existing file must be inspected rather than overwritten.'
        response = requests.get(url, timeout=(8, 22))
        response.raise_for_status()
        assert response.content.startswith(b'%PDF'), 'Not a PDF response'
        with pdf.open('xb') as f:
            f.write(response.content)
        txt = LIB / '_text' / (stem + '.txt')
        assert not txt.exists()
        subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe', '-layout', str(pdf), str(txt)], check=True)
        record.update(status='PDF_ACQUIRED_NOT_FULL_READ', path=str(pdf), bytes=pdf.stat().st_size,
                      sha256=digest(pdf), text_path=str(txt), text_sha256=digest(txt), resolved_url=response.url)
    except Exception as exc:
        record.update(status='ACQUISITION_FAILED', reason=str(exc))
    with (LIB / (stem + '.source.json')).open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def acquire_repo(item):
    repo, names = item
    record = dict(kind='public_source_code', repo=repo, url='https://github.com/' + repo,
                  paper_publication_status='Not verified; current kjh0902 README explicitly says paper link/publication details to be added.',
                  reading_scope='README seen via web; source downloaded below is not yet fully read.', accessed_local_unix=time.time())
    try:
        response = requests.get('https://api.github.com/repos/' + repo + '/commits/main', timeout=(8, 20))
        response.raise_for_status()
        commit = response.json()['sha']
        dest = ROOT / 'sources' / (repo.replace('/', '__') + '_' + commit[:12])
        dest.mkdir(parents=True, exist_ok=False)
        files = []
        for name in names:
            url = 'https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + name
            response = requests.get(url, timeout=(8, 20))
            response.raise_for_status()
            path = dest / name
            with path.open('xb') as f:
                f.write(response.content)
            files.append(dict(path=str(path), url=url, sha256=digest(path), bytes=path.stat().st_size))
        record.update(status='PINNED_SOURCE_ACQUIRED', commit=commit, files=files)
    except Exception as exc:
        record.update(status='ACQUISITION_FAILED', reason=str(exc))
    return record


if __name__ == '__main__':
    target = ROOT / 'source_acquisition_v1.json'
    assert not target.exists(), 'Inspect a partial attempt; never blindly rerun.'
    jobs = [('paper', row) for row in PAPERS] + [('repo', row) for row in REPOS]
    def run(job):
        return acquire_paper(job[1]) if job[0] == 'paper' else acquire_repo(job[1])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(run, jobs))
    with target.open('x', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(json.dumps([{k: r.get(k) for k in ['kind', 'title', 'repo', 'status', 'sha256', 'commit', 'reason']} for r in records], ensure_ascii=False))
