"""Fetch public primary research only; never execute fetched source."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys, urllib.request

sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parent
B = D.parent
L = B.parent / '文献_相关原始论文_20261004'
PAPERS = [
    ('A35_QR_DETR_ACCV2024', 'https://openaccess.thecvf.com/content/ACCV2024/papers/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.pdf'),
    ('A35_SQ_DETR_AppliedSciences2025', 'https://mdpi-res.com/d_attachment/applsci/applsci-15-07686/article_deploy/applsci-15-07686.pdf'),
    ('A35_BUTR_arXiv2610_33430v1', 'https://arxiv.org/pdf/2610.33430v1'),
    ('A35_DifferentialTransformer_arXiv2410_05258v2', 'https://arxiv.org/pdf/2410.05258v2'),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)


def jsave(p, x):
    save(p, (json.dumps(x, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


prior_path = B / '79_A34剩余76图独立复核回执.json'
prior = json.loads(prior_path.read_text(encoding='utf-8'))
for name, h in prior['files_sha256'].items():
    assert sha(name) == h, name
assert len(prior['files_sha256']) == 603
jsave(D / 'handoff_A34_verified.json', dict(
    previous_goal_turn='progress: A34 completed frozen76-image replication, independently verified statistics and rejected the directional hypothesis.',
    prior_receipt=str(prior_path), prior_receipt_sha256=sha(prior_path), verified_files=603,
    model_forwards_before=3168, optimizer_updates_before=528,
    phase_neighbor_branch_closed=True, new_model_call_allocation=0,
    scope='Architecture/source investigation; not a new training or inference allocation.',
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
rows = []
for stem, url in PAPERS:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'UAV-architecture-primary-source-review'})
        with urllib.request.urlopen(req, timeout=30) as f:
            data = f.read()
        assert data.startswith(b'%PDF-'), 'Response is not PDF'
        pdf = L / (stem + '.pdf')
        save(pdf, data)
        txt = L / '_text' / (stem + '.txt')
        assert not txt.exists()
        subprocess.run(['pdftotext', '-layout', str(pdf), str(txt)], check=True)
        row = dict(file=str(pdf), url=url, bytes=len(data), sha256=sha(pdf),
                   text_file=str(txt), text_sha256=sha(txt),
                   retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   read_scope='Pending selected method inspection; no full-reading claim.')
        jsave(L / (stem + '.source.json'), row)
    except Exception as e:
        row = dict(url=url, error=repr(e))
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k in ['file', 'bytes', 'sha256', 'error']}, ensure_ascii=False), flush=True)
jsave(D / 'primary_download_receipt.json', dict(rows=rows))
