"""Record-only provenance packaging. No dataset/model/evaluator/profiler imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, data):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


papers = [
    ('TA-GRU', HERE / 'papers/TA_GRU_Drones2023_7_466.pdf',
     'https://mdpi-res.com/d_attachment/drones/drones-07-00466/article_deploy/drones-07-00466.pdf',
     'Drones2023,7(7),466, doi10.3390/drones7070466, official14page PDF',
     'Actual focus PDF8-13: dataset/protocol, Table2, Table4, explanations of a-f, limitations and data availability. PDF11Table4 visually inspected. Intro/method excerpts also returned by official indexed source; not full-paper claim.'),
    ('RarePlanes', HERE / 'papers/RarePlanes_arXiv2006.02963.pdf',
     'https://arxiv.org/pdf/2006.02963v2',
     'arXiv2006.02963v2 2020-11-10, WACV2021 paper,11pages',
     'PDF6-8 experiment/split/results/FT direct answer; PDF7Table2 visually inspected. Singleclass FasterRCNN boxAP chosen; maskAP and role task not pooled.'),
    ('Strong synthetic baseline', HERE / 'papers/Strong_Baseline_Synthetic_arXiv2405.19822v1.pdf',
     'https://arxiv.org/pdf/2405.19822v1',
     'arXiv2405.19822v1 2024-05-30,13pages',
     'Intro/problem excerpts and relevantPDF4-8 dataset/role protocol/Table2/discussion. Not full experimental audit. Threeclass role result is not comparable to RarePlanes singleclass AP.'),
    ('Dual interval UAV supplementary evidence', HERE / 'papers/DualInterval_UAV_arXiv2605.22605v1.pdf',
     'https://arxiv.org/pdf/2605.22605v1',
     'arXiv2605.22605v1 2026-05-21,8pages,preprint',
     'Abstract and browser text PDF6TableII, related explanations andPDF7 aroundTableIII. Supplementary same-family example only; not another deep candidate or visually verified table.'),
    ('Constellation comparator independently read', PROJECT / '文献_相关原始论文_20261004/Constellation_IJCV2026_s11263-026-03029-1.pdf',
     'https://doi.org/10.1007/s11263-026-03029-1',
     'IJCV2026 volume134 article429,15page publisher PDF, shared source from peer',
     'OnlyPDF4 acquisition/annotation/drift corpus andPDF12section4.8/Table12 read for independent priority comparison; Table12 visually checked. Official author page also read. No fourth deep candidate.'),
]
ledger = []
for title, path, url, version, scope in papers:
    assert path.is_file(), str(path)
    ledger.append({'type': 'primary_paper', 'title': title, 'path': str(path),
                   'url': url, 'version': version, 'bytes': path.stat().st_size,
                   'sha256': sha(path), 'pdf_pages': len(PdfReader(path).pages),
                   'actual_read_scope': scope})

# Automatic arXiv token search can hit reference lists. Rename the field to
# prevent those tokens from being mistaken for the paper's own version.
receipt_file = HERE / 'paper_fetch_receipts.json'
fetches = load(receipt_file)
for row in fetches:
    if 'version_text' in row:
        row['arxiv_tokens_found_in_full_extraction_not_version_proof'] = row.pop('version_text')
receipt_file.write_text(json.dumps(fetches, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
assert all(row.get('fetch') in ['downloaded', 'existing'] for row in fetches)
for row in fetches:
    assert sha(Path(row['file'])) == row['sha256']

snap = HERE / 'resource_snapshots'
commit = load(snap / 'rareplanes_commit.json')['sha']
assert commit == '2d0f3156a906a21978a66cc05dc418d14a90a1c9'
resource_receipts = load(HERE / 'resource_fetch_receipts.json')
for row in resource_receipts:
    path = Path(row['file'])
    assert row.get('http_status') == 200 and sha(path) == row['sha256']
    ledger.append({'type': 'public_resource_text_or_metadata', 'path': str(path),
                   'url': row['url'], 'sha256': row['sha256'],
                   'scope': row['method'], 'HTTP_status': row['http_status']})
heads = load(HERE / 'weight_HEAD_metadata.json')
assert len(heads) == 3 and all(x.get('http_status') == 200 for x in heads)
ledger.append({'type': 'public_weight_access_metadata', 'path': str(HERE / 'weight_HEAD_metadata.json'),
               'sha256': sha(HERE / 'weight_HEAD_metadata.json'),
               'scope': 'Three author FasterRCNN singleclass aircraft weights; HEAD only, no payload download or model loading.',
               'claim_limit': 'HTTP accessibility and listed size only; no deserialization/config/fidelity verification.'})
ledger.append({'type': 'official_author_project_page',
               'url': 'https://mehmetkeremturkcan.com/projects/constellation/',
               'scope': 'Project findings/release/table notes read via browser; seven generic released models do not establish drift-year checkpoints.'})

public = load(HERE / '05_两对话本轮公开讨论_过滤记录.json')
assert all(x['kind'] in ['public_user_message', 'public_agent_message', 'public_interthread_message_sent']
           for x in public['records'])
save('source_ledger.json', {'date': '2026-10-28 America/Los_Angeles', 'sources': ledger,
                          'resource_status': {'TA_GRU_task_code_weights': 'Not confirmed within paper data availability and bounded author/github search; not proof of permanent absence.',
                                              'RarePlanes_task_weights': 'Author README + S3metadata + HTTP200 HEAD; not downloaded or loaded.',
                                              'Constellation_drift_weights': 'Peer reports public directory7 generic entries, no year-named weights found; not independently verified here.',
                                              'Constellation_drift_dataset': 'Peer verified HF fixed version59bc53b19ab4ee372c650bb7107bc979cfc2fd7a,2510865280bytes; not downloaded or independently checked here.'},
                          'scope_limit': 'No local detection cache or dataset audit. Complete extraction is not complete reading.'})

# Simple arithmetic on quoted published table values, not a new experiment or
# dataset statistic. Scientific effect mechanisms remain unverified.
save('published_table_arithmetic.json', {
    'scope': 'Subtractions from visually checked published tables only; percentage points.',
    'TA_GRU_Table4': {'mAP_naive_minus_single': round(16.55 - 18.71, 2),
                      'AP50_naive_minus_single': round(37.29 - 40.26, 2),
                      'mAP_align_minus_single': round(20.03 - 18.71, 2),
                      'AP50_align_minus_single': round(41.68 - 40.26, 2)},
    'RarePlanes_Table2_aircraft_box_AP': {'synthetic_minus_real': round(54.86 - 73.32, 2),
                                         'finetune_minus_synthetic': round(69.16 - 54.86, 2),
                                         'real_minus_finetune': round(73.32 - 69.16, 2)},
    'Constellation_Table12_vehicle_AP50': {'train2020': {'test2020': 96.4, 'test2023': 86.3},
                                         'train2023': {'test2020': 89.0, 'test2023': 99.2},
                                         'same_test2020_in_year_minus_other_training': round(96.4 - 89.0, 2),
                                         'same_test2023_in_year_minus_other_training': round(99.2 - 86.3, 2)}})
save('verification.json', {'status': 'PASS', 'meaning': 'Artifact/source identity and published-value arithmetic, not experimental/scientific validation.',
                          'verified_UTC': datetime.now(timezone.utc).isoformat(),
                          'public_records': len(public['records']),
                          'sources_hashed_or_linked': len(ledger),
                          'model_loads': 0, 'forwards': 0, 'training': 0, 'reevaluation': 0,
                          'profiling': 0, 'old_cache_audits': 0, 'dataset_or_weight_payload_downloads': 0,
                          'subagents': 0, 'resource_or_label_budget_assumed_as_requirement': False,
                          'current_selection': 'One problem for mechanism discrimination (Constellation); no confirmed paper/method route; video and RarePlanes not recommended as alternatives.'})
artifacts = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(HERE.rglob('*')) if path.is_file() and path.name != 'delivery_manifest.json']
save('delivery_manifest.json', {'status': 'PASS', 'artifacts': artifacts,
                               'scope': 'This record directory; no prior data or project outputs overwritten.'})
print(json.dumps({'status': 'PASS', 'source_records': len(ledger), 'artifacts': len(artifacts),
                  'public_records': len(public['records']), 'experiments': 0,
                  'forwards': 0, 'training': 0, 'reevaluation': 0, 'profiling': 0}, ensure_ascii=False))
