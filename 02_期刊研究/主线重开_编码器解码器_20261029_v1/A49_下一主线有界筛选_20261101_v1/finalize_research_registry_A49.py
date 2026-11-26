import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[1] / '文献_相关原始论文_20261004'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_once(p, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if p.exists():
        if p.read_bytes() != data:
            raise RuntimeError('Existing immutable record differs: ' + str(p))
    else:
        p.write_bytes(data)

acquired = json.loads((HERE / 'source_acquisition_v1.json').read_text(encoding='utf-8-sig'))
matrix = json.loads((HERE / '02_核心方法矩阵_v1.json').read_text(encoding='utf-8-sig'))
records = []
for item in matrix['records']:
    r = dict(item)
    matches = [a for a in acquired if a.get('title') == item.get('title')]
    if item['id'] == 'S01':
        matches = [a for a in acquired if a.get('repo') == 'kjh0902/pud_detr']
    if len(matches) != 1:
        raise RuntimeError('Acquisition match ambiguous: ' + item['id'])
    a = matches[0]
    if a['kind'] == 'paper':
        for path_key, sha_key in [('path', 'sha256'), ('text_path', 'text_sha256')]:
            p = Path(a[path_key])
            if sha(p) != a[sha_key]:
                raise RuntimeError('Changed source: ' + str(p))
        r.update({k: a[k] for k in ['path', 'sha256', 'text_path', 'text_sha256', 'resolved_url']})
        r['status'] = 'ABSTRACT_ONLY' if item['id'] in ['D01', 'D02'] else 'SELECTED_METHODS_READ_NOT_FULL_PAPER'
    else:
        for f in a['files']:
            if sha(Path(f['path'])) != f['sha256']:
                raise RuntimeError('Changed source: ' + f['path'])
        r['files'] = a['files']
        r['status'] = 'PINNED_SELECTED_CODE_READ_PUBLICATION_UNVERIFIED'
    records.append(r)

views = {}
for p in sorted((HERE / 'source_views').glob('*.png')):
    views[str(p)] = sha(p)
registry = {
    'scope': 'A49 targeted pre-study, not a systematic review. Acquisition sidecars retain their initial scope; this record gives the later verified reading scope.',
    'records': records,
    'visual_formula_figure_checks': views,
    'excluded_from_method_claims': [
        {'repo': 'AISLAB-PUD-DETR/PUD-DETR', 'commit': '43aaa1bf534d3718319e0cc2bbd16e868e3bf616', 'scope': 'README seen; acquired train_pu.py not method-read. Do not merge this version and authors/results with kjh0902.'},
        {'url': 'https://eshed1.github.io/papers/refineNet.pdf', 'scope': 'Discovery snippet only; not identified/read as a core method or used for a method conclusion.'}
    ],
    'integrity_check': 'PASS for four acquired PDF/text pairs and two pinned kjh0902 source files. This verifies file identity, not experimental reproducibility or novelty.'
}
write_once(HERE / '05_增补检索与阅读范围_v1.json', registry)
write_once(LIB / 'A49_阅读范围与来源核验_20261101_v1.json', registry)

log_files = [HERE / '01_检索原始记录_v1.json'] + sorted((HERE / 'search_read_logs').glob('*.json'))
log_index = {
    'review_type': 'Targeted research screening; no claim of exhaustive coverage or PRISMA counts.',
    'initial_search_log': str(log_files[0]),
    'supplement': 'Original retrieval/search outputs preserved verbatim as tool results. Do not infer full reading from retrieval. Exact initial query metadata is in the initial log; later reading entry points are the source URLs in the matrix and this registry.',
    'query_concepts': ['UAV small object detection, known ignored regions, unmatched query background supervision', 'sparse annotation, pseudo-positive mining, background recalibration, nnPU/DETR', 'query attention/sampling support with ignore supervision', 'delta-rule erase/write mechanisms as a distinct held candidate'],
    'inclusion': 'Primary paper or pinned primary implementation relevant to the selected operation, plus official dataset documentation.',
    'exclusion': 'Secondary summaries cannot support method claims; abstracts are inspiration only; unverified paper/venue remains software evidence.',
    'files_sha256': {str(p): sha(p) for p in log_files},
    'reading_registry': str(HERE / '05_增补检索与阅读范围_v1.json')
}
write_once(HERE / '06_检索记录索引_v1.json', log_index)
print(json.dumps({'status': 'PASS', 'paper_records': 4, 'selected_code_records': 1, 'search_log_files': len(log_files), 'visual_pages': len(views), 'library_registry': str(LIB / 'A49_阅读范围与来源核验_20261101_v1.json')}, ensure_ascii=False))
