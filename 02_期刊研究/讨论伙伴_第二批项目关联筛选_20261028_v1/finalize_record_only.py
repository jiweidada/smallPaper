"""Package a bounded literature screen; no experiments or dataset statistics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


receipt = load(HERE / 'primary_fetch_receipt.json')
paper = Path(receipt['path'])
assert receipt['pages'] == 45
assert receipt['sha256'] == sha(paper) == '5a0710535c4f3f622c4f6b1c5ddb919728e72f3c3edc57cbf6b5602ee1b5205c'
public_file = HERE / '03_第二批公开消息_过滤记录.json'
public = load(public_file)
assert all(x['kind'] in ['public_user_message', 'public_agent_message', 'public_interthread_message_sent']
           for x in public['records'])
sources = [
    {'type': 'primary_preprint', 'title': 'ContextShift: A Controlled Benchmark for Context Dependence in Object Detection',
     'url': 'https://arxiv.org/pdf/2606.09495v1', 'path': str(paper),
     'version': 'arXiv2606.09495v1,2026-06-08,45pages,preprint', 'sha256': sha(paper),
     'actual_read_scope': 'Main3.3.2-5.2.3 and6, A.6, B, C.2-C.7, D.3.1, E.1-E.3, F resource paragraphs; selected related material at physicalPDF4-6,8,13-20,25,35-36. Table15 visual checked on physicalPDF25. Not full45page reading.',
     'interpretation_limit': 'Reported model/background intervention observations only; no network-mechanism, independent-sample-CI or current-VisDrone causal conclusion.'},
    {'type': 'author_repository_public_README',
     'url': 'https://github.com/DanZlotnikov/ContextShiftBenchmark',
     'actual_read_scope': 'README natural multiobject instructions, lines326-332 in browser text, structure and result descriptions. Allpolygon objects retained is a documentation claim.',
     'commit_verified': None, 'implementation_verified': False,
     'unverified': 'Exact nonfocalGT/detection ignore/matching implementation, benchmark checkpoint/config identities. No repository code was executed.'},
    {'type': 'author_dataset_card',
     'url': 'https://huggingface.co/datasets/contextshift/manipulation',
     'actual_read_scope': 'Dataset description, prebuilt versus in-pipeline families, mask/bbox/focal manifest description and out-of-scope causal claims. No dataset download.',
     'version_verified': None},
    {'type': 'peer_public_current_project_contract', 'path': str(public_file), 'sha256': sha(public_file),
     'actual_read_scope': 'Peer reports checked currentargs: DEIM/HGNetv2-B0/HybridEncoder/DFINETransformer/eval640/10classes/132epoch/two-seedEMA; independently read literature risk is not an empirical test of these weights.'}
]
save('source_ledger.json', {'date': '2026-10-28 America/Los_Angeles', 'sources': sources,
                          'bounded_search': 'One selected second-batch event only. COCO-O and other search hits not pursued as additional candidates.',
                          'access_limits': 'Anonymous link unavailable to browser; normal verified HTTPS metadata/raw-code requests hitSSL EOF, browser code page cachemiss. No authentication/certificate bypass; stopped further code/asset audit.',
                          'source_correction': 'An initial generated image namedTable15_multiobject actually rendered physicalPDF24Fig9; renamedContextShift_pdf24_Fig9.png, then rendered and visually inspected physicalPDF25Table15.'})
save('quoted_observation_metadata.json', {
    'scope': 'Quoted paper values, not newly computed dataset metrics.',
    'Table15': {'protocol': 'COCO2017 multiobject focal, cleanoriginal background comparison,30compatibility bins,threshold.25,IoU.5,modeldefaultpostprocessing',
                'source_images': 4952, 'image_background_combinations': 148560,
                'combination_count_is_not_independent_sample_count': True,
                'D_FINE_L': {'AP50_rAUC_ratio': .79, 'relative_FN_per_image_percent': 57.0,
                             'relative_FP_per_image_percent': -19.5, 'relative_predictions_per_image_percent': -18.8},
                'Deformable_DETR': {'AP50_rAUC_ratio': .75, 'relative_FN_per_image_percent': 27.1,
                                    'relative_FP_per_image_percent': -18.0, 'relative_predictions_per_image_percent': -19.2}},
    'Table20_ordinary_answer': {'model': 'YOLO26M,not currentDEIM', 'heldout_images': 990,
                                'baseline_reported_AP50': .877, 'background_aug_reported_AP50': .899,
                                'baseline_FN_per_image': .123, 'background_aug_FN_per_image': .101},
    'Table21_bg_swap': {'baseline_rAUC': .346, 'background_aug_rAUC': .385},
    'local_VisDrone_natural_background_effect': 'not established, not measured in this round',
    'candidate_existence_is_only_IoU_score_proxy': True})
save('verification.json', {'status': 'PASS', 'scope': 'Document/source-identity checks only; no experimental or causal validation.',
                          'verified_UTC': datetime.now(timezone.utc).isoformat(),
                          'public_records': len(public['records']),
                          'experiments': 0, 'forwards': 0, 'training': 0, 'reevaluation': 0,
                          'profiling': 0, 'old_cache_audits': 0, 'new_dataset_statistics': 0,
                          'model_or_dataset_downloads': 0, 'subagents': 0,
                          'local_project_qualified_problems_after_two_batches': 0,
                          'conclusion': 'No recommended current-project mainline found; keep externalrisk only and stop source expansion.'})

first = HERE.parent / '讨论伙伴_外部失败事件选题_20261028_v1'
correction = first / '06_关联纠正_Constellation仅作外部线索.txt'
addendum = first / 'correction_addendum_manifest.json'
assert correction.is_file()
if not addendum.exists():
    payload = {'scope': 'Recommendation-status correction only; earlier artifacts/manifests remain historical snapshots.',
               'correction': {'path': str(correction), 'sha256': sha(correction)},
               'current_project_status': 'Firstbatch not qualified; Constellation externalonly; see secondbatch record for final two-batch conclusion.'}
    with addendum.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')
files = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
         for path in sorted(HERE.rglob('*')) if path.is_file() and path.name != 'delivery_manifest.json']
save('delivery_manifest.json', {'status': 'PASS', 'artifacts': files})
print(json.dumps({'status': 'PASS', 'artifacts': len(files), 'public_records': len(public['records']),
                  'selected_external_event': 1, 'qualified_project_problems': 0,
                  'experiments': 0, 'forwards': 0, 'training': 0, 'reevaluation': 0, 'profiling': 0}, ensure_ascii=False))
