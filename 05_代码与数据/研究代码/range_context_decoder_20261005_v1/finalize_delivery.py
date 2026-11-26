"""Archive integrity and verification receipts; do not contact training hosts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
LIBRARY = ROOT.parents[1] / '文献_相关原始论文_20261004'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    audit_path = ROOT / 'delivery_audit.json'
    if audit_path.exists():
        raise RuntimeError('Preserving previous final delivery audit')
    functional = json.loads((ROOT / 'cpu_verification_v2.json').read_text(encoding='utf8'))
    training = json.loads((ROOT / 'training_identity_cpu.json').read_text(encoding='utf8'))
    common = json.loads((ROOT / 'common_init_cpu.json').read_text(encoding='utf8'))
    traces = json.loads((ROOT / 'trace_geometry_verification.json').read_text(encoding='utf8'))
    design = json.loads((ROOT / 'variants_design.json').read_text(encoding='utf8'))
    before = json.loads((ROOT / 'sealed_source_before.json').read_text(encoding='utf-8-sig'))
    library = json.loads((LIBRARY / '_hashes.json').read_text(encoding='utf8'))
    native = json.loads((ROOT / 'native_source_manifest.json').read_text(encoding='utf8'))
    checks = {
        '48_functional_checks_passed': len(functional['checks']) == 48 and all(functional['checks'].values()),
        'training_DN_BN_loss_gradient_identity_passed': all(training['checks'].values()),
        'common_initialization_verified': all(common['checks'].values()),
        'geometry_all_32_images_170_windows_verified': traces['images'] == 32 and traces['windows'] == 170,
        'original_170_trace_hashes_unchanged': all(sha(WORKSPACE / '英文小论文/data/00_实验计划与证据索引/分布反馈深入自审_20261004/p0_real_cpu_v3_level_control/traces' / r['file']) == r['sha256'] for r in traces['unchanged_trace_hashes']),
        'sealed_source_protocol_all_inventory_hashes_unchanged': all(sha(Path(before['source']) / r['file']) == r['sha256'] for r in before['files']),
        'native_snapshot_hashes_unchanged': all(sha(ROOT / r['local']) == r['sha256'] for r in native['files']),
        'all_archived_PDF_hashes_verified': all(sha(LIBRARY / r['file']) == r['sha256'] for r in library['files']),
        'first_failed_validation_preserved': (ROOT / 'cpu_verification_v1.log').exists() and (ROOT / 'failed_attempts/first_cpu_validation/range_context.py').exists(),
        'design_execution_still_paused': design['execution_status'] == 'paused_by_user_design_only' and not design['training_started'] and not design['platform_uploaded'],
        'no_CPU_verification_initialized_CUDA': all(r['checks']['CUDA_never_initialized'] for r in (functional, training, common)) and not traces['CUDA_initialized'],
    }
    own_py = [p for p in ROOT.glob('*.py')]
    for p in own_py:
        ast.parse(p.read_text(encoding='utf8'), filename=str(p))
    checks['new_top_level_python_syntax_valid'] = True
    assert all(checks.values())
    manifest = {'root': str(ROOT), 'purpose': 'isolated_candidate_source_not_frozen_experiment_or_execution_queue',
                'files': [{'file': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                          for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.py', '.yml', '.json', '.txt')
                          and '__pycache__' not in p.parts and p.name not in ('delivery_manifest.json', 'delivery_audit.json')]}
    (ROOT / 'delivery_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    receipt = {'status': 'isolated_candidate_implemented_CPU_verified_training_paused', 'checks': checks,
               'functional_boolean_checks_total': len(functional['checks']) + len(training['checks']) + len(common['checks']),
               'source_manifest_sha256': sha(ROOT / 'delivery_manifest.json'),
               'original_source_inventory_verified_files': len(before['files']),
               'native_manifest_verified_files': len(native['files']), 'library_verified_PDFs': len(library['files']),
               'new_reader_parameters': functional['new_reader_parameters'],
               'small_trace_distance_mean_px': traces['groups']['small_nontruncated_firstIoU_ge_0.1']['distance_mean_px'],
               'limitations': ['No candidate AP, repeated training or novelty confirmation',
                               'No GPU capacity or measured latency', 'No platform upload or train/queue resume',
                               'Remote train status not inspected in this coding turn']}
    audit_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    run()
