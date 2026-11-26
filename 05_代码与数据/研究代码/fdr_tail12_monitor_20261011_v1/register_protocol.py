"""Freeze the local reader before new terminal results become available."""
import ast
import sys
from verify_terminal_local import CORE, HERE, ROOT, PROTOCOL, read, sha, write


def main():
    assert not PROTOCOL.exists()
    freeze = read(ROOT / 'outputs/fdr_tail12_20261011_v2/package_freeze.json')
    assert freeze['schema'] == 'fdr_tail12_package_freeze_v2'
    for name, h in freeze['files'].items():
        assert sha(CORE / name) == h, name
    tree = ast.parse((CORE / 'runtime.py').read_text(encoding='utf8'))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'IMPLEMENTATION' for t in n.targets))
    names = ast.literal_eval(assignment.value)
    assert len(names) == 19 and set(names) == set(freeze['files']) - {'construction_provenance.json'}
    reuse = {}
    for seed in (1, 2):
        path = ROOT / f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1'
        report = path / 'verification_report_v1.json'
        r = read(report)
        assert r['status'] == 'PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
        assert r['identity']['seed'] == seed
        reuse[str(seed)] = dict(packet_relative_path=(path / 'extracted').relative_to(ROOT).as_posix(),
            report_relative_path=report.relative_to(ROOT).as_posix(), report_sha256=sha(report),
            manifest_sha256=sha(path / 'extracted/manifest.json'))
    reader_names = ('verify_terminal_local.py', 'decide_fixed12.py', 'register_protocol.py', 'validate_cpu.py')
    for name in reader_names:
        ast.parse((HERE / name).read_text(encoding='utf8'))
    value = dict(schema='fdr_fixed12_local_reader_v1',
        endpoint=dict(epochs=12, zero_based_epoch=11, weights='EMA'),
        acceptance=dict(per_seed_APs_delta_pp='strictly_positive', per_seed_AP_delta_pp='nonnegative', required_seeds=[1, 2]),
        core_files_sha256=freeze['files'], implementation_sha256={n:freeze['files'][n] for n in names},
        reader_files_sha256={n:sha(HERE / n) for n in reader_names}, reused_local_evidence=reuse,
        common_init_relative_path='数据/03_本轮实验记录_20260928/权重/common_init_seed0.pth',
        reference_checker_relative_path='英文小论文/code/scale_protection12_monitor_20261010_v1/verify_terminal_local.py',
        reference_checker_sha256=sha(ROOT / '英文小论文/code/scale_protection12_monitor_20261010_v1/verify_terminal_local.py'),
        prepared_before_new_terminal_file_consumption=True, new_FDR_terminal_packets_validated=0,
        new_model_forwards=0, new_training_epochs=0,
        note='New terminal archives reuse the already downloaded native sources and historical references; current bytes are rechecked against the new immutable binding. No scale-protection/Mosaic-event or inactive-first4 assumptions.')
    assert 'torch' not in sys.modules
    write(PROTOCOL, value)
    print('FDR_LOCAL_READER_FROZEN_BEFORE_NEW_TERMINAL_CONSUMPTION', sha(PROTOCOL), flush=True)


if __name__ == '__main__':
    main()
