"""Create an isolated CPU-only revision; never deploy or start training."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / '英文小论文/code/range_context_decoder_20261005_v1'
DST = ROOT / '英文小论文/code/range_context_decoder_20261005_v3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DST.exists():
        raise RuntimeError('Preserve existing revision; do not overwrite')
    files = ['bootstrap.py', 'dataset.py', 'dual_range.py', 'dual_view.py', 'factory.py',
             'geometry.py', 'range_context.py', 'native_recipe.yml',
             'native_source_manifest.json', 'native_source_manifest_windows_v1.json',
             'sealed_source_before.json', 'verify_training_identity_cpu.py']
    before = {str(p.relative_to(SRC)): digest(p) for p in SRC.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    DST.mkdir()
    for name in files:
        shutil.copy2(SRC / name, DST / name)
    for name in ('native', 'local_deps'):
        shutil.copytree(SRC / name, DST / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    report = {'status': 'isolated_CPU_stage0_only_training_paused',
              'source': str(SRC), 'destination': str(DST),
              'original_source_hashes': before,
              'not_a_training_bundle': True, 'no_remote_actions': True}
    (DST / 'source_before_revision.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'destination': str(DST), 'source_files_preserved': len(before)}))


if __name__ == '__main__':
    main()
