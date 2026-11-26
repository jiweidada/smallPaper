"""Check whether hidden files or ignore rules change the bounded args inventory."""
from hashlib import sha256
from pathlib import Path
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INVENTORY = HERE / 'archived_args_inventory.json'
OUT = HERE / 'hidden_noignore_rescan.json'
EXCLUSIONS = ['!**/.git', '!**/.codex', '!**/.agents', '!**/runtime*/**', '!**/local_deps/**',
              '!**/build_tools/**', '!**/yolo_smoke_runtime_20261029', '!**/cpu_smoke_runtime_20261029',
              '!**/native_import_deps', '!**/isolated_runtime', '!**/analysis_tools',
              '!**/aps_official_eval_deps', '!**/c10_eval_test_deps']


def main():
    assert not OUT.exists()
    original = json.loads(INVENTORY.read_text(encoding='utf-8'))
    cmd = [shutil.which('rg'), '--files', '--hidden', '--no-ignore', '04_实验记录', '05_代码与数据', '90_历史与工具', '-g', 'args.json']
    for exclusion in EXCLUSIONS:
        cmd.extend(['-g', exclusion])
    result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace', check=False)
    assert result.returncode == 0 and not result.stderr
    found = {(ROOT / x.strip()).resolve() for x in result.stdout.splitlines() if x.strip()}
    old = {Path(x['args_path']).resolve() for x in original['records']}
    assert found == old
    output = {'status': 'HIDDEN_NOIGNORE_RESCAN_MATCHED',
              'inventory_sha256': sha256(INVENTORY.read_bytes()).hexdigest(),
              'files_found': len(found), 'added_or_missing_paths': 0, 'rg_exit_code': result.returncode,
              'rg_error_line_count': len(result.stderr.splitlines()), 'excluded_globs': EXCLUSIONS,
              'limits': ['Archives, excluded code/dependency directories, other filenames and remote systems are outside scope.',
                         'Names and configurations do not authenticate historical data or code.',
                         'No model, evaluation or training was run.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': output['status'], 'files_found': len(found),
                      'sha256': sha256(OUT.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
