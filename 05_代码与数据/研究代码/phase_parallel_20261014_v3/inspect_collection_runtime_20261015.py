import json
import pathlib
root = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if pathlib.Path('/25085400120/DEIM').is_dir()
                    else '/24085403014/experiments/phase_mechanism_20261012')
result = {'files': {}, 'stdin_python_processes': [], 'meminfo': pathlib.Path('/proc/meminfo').read_text().splitlines()[:3]}
for path in [root/'artifacts/common_init_seed0.pth', *list((root/'outputs').glob('*/best_stg2.pth'))]:
    stat = path.stat()
    result['files'][str(path)] = {'bytes': stat.st_size, 'mtime': stat.st_mtime}
for directory in pathlib.Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        args = [v.decode(errors='replace') for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
        if args and args[-1] == '-' and any('python' in a for a in args[:1]):
            result['stdin_python_processes'].append({'pid': directory.name, 'argv': args,
                'stat': (directory/'stat').read_text(), 'wchan': (directory/'wchan').read_text()})
    except OSError:
        pass
print(json.dumps(result))
