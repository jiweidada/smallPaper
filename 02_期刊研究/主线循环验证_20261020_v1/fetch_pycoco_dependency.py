"""Fetch one pinned official PyPI wheel after pip setup stalled; no global install."""
import hashlib, json, struct, sys, urllib.request, zipfile
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'dependency_setup'
DEST.mkdir(exist_ok=True)
process = psutil.Process()
process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
process.cpu_affinity(process.cpu_affinity()[-1::-2][:2])
assert sys.version_info >= (3, 12) and struct.calcsize('P') == 8
assert sys.platform == 'win32' and 'AMD64' in sys.version
metadata_url = 'https://pypi.org/pypi/pycocotools/2.0.11/json'
expected_name = 'pycocotools-2.0.11-cp312-abi3-win_amd64.whl'
try:
    print('Fetching pinned PyPI metadata', flush=True)
    with urllib.request.urlopen(metadata_url, timeout=20) as r:
        metadata = json.load(r)
    entry = next(x for x in metadata['urls'] if x['filename'] == expected_name)
    assert entry['url'].startswith('https://files.pythonhosted.org/packages/')
    print('Fetching one Windows wheel', flush=True)
    with urllib.request.urlopen(entry['url'], timeout=20) as r:
        blob = r.read(2*1024*1024)
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == entry['digests']['sha256']
    wheel = DEST/expected_name
    with wheel.open('xb') as f:
        f.write(blob)
    runtime = ROOT/'runtime_pycoco'
    runtime.mkdir(exist_ok=False)
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            path = (runtime/info.filename).resolve()
            assert path.is_relative_to(runtime.resolve())
        z.extractall(runtime)
    receipt = {'state': 'complete', 'metadata_url': metadata_url, 'url': entry['url'],
               'wheel': str(wheel), 'sha256': digest, 'bytes': len(blob),
               'runtime': str(runtime), 'global_install': False,
               'previous_pip_attempt': 'Stopped only owned verified pip-download process/session6877 after no output and excess setup time; no wheel produced; no experiment run.',
               'TLS_certificate_verification': 'default verified SSL context, not disabled'}
    with (DEST/'receipt.json').open('x', encoding='utf8') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)
except BaseException as e:
    with (DEST/'failure.json').open('x', encoding='utf8') as f:
        json.dump({'state': 'failed', 'error': type(e).__name__+': '+str(e),
                   'no_model_execution': True, 'no_automatic_identical_retry': True}, f, indent=2)
    raise
