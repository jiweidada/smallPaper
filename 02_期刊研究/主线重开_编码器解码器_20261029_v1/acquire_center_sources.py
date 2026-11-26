"""Acquire three author-linked source trees for reading only; no model execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'comparison_sources_v1'
GIT = r'C:\Program Files\Git\cmd\git.exe'
ENV = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never', GIT_LFS_SKIP_SMUDGE='1')
OPTS = [GIT, '-c', 'credential.helper=', '-c', 'http.sslVerify=true',
        '-c', 'http.sslBackend=openssl', '-c', 'http.version=HTTP/1.1',
        '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=15',
        '-c', 'core.autocrlf=false']
REPOS = [('DS-Det', 'https://github.com/Med-Process/DS-Det.git'), ('DAB-DETR', 'https://github.com/IDEA-Research/DAB-DETR.git'), ('dsntnn', 'https://github.com/anibali/dsntnn.git')]

def git(*args, timeout=40):
    return subprocess.run(OPTS + list(args), env=ENV, capture_output=True,
                          text=True, encoding='utf-8', errors='replace', timeout=timeout)

receipts = []
for name, url in REPOS:
    dest = DEST / name
    existed = dest.exists()
    row = {'repo': url, 'local': str(dest), 'utc': datetime.now(timezone.utc).isoformat(),
           'scope': 'source acquisition and Git blob verification only; no imported project code',
           'existing_checkout_reused': existed}
    try:
        if existed:
            origin = git('-C', str(dest), 'remote', 'get-url', 'origin')
            assert origin.returncode == 0 and origin.stdout.strip() == url
        else:
            result = git('clone', '--depth=1', '--single-branch', '--no-tags', url, str(dest))
            assert result.returncode == 0, result.stderr[:1500]
        commit = git('-C', str(dest), 'rev-parse', 'HEAD')
        assert commit.returncode == 0
        tree = subprocess.check_output(OPTS + ['-C', str(dest), 'ls-tree', '-r', '-z', 'HEAD'], env=ENV)
        checked = []
        for entry in tree.split(b'\0'):
            if not entry:
                continue
            meta, filename = entry.split(b'\t', 1)
            mode, kind, oid = meta.split()
            assert kind == b'blob' and mode not in (b'120000', b'160000')
            rel = filename.decode('utf-8')
            path = (dest / rel).resolve()
            assert path.is_relative_to(dest.resolve())
            data = path.read_bytes()
            digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert digest == oid.decode(), rel
            checked.append({'path': rel, 'bytes': len(data), 'git_blob_sha': digest,
                            'sha256': hashlib.sha256(data).hexdigest()})
        row.update(status='ALL_TRACKED_BLOBS_VERIFIED', commit=commit.stdout.strip(), files=checked)
    except Exception as exc:
        row.update(status='ACQUISITION_OR_VERIFICATION_FAILED', error=str(exc)[:1600])
    receipts.append(row)
    (DEST / 'center_acquisition_receipts.json').write_text(
        json.dumps(receipts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in row.items() if k != 'files'} |
                     {'tracked_file_count': len(row.get('files', []))}, ensure_ascii=False), flush=True)
