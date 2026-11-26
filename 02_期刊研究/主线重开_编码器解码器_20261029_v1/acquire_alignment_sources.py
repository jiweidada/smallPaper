"""Read-only author-source acquisition. Never import or execute fetched projects."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'comparison_sources_v1'
GIT = r'C:\Program Files\Git\cmd\git.exe'
ENV = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never', GIT_LFS_SKIP_SMUDGE='1')
OPTS = [GIT, '-c', 'credential.helper=', '-c', 'http.sslVerify=true',
        '-c', 'http.sslBackend=openssl', '-c', 'http.version=HTTP/1.1',
        '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=15', '-c', 'core.autocrlf=false']

def command(*args, timeout=45):
    return subprocess.run(OPTS + list(args), env=ENV, capture_output=True,
                          timeout=timeout, encoding='utf-8', errors='replace')

def acquire(name, url, branch=None):
    dest = DEST / name
    row = dict(repo=url, branch=branch, local=str(dest), utc=datetime.now(timezone.utc).isoformat(),
               scope='source reading and Git blob verification only; no imports/builds/model execution')
    try:
        if dest.exists():
            result = command('-C', str(dest), 'remote', 'get-url', 'origin')
            assert result.returncode == 0 and result.stdout.strip() == url
        else:
            args = ['clone', '--depth=1', '--single-branch', '--no-tags']
            if branch:
                args += ['--branch', branch]
            result = command(*args, url, str(dest))
            assert result.returncode == 0, result.stderr[:1500]
        result = command('-C', str(dest), 'rev-parse', 'HEAD')
        assert result.returncode == 0
        row['commit'] = result.stdout.strip()
        entries = subprocess.check_output(OPTS + ['-C', str(dest), 'ls-tree', '-r', '-z', 'HEAD'], env=ENV).split(b'\0')
        checked = []
        for entry in filter(None, entries):
            meta, filename = entry.split(b'\t', 1)
            mode, kind, oid = meta.split()
            assert kind == b'blob' and mode not in (b'120000', b'160000')
            rel = filename.decode('utf-8')
            path = (dest / rel).resolve()
            assert path.is_relative_to(dest.resolve())
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert blob == oid.decode(), rel
            checked.append(dict(path=rel, bytes=len(data), git_blob_sha=blob,
                                sha256=hashlib.sha256(data).hexdigest()))
        row.update(status='ALL_TRACKED_BLOBS_VERIFIED', files=checked)
    except Exception as exc:
        row.update(status='ACQUISITION_OR_VERIFICATION_FAILED', error=str(exc)[:1600])
    return row

if __name__ == '__main__':
    import sys
    spec = sys.argv[1]
    if spec == 'DAPE':
        row = acquire('DAPE', 'https://github.com/xiuqhou/DAPE.git')
    elif spec == 'TAPTRv2':
        row = acquire('TAPTRv2', 'https://github.com/IDEA-Research/TAPTR.git', 'TAPTRv2')
    else:
        raise ValueError(spec)
    (DEST / (spec + '_source_verification.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k != 'files'} | {'tracked_files':len(row.get('files', []))}, ensure_ascii=False))
