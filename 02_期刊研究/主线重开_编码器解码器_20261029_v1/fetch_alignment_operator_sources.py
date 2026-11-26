"""Pinned public HTTPS source retrieval; no fetched-code execution or imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import base64
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'source_audit_v1'
DEST = ROOT / 'comparison_sources_v1'
CURL = r'D:\Anaconda\Library\bin\curl.exe'

def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    assert target.resolve().is_relative_to(ROOT.resolve())
    result = subprocess.run([CURL, '--location', '--fail', '--max-time', '25', '--silent',
                             '--show-error', url, '--output', str(target)],
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[:1200]
    return target.read_bytes()

def operator_sources(name, repo, commit, tree, paths):
    bypath = {x['path']:x for x in tree['tree'] if x['type']=='blob'}
    out = DEST / (name + '_operators_' + commit[:7])
    files = []
    receipt = dict(repo=repo, commit=commit, tree_response_locator=tree['sha'], local=str(out),
                   scope='listed operator/config/README files only; not full repository; no imports/builds/forward',
                   utc=datetime.now(timezone.utc).isoformat(), files=files)
    for rel in paths:
        entry = bypath[rel]
        assert entry['mode'] != '120000'
        path = out / rel
        url = 'https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + rel
        if path.exists():
            data = path.read_bytes()
        else:
            # Git/raw hosts currently fail TLS from this workstation. Use the
            # public Git blob API on the metadata host instead, preserving TLS.
            api_url = 'https://api.github.com/repos/' + repo + '/git/blobs/' + entry['sha']
            blob_json = out / '_retrieval_metadata' / (entry['sha'] + '.json')
            payload = json.loads(blob_json.read_bytes() if blob_json.exists() else fetch(api_url, blob_json))
            assert payload['sha'] == entry['sha'] and payload['encoding'] == 'base64'
            data = base64.b64decode(payload['content'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert blob == entry['sha'], rel
        files.append(dict(path=rel, bytes=len(data), git_blob_sha=blob,
                          sha256=hashlib.sha256(data).hexdigest(), url=url))
        print(name + ': verified ' + rel, flush=True)
        (DEST / (name + '_operator_source_receipt.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    receipt['status'] = 'ALL_LISTED_OPERATOR_FILES_GIT_BLOB_VERIFIED'
    (DEST / (name + '_operator_source_receipt.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'} | {'files':len(files)}, ensure_ascii=False))

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'DAPE':
        meta = json.loads((AUDIT/'DAPE_author_commit.json').read_text(encoding='utf-8'))
        tree = json.loads((AUDIT/'DAPE_git_tree_current.json').read_text(encoding='utf-8'))
        # The public endpoint response uses the supplied commit locator in its
        # sha/url fields. Keep it distinct from commit.tree.sha; verify files
        # against the listed Git blob IDs rather than mislabeling a tree ID.
        assert tree['sha'] in (meta['sha'], meta['commit']['tree']['sha'])
        operator_sources('DAPE', 'xiuqhou/DAPE', meta['sha'], tree,
                         sys.argv[2:] or ['models/dape/dape_transformer.py', 'models/dape/dape.py',
                                          'configs/dape/models/dape_r50.py', 'README.md'])
    elif mode == 'TAPTRtree':
        branches = json.loads((AUDIT/'TAPTR_author_branches.json').read_text(encoding='utf-8'))
        commit = next(x['commit']['sha'] for x in branches if x['name']=='v2')
        target = AUDIT/'TAPTRv2_git_tree.json'
        data = fetch('https://api.github.com/repos/IDEA-Research/TAPTR/git/trees/'+commit+'?recursive=1', target)
        tree = json.loads(data)
        print(json.dumps(dict(commit=commit, tree_sha=tree['sha'], truncated=tree.get('truncated'),
                              paths=[x['path'] for x in tree['tree'] if x['type']=='blob' and any(k in x['path'].lower() for k in ('transformer', 'attention', 'readme', 'config'))]), indent=2))
    elif mode == 'TAPTRv2':
        branches = json.loads((AUDIT/'TAPTR_author_branches.json').read_text(encoding='utf-8'))
        commit = next(x['commit']['sha'] for x in branches if x['name']=='v2')
        tree = json.loads((AUDIT/'TAPTRv2_git_tree.json').read_text(encoding='utf-8'))
        operator_sources('TAPTRv2', 'IDEA-Research/TAPTR', commit, tree, sys.argv[2:])
    else:
        raise ValueError(mode)
