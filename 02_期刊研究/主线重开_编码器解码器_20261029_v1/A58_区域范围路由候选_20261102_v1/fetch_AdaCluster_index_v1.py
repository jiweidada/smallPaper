"""Read public author repository metadata only; no downloaded code is executed."""
from pathlib import Path
import urllib.request
import json
import hashlib

P = Path(__file__).resolve().parent
D = P / 'sources/AdaCluster_index_v1'
D.mkdir(parents=True, exist_ok=False)


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'bounded-literature-source-check',
                                              'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=20) as response:
        assert response.status == 200
        return response.read()


try:
    commit = fetch('https://api.github.com/repos/USTC-MLSys/Adacluster/commits/main')
    obj = json.loads(commit)
    revision = obj['sha']
    assert len(revision) == 40 and all(v in '0123456789abcdef' for v in revision)
    tree = fetch('https://api.github.com/repos/USTC-MLSys/Adacluster/git/trees/' + revision + '?recursive=1')
    rows = json.loads(tree)
    assert not rows['truncated']
    (D / 'commit.json').write_bytes(commit)
    (D / 'tree.json').write_bytes(tree)
    selected = [r['path'] for r in rows['tree'] if r['type'] == 'blob' and r['path'].endswith('.py')
                and any(v in r['path'] for v in ['spas_sage_attn', 'modify_cogvideo', 'triton_kernel'])]
    receipt = {'status': 'PUBLIC_AUTHOR_INDEX_FETCHED', 'commit_sha': revision,
               'commit_metadata_sha256': hashlib.sha256(commit).hexdigest(),
               'tree_sha256': hashlib.sha256(tree).hexdigest(),
               'candidate_source_paths': selected, 'source_files_downloaded': 0,
               'downloaded_code_executed': False, 'new_model_forwards': 0}
    with (D / 'receipt.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt))
except BaseException as e:
    with (D / 'failure.json').open('x', encoding='utf-8') as f:
        json.dump({'status': 'INDEX_READ_FAILED', 'error_type': type(e).__name__, 'error': str(e),
                   'downloaded_code_executed': False, 'new_model_forwards': 0}, f, indent=2)
    raise
