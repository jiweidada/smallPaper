"""Retrieve public pinned source metadata/operators; do not execute fetched code."""
from pathlib import Path
import json
import sys
from fetch_alignment_operator_sources import fetch, operator_sources

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'source_audit_v1'
REPOS = {'LPS': 'raymondyeh07/learnable_polyphase_sampling',
         'SPD': 'LabSAINT/SPD-Conv', 'SAPA': 'tiny-smart/sapa',
         'LoftUp': 'andrehuang/loftup'}
mode, name = sys.argv[1:3]
repo = REPOS[name]
commit_file = AUDIT / (name + '_A14_author_commit.json')
tree_file = AUDIT / (name + '_A14_author_tree.json')
if mode == 'metadata':
    meta = json.loads(commit_file.read_bytes() if commit_file.exists() else
        fetch('https://api.github.com/repos/' + repo + '/commits/HEAD', commit_file))
    commit = meta['sha']
    tree = json.loads(tree_file.read_bytes() if tree_file.exists() else fetch(
        'https://api.github.com/repos/' + repo + '/git/trees/' + commit + '?recursive=1', tree_file))
    assert not tree.get('truncated')
    assert tree['sha'] in (commit, meta['commit']['tree']['sha'])
    words = ['poly', 'upsampl', 'sapa', 'loft', 'model', 'common.py', 'readme']
    paths = [x['path'] for x in tree['tree'] if x['type'] == 'blob'
        and x['path'].lower().endswith(('.py', '.md', '.yaml'))
        and any(w in x['path'].lower() for w in words)]
    print(json.dumps({'name': name, 'repo': repo, 'commit': commit, 'paths': paths}, indent=2))
elif mode == 'operators':
    meta = json.loads(commit_file.read_bytes())
    tree = json.loads(tree_file.read_bytes())
    assert sys.argv[3:]
    operator_sources(name + '_A14', repo, meta['sha'], tree, sys.argv[3:])
else:
    raise ValueError(mode)
