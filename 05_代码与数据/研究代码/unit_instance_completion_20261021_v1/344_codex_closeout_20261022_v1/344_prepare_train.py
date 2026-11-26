"""New CPU smoke entry binds exactly the sealed 6470 raw-mainline targets."""
import importlib,difflib,ast
C=importlib.import_module('344_context')
manifest=C.HERE/'344_development6470_逐条SHA清单.json';mh=C.sha(manifest)
old=(C.S2/'train342.py').read_text(encoding='utf-8')
new=old.replace('342 CPU pretraining review link','344 CPU pretraining review link').replace('Based on train341.','Based on train342; sealed new targets only.')
new=new.replace('import geom342 as geom',"from importlib import import_module as _import_module\n_ctx344 = _import_module('344_context')\ngeom = _ctx344.load_geometry()")
start=new.index('def load_entry(seed, iid):');end=new.index('\ndef collate(entries):',start)
loader='''MANIFEST_PATH = _ctx344.HERE / '344_development6470_逐条SHA清单.json'
MANIFEST_SHA256 = 'REPLACE_MANIFEST_SHA'
_bound_manifest = None
INPUT_COUNTS = {'manifest_SHA_checks': 0, 'entry_SHA_checks': 0, 'payload_SHA_checks': 0, 'entry_loads': 0, 'nonpositive_raw_masks_verified': 0}

def _manifest():
    global _bound_manifest
    assert _ctx344.sha(MANIFEST_PATH) == MANIFEST_SHA256, 'sealed manifest SHA mismatch'
    INPUT_COUNTS['manifest_SHA_checks'] += 1
    if _bound_manifest is None:
        document = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        records = document['entries']
        assert len(records) == 6470
        _bound_manifest = {(r['seed'],r['image_id']):r for r in records}
        ids = set(load_order())
        assert set(_bound_manifest) == {(s,i) for s in (1,2) for i in ids}
        assert sum(len(r['fullp_nonpositive_indices']) for r in records) == 1
    return _bound_manifest

def _verify_entry_record(seed, iid, record):
    from pathlib import Path
    expected = (_ctx344.HERE/'344_cache_development_seed2/344_001466.pt' if (seed,iid)==(2,1466)
                else _ctx344.S343/f'343_cache_development_seed{seed}'/f'343_{iid:06d}.pt')
    path = Path(record['path'])
    assert path.resolve() == expected.resolve(), 'unapproved cache source'
    assert _ctx344.sha(path) == record['file_sha256'], 'entry file SHA mismatch'
    INPUT_COUNTS['entry_SHA_checks'] += 1
    e = torch.load(path, map_location='cpu', weights_only=True)
    INPUT_COUNTS['entry_loads'] += 1
    assert e['seed']==seed and e['image_id']==iid
    # Reuse the already-approved byte serialization, with no builder import.
    def payload_sha(value):
        h=hashlib.sha256()
        def add(v):
            if torch.is_tensor(v):
                h.update(b'tensor');h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
                h.update(v.contiguous().numpy().tobytes())
            elif isinstance(v,dict):
                for k in sorted(v):
                    if k=='stage3_payload_sha256':continue
                    h.update(k.encode());add(v[k])
            elif isinstance(v,(list,tuple)):
                h.update(str(len(v)).encode())
                for x in v:add(x)
            else:h.update(json.dumps(v,sort_keys=True).encode())
        add(value);return h.hexdigest()
    assert payload_sha(e)==e['stage3_payload_sha256']==record['payload_sha256'], 'payload SHA mismatch'
    INPUT_COUNTS['payload_SHA_checks'] += 1
    assert e['raw_cache_file_sha256']==record['source_340_file_sha256']
    assert e['checkpoint_sha256']==record['checkpoint_sha256']
    bad=geom.fullp_nonpositive_mask(e['final300_boxes'],e['fullp_query_index'])
    assert bad.nonzero().flatten().tolist()==record['fullp_nonpositive_indices']
    if 'fullp_nonpositive_mask' in e:assert torch.equal(bad,e['fullp_nonpositive_mask'])
    assert int(bad.sum())==int((seed,iid)==(2,1466))
    assert not e['m0_legal_mask'][bad].any() and (e['m0_cand_owner'][bad]==-1).all()
    # Old 343 files are immutable; attach their verified all-false field in RAM.
    e['fullp_nonpositive_mask']=bad
    geom.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],bad)
    INPUT_COUNTS['nonpositive_raw_masks_verified'] += 1
    return e

def load_entry(seed, iid):
    assert seed in (1,2) and iid in set(load_order()), 'development-only input'
    record=_manifest()[seed,iid]
    return _verify_entry_record(seed,iid,record)
'''.replace('REPLACE_MANIFEST_SHA',mh)
new=new[:start]+loader+new[end:]
assert new.count('keep = geom.intersecting_candidates(b, W)')==2
new=new.replace('keep = geom.intersecting_candidates(b, W)', 'keep = geom.intersecting_fullp_candidates(b, W, e["fullp_nonpositive_mask"])')
new+='''

if __name__ == '__main__':
    raise SystemExit('CPU bounded smoke module only; use the sealed 344 engineering validator. No real-training entry.')
'''
ast.parse(new)
C.text('train344.py',new)
C.text('344_train344_vs_train342.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='342_codex_pretrain_20261022_v1/train342.py',tofile='344_codex_closeout_20261022_v1/train344.py')))
C.write('344_result_train344输入绑定生成.json',{'status':'已生成，待验证','train344_path':str(C.HERE/'train344.py'),
    'train344_sha256':C.sha(C.HERE/'train344.py'),'manifest_path':str(manifest),'manifest_sha256':mh,
    '已执行':{'new_training_module_writes':1,'diff_writes':1,'AST_parses':1,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
    '未执行':['真实训练','GPU','阶段5'],'未核':['新入口全量读取与冒烟','主控独立复核'],'source_sha256':C.sources()})
