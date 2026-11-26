from pathlib import Path
import ast,difflib,hashlib,json
root=Path('英文小论文/code/warm_stem_primary_collection12_20261011_v1')
old=root/'verify_primary.py';s=old.read_text(encoding='utf8')
s=s.replace("cfg=runtime.resolved_config(plan,{k:Path(v) for k,v in receipt['paths'].items()})","cfg=runtime.resolved_config(plan,{k:PurePosixPath(v) for k,v in receipt['paths'].items()})")
b="        destination.mkdir(exist_ok=False)\n";assert s.count(b)==1
start=s.index(b);end=s.index("    run=destination/'run'",start)
block=s[start:end]
new="""        if destination.exists():
            assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}==set(names)
            assert z.testzip() is None
            assert sha(destination/'primary_manifest.json')==meta['manifest_sha256']
            for n,item in manifest['files'].items():
                assert (destination/n).stat().st_size==item['bytes'] and sha(destination/n)==item['sha256'],n
        else:
"""+''.join('    '+line if line.strip() else line for line in block.splitlines(True))
s=s[:start]+new+s[end:];ast.parse(s)
p=root/'verify_primary_v2.py';p.write_text(s,encoding='utf8',newline='\n')
(root/'verification_repair_v2.diff').write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf8').splitlines(True),s.splitlines(True),fromfile=str(old),tofile=str(p))),encoding='utf8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
failure=dict(status='INITIAL_LOCAL_READER_FAILED_REMOTE_POSIX_PATHS_CONVERTED_TO_WINDOWS_PATHS',
 initial_source_sha256=sha(old),failure_line=76,artifact_corruption_found=False,
 actual_run_args_equal_bundle_resolved=True,computed_difference_only_four_remote_path_fields=True,
 corrected_reader_sha256=sha(p),repair='Use PurePosixPath for immutable remote path values; inspect and hash complete existing extraction rather than extracting again.',
 original_reader_kept=True,remote_frozen_source_changed=False,new_model_forwards=0,new_training_epochs=0)
with Path('outputs/warm_stem_primary_collection12_20261011_v1/local_reader_failure_v1.json').open('x',encoding='utf8') as f:json.dump(failure,f,indent=2)
analysisroot=Path('英文小论文/code/warm_stem_primary_analysis12_20261011_v1')
oldanalysis=analysisroot/'analyze.py';a=oldanalysis.read_text(encoding='utf8')
assert a.count("HERE/'protocol.json'")==2;a=a.replace("HERE/'protocol.json'","HERE/'protocol_v2.json'")
a=a.replace('英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary.py','英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py')
ast.parse(a);newanalysis=analysisroot/'analyze_v2.py';newanalysis.write_text(a,encoding='utf8',newline='\n')
protocol=json.loads((analysisroot/'protocol.json').read_bytes());d=protocol['dependencies_sha256']
del d['英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary.py']
d['英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py']=sha(p)
protocol.update(analysis_sha256=sha(newanalysis),supersedes_unexecuted_analysis_protocol='protocol.json',local_reader_repair_only=True)
with (analysisroot/'protocol_v2.json').open('x',encoding='utf8') as f:json.dump(protocol,f,indent=2)
print(json.dumps(dict(status='POSIX_PATH_READER_REPAIR_SOURCE_BOUND_NO_REMOTE_EXPERIMENT_RETRY',verification=sha(p),analysis=sha(newanalysis))))
