"""Bind local result readers to the immutable Stem experiment."""
import ast,sys
from verify_terminal_local import CORE,HERE,ROOT,PROTOCOL,read,sha,write
assert not PROTOCOL.exists()
files=read(ROOT/'outputs/stem_component12_20261011_v1/local_code_manifest.json')
assert len(files)==25 and all(sha(CORE/n)==h for n,h in files.items())
tree=ast.parse((CORE/'runtime.py').read_text(encoding='utf8'))
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IMPLEMENTATION' for t in n.targets))
assert set(ast.literal_eval(node.value))==set(files)
old=read(ROOT/'英文小论文/code/support_matching12_monitor_20261011_v1/local_reader_protocol.json')
names=('verify_terminal_local.py','decide_fixed12.py','register_protocol.py','validate_cpu.py','compare_live_history.py','compare_components.py')
for n in names:ast.parse((HERE/n).read_text(encoding='utf8'))
p=dict(schema='stem_component_fixed12_local_reader_v1',endpoint=old['endpoint'],acceptance=old['acceptance'],
    arms=list(read(CORE/'training_protocol.json')['execution_allowlist']),core_files_sha256=files,implementation_sha256=files,reader_files_sha256={n:sha(HERE/n) for n in names},
    reused_local_evidence=old['reused_local_evidence'],common_init_relative_path=old['common_init_relative_path'],
    prepared_before_new_terminal_file_consumption=True,new_stem_terminal_packets_validated=0,
    new_model_forwards=0,new_training_epochs=0,source_reader_sha256=sha(ROOT/'英文小论文/code/support_matching12_monitor_20261011_v1/verify_terminal_local.py'),
    notes=['One changed operation per control within the original Stem; all parameters retained.',
        'Shared initialization outside Stem verified; the original Stem uses seed-specific default initialization.',
        'Positive fixed12 results permit mechanism followup but do not themselves explain which Stem branch caused improvement.'])
assert 'torch' not in sys.modules
write(PROTOCOL,p);print('ORIGINAL_STEM_READER_FROZEN',sha(PROTOCOL))
