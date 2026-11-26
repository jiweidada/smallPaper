"""Bind local result readers to the immutable Stem experiment."""
import ast,sys
from verify_terminal_local import CORE,HERE,ROOT,PROTOCOL,read,sha,write
assert not PROTOCOL.exists()
files=read(ROOT/'outputs/native_stem_scratch12_20261011_v3/local_code_manifest.json')
assert len(files)==24 and all(sha(CORE/n)==h for n,h in files.items())
tree=ast.parse((CORE/'runtime.py').read_text(encoding='utf8'))
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IMPLEMENTATION' for t in n.targets))
assert set(ast.literal_eval(node.value))==set(files)
old=read(ROOT/'英文小论文/code/support_matching12_monitor_20261011_v1/local_reader_protocol.json')
names=('verify_terminal_local.py','decide_fixed12.py','register_protocol.py','validate_cpu.py','compare_live_history.py','compare_stem_initialization.py')
for n in names:ast.parse((HERE/n).read_text(encoding='utf8'))
p=dict(schema='native_stem_scratch_fixed12_local_reader_v1',endpoint=old['endpoint'],acceptance=old['acceptance'],
    core_files_sha256=files,implementation_sha256=files,reader_files_sha256={n:sha(HERE/n) for n in names},
    reused_local_evidence=old['reused_local_evidence'],common_init_relative_path=old['common_init_relative_path'],
    prepared_before_new_terminal_file_consumption=True,new_stem_terminal_packets_validated=0,
    new_model_forwards=0,new_training_epochs=0,source_reader_sha256=sha(ROOT/'英文小论文/code/support_matching12_monitor_20261011_v1/verify_terminal_local.py'),
    notes=['Only native Stem initialization changes from cached pretrained BASE; all structure, native matching, criterion and augmentation remain.',
        'Shared initialization outside Stem verified; the original Stem uses seed-specific default initialization.',
        'New user-authorized scratch reference, no pre-existing BASE retraining. Its own fixed12 versus cached BASE gate is descriptive; enhanced Stem versus scratch is a separate comparison.'])
assert 'torch' not in sys.modules
write(PROTOCOL,p);print('NATIVE_STEM_SCRATCH_READER_FROZEN',sha(PROTOCOL))
