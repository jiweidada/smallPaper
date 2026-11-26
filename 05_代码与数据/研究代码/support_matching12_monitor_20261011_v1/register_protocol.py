"""Freeze this reader against the already dispatched immutable implementation."""
import ast
import sys
from verify_terminal_local import CORE,HERE,ROOT,PROTOCOL,read,sha,write

def main():
    assert not PROTOCOL.exists()
    construction=read(ROOT/'outputs/support_matching_20261011_v2/training_construction.json')
    files=construction['implementation'];assert len(files)==22
    for name,h in files.items():assert sha(CORE/name)==h,name
    tree=ast.parse((CORE/'runtime.py').read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IMPLEMENTATION' for t in n.targets))
    assert set(ast.literal_eval(node.value))==set(files)
    old=read(ROOT/'英文小论文/code/fdr_tail12_monitor_20261011_v1/local_reader_protocol.json')
    reader_names=('verify_terminal_local.py','decide_fixed12.py','register_protocol.py','validate_cpu.py')
    for n in reader_names:ast.parse((HERE/n).read_text(encoding='utf8'))
    p=dict(schema='support_cost_fixed12_local_reader_v1',endpoint=old['endpoint'],acceptance=old['acceptance'],
        core_files_sha256=files,implementation_sha256=files,reader_files_sha256={n:sha(HERE/n) for n in reader_names},
        reused_local_evidence=old['reused_local_evidence'],common_init_relative_path=old['common_init_relative_path'],
        prepared_before_new_terminal_file_consumption=True,new_support_terminal_packets_validated=0,
        new_model_forwards=0,new_training_epochs=0,
        source_reader_sha256=sha(ROOT/'英文小论文/code/fdr_tail12_monitor_20261011_v1/verify_terminal_local.py'),
        notes=['Inherited remote collector status FDR_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED is a legacy string; manifest and identity must be support-matching schemas.',
            'DN assignment/formulas remain native; ordinary GO counts can indirectly alter native DDF normalization.',
            'Snapshots validate actual candidate-output assignments and structural counts; do not infer a real BASE training trajectory.'])
    assert 'torch' not in sys.modules
    write(PROTOCOL,p)
    print('SUPPORT_COST_READER_FROZEN',sha(PROTOCOL),flush=True)

if __name__=='__main__':main()
