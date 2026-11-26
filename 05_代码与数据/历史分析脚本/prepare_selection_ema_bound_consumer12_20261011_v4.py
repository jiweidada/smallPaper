"""Bind six-record consumer to exact checkpoint buffers; preserve prior versions."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent.parent
old=ROOT/'英文小论文/code/native_selection_analysis12_20261011_v3'
new=ROOT/'英文小论文/code/native_selection_analysis12_20261011_v4'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=json.loads((old/'protocol.json').read_bytes());assert {n:sha(old/n) for n in prior['files']}==prior['files']
new.mkdir(exist_ok=False)
helper='''
def checkpoint_anchor_reference(job, trained):
    """An exported actual savedEMA buffer, never a freshly generated proxy."""
    assert sha(job['anchor_audit_report'])==job['anchor_audit_report_sha256']
    report=read(job['anchor_audit_report']);assert report['status']=='PASS_CHECKPOINT_BOUND_EMA_MODEL_AND_NATIVE_GENERATED_ANCHOR_TENSOR_AUDIT'
    assert report['seed']==job['seed'] and report['model_forwards']==report['training_epochs']==report['optimizer_updates']==0 and report['CUDA_initialized'] is False
    assert report['native_generator_source_sha256']=='b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5'
    assert report['EMA_source_sha256']=='aba8ace1dacfe68b2b24e51bdb908ea17c0c2b2e1ebf2e0496cd1499a16a2cff'
    case=report['cases'][job['arm']];assert case['checkpoint_sha256']==trained['last_checkpoint_sha256']
    assert sha(job['anchor_export'])==job['anchor_export_sha256']==case['export_sha256']
    values=npz(job['anchor_export']);assert set(values)=={'EMA_anchors','model_anchors','generated_anchors'}
    a=values['EMA_anchors'];assert a.shape==(8400,4) and a.dtype==np.float32 and hashlib.sha256(a.tobytes()).hexdigest()==case['array_sha256']
    assert case['model_equal_generated'] and np.array_equal(values['model_anchors'],values['generated_anchors'])
    verify_grid_centers(dict(dense_anchor_logits=a),a)
    return a

'''
grid='''def verify_grid_centers(record, checkpoint_anchors=None):
    """Exact savedEMA equality, then validate nominal feature-cell assignment."""
    anchors=np.asarray(record['dense_anchor_logits'])
    assert checkpoint_anchors is not None, 'Require a checkpoint-bound actual EMA anchor buffer'
    assert anchors.shape==(8400,4) and anchors.dtype==np.float32
    assert np.array_equal(anchors,checkpoint_anchors), 'Observed buffer differs from bound savedEMA'
    valid=np.isfinite(anchors).all(1);invalid=np.isposinf(anchors).all(1);assert (valid|invalid).all() and valid.any()
    expected=[];sizes=[]
    for height,width in SHAPES:
        row,col=np.meshgrid(np.arange(height),np.arange(width),indexing='ij')
        expected.append(np.stack([(col+.5)/width,(row+.5)/height],-1).reshape(-1,2))
        sizes.append(np.tile([width,height],(height*width,1)))
    expected=np.concatenate(expected);sizes=np.concatenate(sizes)
    xy=1/(1+np.exp(-anchors[valid,:2].astype(np.float64)))
    assert np.array_equal(np.floor(xy*sizes[valid]),np.floor(expected[valid]*sizes[valid])), 'EMA center moved into a different nominal cell'
    return float(np.max(np.abs(xy-expected[valid])))


'''
for p in old.glob('*.py'):
    s=p.read_text(encoding='utf8').replace('actual_selection_offline_analysis_v3','actual_selection_offline_analysis_v4').replace('selection_v3_','selection_v4_')
    if p.name=='selection_core.py':
        start=s.index('def verify_grid_centers(');end=s.index('def validate_record(',start);s=s[:start]+grid+s[end:]
    if p.name=='analyze.py':
        s=s.replace('\ndef binding(job, p, reader, base):','\n'+helper+'\ndef binding(job, p, reader, base):')
        s=s.replace('    return identity\n','    checkpoint_anchor_reference(job,trained)\n    return identity\n')
        s=s.replace("    folder, run = Path(job['diagnostic']), Path(job['packet']) / 'run'", "    folder, run = Path(job['diagnostic']), Path(job['packet']) / 'run'\n    anchor_reference=checkpoint_anchor_reference(job,read(run/'training_completion.json'))")
        s=s.replace('verify_grid_centers(selection)','verify_grid_centers(selection,anchor_reference)')
    compile(s,str(new/p.name),'exec');(new/p.name).write_text(s,encoding='utf8',newline='\n')
assert {n:sha(old/n) for n in prior['files']}==prior['files']
print('Six-record consumer v4 prepared; exact cachedEMA buffer required; prior files preserved')
