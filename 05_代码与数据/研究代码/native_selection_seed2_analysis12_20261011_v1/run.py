"""Explicit seed2 descriptive analysis; the six-record consumer remains unfulfilled."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def prepared():
    spec=read(HERE/'protocol.json');assert spec['reader_sha256']==sha(__file__)
    for n,h in spec['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    sys.path.insert(0,str(ROOT/'英文小论文/code/native_selection_analysis12_20261011_v4'))
    import analyze as a
    p,helper,parts,aggregate=a.prepared();assert p['all_six_formal_terminals_and_exact_replays_required']
    return spec,a,p,helper,parts,aggregate
def jobs():
    result=[]
    for arm,kind in [('original_stem','full'),('native_stem_scratch','reference'),('highfreq_rgb','control')]:
        formal=ROOT/f'outputs/stem_fixed12_seed2_formal_collection12_20261011_v1/seed2_{arm}';diag=ROOT/f'outputs/native_selection_collection12_20261011_v2/seed2_{arm}/extracted/diagnostic';anchors=ROOT/'outputs/native_selection_anchor_audit12_20261011_v2/seed2'
        result.append(dict(arm=arm,kind=kind,seed=2,packet=str(formal/'extracted'),report=str(formal/'verification.json'),report_sha256=sha(formal/'verification.json'),diagnostic=str(diag),verification_sha256=sha(diag/'verification.json'),selection_manifest_sha256=sha(diag/'selection_epoch12/manifest.json'),anchor_audit_report=str(anchors/'report.json'),anchor_audit_report_sha256=sha(anchors/'report.json'),anchor_export=str(anchors/(arm+'.npz')),anchor_export_sha256=sha(anchors/(arm+'.npz')),tensor_verification=str(ROOT/f'outputs/native_selection_collection12_20261011_v2/seed2_{arm}/tensor_verification_v4.json')))
    assert {j['arm'] for j in result}=={'original_stem','native_stem_scratch','highfreq_rgb'} and all(j['seed']==2 for j in result)
    return result
def run(arm):
    spec,a,p,helper,parts,aggregate=prepared();stage_p,geometry,onset,rank,readers=parts
    base=helper.cached_baseline(2,stage_p,readers['full']);job=next(j for j in jobs() if j['arm']==arm)
    tensor=read(job['tensor_verification']);assert tensor['status']=='PASS_ACTUAL548_SELECTION_TENSORS_AND_FORMAL_NATIVE_ENDPOINT_STAGE_JOIN' and tensor['all548_observed_anchors_exact_saved_EMA_buffer']
    identity=a.binding(job,p,readers[job['kind']],base);assert identity==tensor['identity']
    output=ROOT/f'outputs/native_selection_seed2_analysis12_20261011_v1/{arm}';output.mkdir(parents=True,exist_ok=False)
    values,stages=helper.inspect(job,stage_p,geometry,onset,rank,readers,base,output)
    assert stages['native_TP_proxy_conflicts']==0 and not stages['score_mismatch_images']
    joined,description=a.describe_selection(job,values,base['annotation'],geometry,aggregate,output)
    report=dict(status='SEED2_SINGLE_ARM_ACTUAL_SELECTOR_AND_SAVED_GEOMETRY_DESCRIPTION',seed=2,arm=arm,identity=identity,job=job,stages=stages,selection=description,reader_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),all548_images=True,all38759_GT=True,two_seed_scientific_analysis_complete=False,causal_root_proven=False,novelty_proven=False,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,scope=spec['scope'])
    write(output/'report.json',report);write(output/'manifest.json',dict(files={x.relative_to(output).as_posix():sha(x) for x in output.rglob('*') if x.is_file()}))
    print(json.dumps(dict(status=report['status'],arm=arm,small=description['groups']['COCO_small']['all'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('original_stem','native_stem_scratch','highfreq_rgb'),required=True);run(p.parse_args().arm)
