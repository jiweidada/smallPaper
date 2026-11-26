"""Reuse common-site calculation with all six actual validated records."""
from pathlib import Path
import ast,json,hashlib
R=Path(__file__).resolve().parent.parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
old=R/'英文小论文/code/native_selection_seed2_common_location12_20261011_v2/analyze.py'
new=R/'英文小论文/code/native_selection_common_location12_20261011_v3';new.mkdir(exist_ok=False)
s=old.read_text(encoding='utf8');s=s.replace('import hashlib,importlib.util,json,sys','import argparse,hashlib,importlib.util,json,sys').replace('def main():','def main(seed):')
s=s.replace('native_selection_analysis12_20261011_v4','native_selection_analysis12_20261011_v5').replace('cached_baseline(2,','cached_baseline(seed,')
start=s.index('    for arm in arms:\n');end=s.index('    groups=helper.groups',start)
replacement='''    allroot=ROOT/'outputs/native_selection_analysis12_20261011_v5/actual_six'
    complete=read(allroot/'report.json');assert complete['status']=='ANALYZED_SIX_FIXED12_ACTUAL_TOPK_AND_SAVED_GEOMETRY_DESCRIPTIONS'
    assert complete['seeds']==[1,2] and set(complete['arms'])==set(arms)
    assert complete['protocol_sha256']==sha(ROOT/'英文小论文/code/native_selection_analysis12_20261011_v5/protocol.json')
    m=read(allroot/'manifest.json');assert all(sha(allroot/n)==h for n,h in m['files'].items())
    jobs=read(ROOT/'outputs/native_selection_analysis12_20261011_v5/six_actual_jobs.json')['jobs'];selection.validate_job_set(jobs)
    for arm in arms:
        job=next(j for j in jobs if j['seed']==seed and j['arm']==arm)
        label=arm+'_seed'+str(seed);root=allroot/label
        assert complete['descriptions'][label]['stages']['report_sha256']==job['report_sha256']==sha(job['report'])
        assert complete['descriptions'][label]['selection']['diagnostic_verification_sha256']==job['verification_sha256']
        values[arm]=npz(root/'per_GT.npz');descriptions[arm]=npz(root/'selection_per_GT.npz');roots[arm]=Path(job['diagnostic'])
        for key in ('annotation_id','category_id','image_id','original_area'):assert np.array_equal(values[arm][key],base['values'][key])
'''
s=s[:start]+replacement+s[end:]
s=s.replace("output=ROOT/'outputs/native_selection_seed2_common_location12_20261011_v2'","output=ROOT/f'outputs/native_selection_common_location12_20261011_v3/seed{seed}'")
s=s.replace("status='SEED2_SHARED_PHYSICAL_LOCATION_INITIAL_REFERENCE_AND_NATIVE_TP_PARTITIONS',seed=2", "status='SIX_RECORD_MEMBER_SHARED_PHYSICAL_LOCATION_INITIAL_REFERENCE_AND_NATIVE_TP_PARTITIONS',seed=seed")
s=s.replace("if __name__=='__main__':main()", "if __name__=='__main__':\n    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(1,2),required=True);args=parser.parse_args();main(args.seed)")
ast.parse(s);(new/'analyze.py').write_text(s,encoding='utf8')
actual=R/'outputs/native_selection_analysis12_20261011_v5/actual_six';assert (actual/'manifest.json').exists()
deps=['英文小论文/code/native_selection_analysis12_20261011_v5/'+n for n in ['analyze.py','selection_core.py','protocol.json']]
deps+=['outputs/native_selection_analysis12_20261011_v5/'+n for n in ['six_actual_jobs.json','actual_six/report.json','actual_six/manifest.json']]
protocol=dict(reader_sha256=sha(new/'analyze.py'),dependencies_sha256={n:sha(R/n) for n in deps},calculation_source_sha256=sha(old),scope='Read all six verified formal and actual native records. Join physical feature locations, not query identities. No new model/evaluator/train calls; same calculations as seed2v2.')
(new/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
out=R/'outputs/native_selection_common_location12_20261011_v3';out.mkdir(exist_ok=False)
(out/'source_review.json').write_text(json.dumps(dict(status='ALL_SIX_ACTUAL_COMPLETE_BINDINGS_PINNED_FOR_PHYSICAL_LOCATION_ANALYSIS',reader_sha256=protocol['reader_sha256'],protocol_sha256=sha(new/'protocol.json'),old_source_unchanged=sha(old)==protocol['calculation_source_sha256'],new_model_forwards=0),indent=2),encoding='utf8')
print('ALL_SIX_ACTUAL_COMPLETE_BINDINGS_PINNED_FOR_PHYSICAL_LOCATION_ANALYSIS')
