"""Require exact local graphs/statistics/CSV rows versus remote SciPy references."""
from pathlib import Path
import csv,hashlib,itertools,json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/encoder_pool_diagnostic_20261010_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def compare_csv(local,remote,seed):
    with local.open(encoding='utf-8-sig',newline='') as lf,remote.open(encoding='utf-8-sig',newline='') as rf:
        lr=csv.DictReader(lf);rr=csv.DictReader(rf);assert lr.fieldnames==rr.fieldnames
        selected=(r for r in lr if int(r['seed'])==seed);count=0
        for left,right in itertools.zip_longest(selected,rr):
            assert left==right,(seed,local.name,count,left,right)
            count+=1
    return count

def main():
    local=OUT/'local_analysis';summary=read(local/'summary.json')
    cert=read(local/'matching_certificates.json')['certificates']
    arrays=read(local/'array_verification.json')['batch_checks'];checks=[]
    assert summary['diagnostic_priority_gate_both_seed_ge10pp'] is False
    assert len(cert)==3840 and len(arrays)==80
    for seed in (1,2):
        root=OUT/f'analysis_reference_seed{seed}/analysis_scipy_reference'
        reference=read(root/'summary.json');rcert=read(root/'matching_certificates.json')['certificates']
        lcert=[r for r in cert if r['seed']==seed]
        assert len(lcert)==1920 and lcert==rcert
        for field in ['primary','strata']:
            assert [r for r in summary[field] if r['seed']==seed]==reference[field]
        assert [r for r in arrays if r['seed']==seed]==read(root/'array_verification.json')['batch_checks']
        for field in ['status','analysis_code_sha256','protocol_sha256','single_seed_gate_ge10pp','geometry_oracle_is_AP','root_cause_established','epochs_added','optimizer_steps']:
            assert summary[field]==reference[field],field
        counts={}
        for name in ['per_GT_pool_coverage.csv','per_image_unique_coverage.csv']:
            counts[name]=compare_csv(local/name,root/name,seed)
        assert counts['per_GT_pool_coverage.csv']==reference['GT_exposures']
        assert counts['per_image_unique_coverage.csv']==reference['image_exposures']==160
        assert read(OUT/f'download_verification_seed{seed}.json')['all_capture_files_verified']==165
        checks.append({'seed':seed,'exact_adjacency_graph_hashes':len(lcert),'independent_SciPy_maximum_cardinality_checks':2*len(lcert),'CSV_rows_exact':counts,'primary_and_all_strata_exact':True,'all_capture_files_verified':165})
    result={'status':'PASS_LOCAL_FULL_ARRAY_RECOMPUTATION_ALL_GRAPHS_OPTIMALITY_AND_CSV_ROWS','scientific_scope':'ANALYZED','engineering_array_source_selection_graph_parity':'VERIFIED','checks':checks,'local_files':{p.name:sha(p) for p in sorted(local.iterdir()) if p.is_file()},'limits':['Local SciPy unavailable. Local graph bytes and witness cardinalities exactly match remote independently computed SciPy optima, for both all GT and tiny-only graphs.','Verification-only graph instrumentation was added after remote statistics; original matcher, primary and gate unchanged.','No AP estimate or detector training experiment.']}
    (OUT/'local_analysis_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'graphs':3840,'independent_cardinality_checks':7680,'GT_exposures':summary['GT_exposures'],'primary':summary['primary'],'gate':summary['diagnostic_priority_gate_both_seed_ge10pp']}))

if __name__=='__main__':main()
