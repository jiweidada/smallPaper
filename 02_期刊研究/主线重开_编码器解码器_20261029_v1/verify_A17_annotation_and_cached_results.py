"""Independent primary annotation grouping and cached-pair arithmetic, no model."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A17_新编码器机制与多实例映射审计_v1'
assert not (OUT/'independent_verification.json').exists()
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
manifest=load(OUT/'annotation_input_manifest.json')
source=Path('D:/数据集/visdrone2019/VisDrone2019-DET-train/annotations')
totals=Counter();rational_totals=Counter();rounding=Counter()
for row in manifest:
    raw=(source/row['annotation']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['annotation_sha256']
    data=np.asarray([[float(t.strip()) for t in line.split(',') if t.strip()] for line in raw.decode('utf-8-sig').splitlines()])
    valid=(data[:,4]>0)&(data[:,5]>=1)&(data[:,5]<=10);data=data[valid]
    lower=np.maximum(data[:,:2],0);lower=np.minimum(lower,[row['image_width'],row['image_height']])
    upper=np.maximum(data[:,:2]+data[:,2:4],0);upper=np.minimum(upper,[row['image_width'],row['image_height']])
    use=(upper>lower).all(1);lower=lower[use];upper=upper[use];data=data[use]
    # Preserve the frozen coordinate arithmetic: resize each edge,then subtract.
    lo800=lower*800/[row['image_width'],row['image_height']]
    hi800=upper*800/[row['image_width'],row['image_height']]
    size=np.sqrt(np.prod(hi800-lo800,axis=1))
    centres=(lo800+hi800)*.5
    cells=np.floor(centres/4).astype(np.int64);labels=data[:,5].astype(np.int64)
    allcounts=Counter(map(tuple,cells));samecounts=Counter((int(l),int(x),int(y)) for l,(x,y) in zip(labels,cells))
    duplicates=Counter((int(l),*lo,*hi) for l,lo,hi in zip(labels,lower,upper))
    tiny=size<8
    totals.update(valid_GT=len(data),tiny_GT=int(tiny.sum()),duplicate_extra_GT=sum(n-1 for n in duplicates.values()),
        any_class=sum(allcounts[tuple(c)]>1 for c,t in zip(cells,tiny) if t),
        same_class=sum(samecounts[int(l),int(c[0]),int(c[1])]>1 for l,c,t in zip(labels,cells,tiny) if t))
    # Independent exact rational boundary check; all valid coordinates are integral.
    assert np.array_equal(lower,np.rint(lower)) and np.array_equal(upper,np.rint(upper))
    lo=lower.astype(np.int64);hi=upper.astype(np.int64)
    exact_tiny=(10000*(hi[:,0]-lo[:,0])*(hi[:,1]-lo[:,1])<row['image_width']*row['image_height'])
    exact_cells=np.stack((100*(lo[:,0]+hi[:,0])//row['image_width'],100*(lo[:,1]+hi[:,1])//row['image_height']),axis=1)
    ec=Counter(map(tuple,exact_cells));sc=Counter((int(l),int(x),int(y)) for l,(x,y) in zip(labels,exact_cells))
    rational_totals.update(tiny_GT=int(exact_tiny.sum()),any_class=sum(ec[tuple(c)]>1 for c,t in zip(exact_cells,exact_tiny) if t),
        same_class=sum(sc[int(l),int(c[0]),int(c[1])]>1 for l,c,t in zip(labels,exact_cells,exact_tiny) if t))
    rounding.update(tiny_membership_differences=int(np.count_nonzero(exact_tiny!=tiny)),
                    GT_cell_coordinate_differences=int(np.count_nonzero(np.any(exact_cells!=cells,axis=1))))
summary=load(OUT/'annotation_summary.json')
assert len(manifest)==summary['images']==6471
for key in ['valid_GT','tiny_GT','duplicate_extra_GT']:assert totals[key]==summary['counts'][key],('primary_counts',dict(totals),summary['counts'])
for key in ['any_class','same_class']:assert totals[key]==summary['primary'][key]
assert totals['valid_GT']==343204 and totals['tiny_GT']==52210
assert totals['same_class']==2855 and totals['any_class']==6092
numeric_audit={'status':'CANONICAL_ARITHMETIC_VERIFIED_BOUNDARY_SENSITIVITY_DISCLOSED',
    'frozen_edge_resize_then_subtract_counts':dict(totals),'exact_rational_counts':dict(rational_totals),
    'rounding_boundary_counts':dict(rounding),
    'initial_independent_check':'Scale-width-before-edge-subtraction arithmetic yielded52164tiny,6086any,2853same and failed strict equality. No scientific analysis/model rerun;verifier corrected to frozen edge arithmetic.',
    'same_class_fraction_exact_rational':rational_totals['same_class']/rational_totals['tiny_GT'],
    'resource_gate_unchanged':bool((rational_totals['same_class']/rational_totals['tiny_GT']>=.05)==summary['resource_screen_passed']),
    'interpretation':'Strict8pixel and grid-boundary membership depends on floating-point order. Exact counts are a numerical audit,not a new selected threshold or replacement of frozen results.'}
assert numeric_audit['resource_gate_unchanged']
(OUT/'annotation_numeric_boundary_audit.json').write_text(json.dumps(numeric_audit,indent=2)+'\n',encoding='utf-8')
perimage=load(OUT/'annotation_per_image.json')
for key,expected in summary['grid_counts'].items():
    for field in ['any_class','same_class']:assert sum(r['grid_counts'][key][field] for r in perimage)==expected[field]
protocol=load(OUT/'cached_cell_protocol.json')
for r in protocol['inputs']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
assert hashlib.sha256((ROOT/'a17_cached_cell_analysis.py').read_bytes()).hexdigest()==protocol['code_sha256']
rows=load(OUT/'cached_cell_per_GT.json');pairs=load(OUT/'cached_cell_pairs.json');images=load(OUT/'cached_cell_per_image.json')
groups=defaultdict(list)
for r in rows:groups[r['case']].append(r)
assert len(rows)==853 and len(groups)==37
for case,rr in groups.items():
    with np.load(ROOT/'A16_实际特征空间变化审计_v1'/(('old' if case.startswith('old') else 'new')+'_patches')/('case'+case[-2:]+'.npz'),allow_pickle=False) as z:
        centres=(z['gt_boxes800'].astype(np.float64)[:,:2]+z['gt_boxes800'].astype(np.float64)[:,2:])*.5
        labels=z['gt_labels'];uid=z['gt_uids'];filename=str(z['image_filename'].item())
    rawpath=Path('D:/数据集/visdrone2019/VisDrone2019-DET-val/annotations')/(Path(filename).stem+'.txt')
    raw=rawpath.read_bytes();source_rows=raw.decode('utf-8-sig').splitlines()
    assert hashlib.sha256(raw).hexdigest()==next(x['sha256'] for x in load(OUT/'cached_cell_annotation_inputs.json') if x['file']==rawpath.name)
    cell=np.floor(centres/4).astype(int)
    cc=Counter((int(l),int(c[0]),int(c[1])) for l,c in zip(labels,cell))
    for r in rr:
        g=r['gt'];assert r['same_class_same_cell']==(cc[int(labels[g]),int(cell[g,0]),int(cell[g,1])]>1)
        meta=[float(x) for x in source_rows[int(str(uid[g]).rsplit('_',1)[1])].split(',') if x.strip()]
        assert int(meta[6])==r['occlusion'] and int(meta[7])==r['truncation']
        count=int((((centres-centres[g])**2).sum(1)<=256).sum()-1)
        assert count==r['neighbours_within16'] and int(math.log2(count+1))==r['density_bin']
expected=[]
for case,rr in groups.items():
    for x in rr:
        if not x['same_class_same_cell']:continue
        for y in rr:
            if y['same_class_same_cell']:continue
            if any(x[k]!=y[k] for k in ['label','occlusion','truncation','density_bin']):continue
            if max(x['size'],y['size'])/min(x['size'],y['size'])>1.5:continue
            expected.append((case,x['gt'],y['gt']))
assert set(expected)=={(r['case'],r['exposed_gt'],r['control_gt']) for r in pairs}
assert len(expected)==len(pairs)==69
bygt={(r['case'],r['gt']):r for r in rows};values=[]
for image in images:
    pp=[p for p in pairs if p['case']==image['case']]
    if not pp:continue
    diffs=[int(bygt[p['case'],p['exposed_gt']]['native_iou']<.5)-int(bygt[p['case'],p['control_gt']]['native_iou']<.5) for p in pp]
    values.append(sum(diffs)/len(diffs))
    assert abs(values[-1]-image['effects']['failure50_difference'])<1e-14
a=np.asarray(values);rng=np.random.default_rng(protocol['bootstrap_seed'])
b=np.asarray([a[rng.integers(len(a),size=len(a))].mean() for _ in range(2000)])
stats={'mean':float(a.mean()),'low':float(np.quantile(b,.025)),'high':float(np.quantile(b,.975))}
cached=load(OUT/'cached_cell_summary.json')
for k,v in stats.items():assert abs(v-cached['statistics']['failure50_difference'][k])<1e-12
assert not cached['gate_passed'] and cached['adequate_matched_cohort'] and stats['low']<0
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_PRIMARY_POPULATION_COUNTS_AND_CACHED_PAIR_ARITHMETIC',
    'annotation_files_reparsed_and_hashed':6471,'primary_counts':dict(totals),
    'exact_rational_annotation_counts':dict(rational_totals),'numeric_boundary_audit':'annotation_numeric_boundary_audit.json',
    'secondary_grid_aggregate_cells_checked':len(summary['grid_counts'])*2,
    'cached_GT_checked':853,'cached_pairs_checked':69,'image_clusters':len(a),'primary_statistics':stats,
    'new_complete_model_forwards':0,'new_suffix_forwards':0,'new_fits':0,'mainline_validated':False,
    'scope':'Independent annotation-primary grouping and cached geometry/aggregation verification;not another detector run,causal evidence or equivalence proof.'}
(OUT/'independent_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['status','annotation_files_reparsed_and_hashed','cached_GT_checked','cached_pairs_checked','primary_statistics']}))
