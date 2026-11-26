"""Local descriptive tables from saved diagnostics; no torch/model/inference."""
import csv,json,pathlib,statistics
import networkx as nx
ROOT=pathlib.Path(__file__).resolve().parents[2]/'data/08_匹配聚合覆盖探针_20261019_v1'
RULES=['native_union','coverage_constrained','last_anchor_fill']
def mean(x):return statistics.mean(x) if x else None
def ratio(a,b):return a/b if b else None

def main():
    manifest=json.loads((ROOT/'shared_inputs/inputs/manifest.json').read_text(encoding='utf8'))
    bm={b['batch']:b for b in manifest['batches']};g=nx.Graph();g.add_nodes_from(bm)
    for i in bm:
        for j in bm:
            if j<i and set(bm[i]['source_ids'])&set(bm[j]['source_ids']):g.add_edge(i,j)
    comps=[sorted(c) for c in nx.connected_components(g)]
    result={'scope':'frozen64-input descriptive diagnostics only','models':{},'statistical_unit':'batch/shared-source connected component; not GT or decoder layer'};rows=[]
    for account,seed in [('A',1),('B',2)]:
        folder=ROOT/account/f'results_seed{seed}'
        completion=json.loads((folder/'supplement_complete.json').read_text());assert completion['state']=='complete'
        data={i:json.loads((folder/f'batch{i:02d}_analysis.json').read_text()) for i in bm}
        groups={}
        for group in ['development','heldout']:
            ids=[i for i in bm if bm[i]['group']==group];cases=[r for i in ids for r in data[i]['inputs']]
            eligible=[r for r in cases if r['capacity_eligible']];over=[r for r in cases if r['overflow']]
            capacity_GT=sum(r['GT'] for r in eligible);missing=sum(len(r['native_missing_GT']) for r in eligible)
            units=[]
            for comp in [c for c in comps if bm[c[0]]['group']==group]:
                unitcases=[r for i in comp for r in data[i]['inputs']];n=sum(r['GT'] for r in unitcases if r['capacity_eligible'])
                unit={'batches':comp,'inputs':len(unitcases),'eligible_missing_GT':sum(len(r['native_missing_GT']) for r in unitcases if r['capacity_eligible']),
                      'eligible_GT':n,'overflow_inputs':sum(r['overflow'] for r in unitcases),'rules':{}}
                unit['eligible_missing_rate']=ratio(unit['eligible_missing_GT'],n)
                for rule in RULES:
                    branchdata=[v[rule] for r in unitcases for v in r['branches'].values()]
                    affected=[v[rule] for r in unitcases if r['capacity_eligible'] and r['native_missing_GT'] for v in r['branches'].values()]
                    pairs=sum(v['pairs'] for v in branchdata)
                    unit['rules'][rule]={
                        'mean_image_layer_pair_IoU':mean([v['mean_iou'] for v in branchdata if v['mean_iou'] is not None]),
                        'mean_image_layer_q25_IoU':mean([v['q25_iou'] for v in branchdata if v['q25_iou'] is not None]),
                        'diagnostic_omitted_images_mean_pair_IoU':mean([v['mean_iou'] for v in affected if v['mean_iou'] is not None]),
                        'diagnostic_omitted_images_mean_q25_IoU':mean([v['q25_iou'] for v in affected if v['q25_iou'] is not None]),
                        'weighted_L1':mean([data[i]['batch_losses'][rule]['ordinary_direct_components']['bbox'] for i in comp]),
                        'weighted_GIoU':mean([data[i]['batch_losses'][rule]['ordinary_direct_components']['giou'] for i in comp]),
                        'weighted_FGL':mean([data[i]['batch_losses'][rule]['ordinary_direct_components']['fgl'] for i in comp]),
                        'changed_query_count':sum(r['changed_queries'][rule] for r in unitcases),
                        'positive_classification_regression_identity_conflicts':sum(v.get('own_positive_identity_disagreements',0) for v in branchdata),
                        'positive_identity_conflict_rate':ratio(sum(v.get('own_positive_identity_disagreements',0) for v in branchdata),sum(v.get('own_positive_queries',0) for v in branchdata)),
                        'regression_queries_classified_background':sum(v.get('classification_background_queries',0) for v in branchdata),
                        'background_among_regression_pairs_rate':ratio(sum(v.get('classification_background_queries',0) for v in branchdata),pairs)}
                    rows.append({'account':account,'seed':seed,'group':group,'source_unit':','.join(map(str,comp)),'rule':rule,
                                 'eligible_missing_rate':unit['eligible_missing_rate'],**unit['rules'][rule]})
                dnrows=[]
                for i in comp:
                    dn=data[i]['DN_geometry'];assert dn['present'],'Real train-output probe requires DN diagnostics'
                    for name,branch in dn['branches'].items():
                        for image in branch['images']:
                            for gt in image['GT']:
                                if gt['omitted_by_U0']:dnrows.append({'branch':name,'batch':i,**gt})
                unit['omitted_GT_DN']={
                    'GT_DN_layer_appearances':len(dnrows),'zero_positive_slot_appearances':sum(v['positive_DN_slots']==0 for v in dnrows),
                    'mean_GT_DN_layer_IoU':mean([v['mean_iou'] for v in dnrows if v['mean_iou'] is not None]),
                    'mean_GT_DN_layer_q25_IoU':mean([v['q25_iou'] for v in dnrows if v['q25_iou'] is not None]),
                    'weighted_batch_contribution_per_batch':{term:sum(v['weighted_batch_contribution'].get(term,0) for v in dnrows)/len(comp) for term in ['bbox','giou','fgl']},
                    'raw_GT_DN_layer_mean':{term:mean([v['raw_pair_mean'][term] for v in dnrows if term in v['raw_pair_mean'] and v['raw_pair_mean'][term] is not None]) for term in ['bbox','giou','fgl']}}
                unit['omitted_GT_own_layer_IoU']={name:mean([v['IoU'] for r in unitcases for v in r['missing_GT_own_layer_IoU'][name] if v['IoU'] is not None]) for name in data[comp[0]]['branch_names']}
                units.append(unit)
            groups[group]={'inputs':len(cases),'batches':len(ids),'source_units':units,'capacity_eligible_GT':capacity_GT,
                'missing_eligible_GT':missing,'missing_eligible_GT_rate':ratio(missing,capacity_GT),
                'source_equal_missing_eligible_GT_rate':mean([u['eligible_missing_rate'] for u in units if u['eligible_missing_rate'] is not None]),
                'overflow_inputs':len(over),'overflow_input_fraction':ratio(len(over),len(cases)),
                'overflow_batches':sum(any(r['overflow'] for r in data[i]['inputs']) for i in ids),
                'overflow_batch_fraction':ratio(sum(any(r['overflow'] for r in data[i]['inputs']) for i in ids),len(ids)),
                'overflow_GT':sum(r['GT'] for r in over),'overflow_native_missing_GT':sum(len(r['native_missing_GT']) for r in over),
                'overflow_missing_exceeding_K_capacity':sum(r['excess_missing_beyond_K_capacity'] for r in over),
                'top_frequency_tied_queries':sum(r['tie_sensitivity']['top_vote_tied_queries'] for r in cases),
                'top_frequency_tied_query_fraction':ratio(sum(r['tie_sensitivity']['top_vote_tied_queries'] for r in cases),sum(r['K'] for r in cases)),
                'native_differs_deterministic_min_GT_queries':sum(r['tie_sensitivity']['native_differs_min_GT_queries'] for r in cases),
                'deterministic_min_GT_eligible_missing_GT':sum(len(r['tie_sensitivity']['min_GT_union_missing_GT']) for r in eligible),
                'candidate_changed_queries':sum(r['changed_queries']['coverage_constrained'] for r in eligible),
                'candidate_differs_anchor_queries':sum(sum(g!=dict(zip(r['branches']['final']['last_anchor_fill']['pair_query'],r['branches']['final']['last_anchor_fill']['pair_GT']))[q] for q,g in zip(r['branches']['final']['coverage_constrained']['pair_query'],r['branches']['final']['coverage_constrained']['pair_GT'])) for r in eligible)}
        checks=[c for d in data.values() for br in d['DN_geometry']['branches'].values() for c in br['full_native_sum_checks'].values()]
        result['models'][account]={'seed':seed,'groups':groups,'all_DN_sum_checks_passed':all(c['abs_error']<=1e-5*max(1,abs(c['native_total'])) for c in checks),'DN_sum_max_abs_error':max(c['abs_error'] for c in checks)}
    (ROOT/'阶段0补充诊断_DN并列溢出分项.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (ROOT/'阶段0补充诊断_来源组分项.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'saved':str(ROOT/'阶段0补充诊断_DN并列溢出分项.json'),'models':{a:{g:{k:v for k,v in s.items() if k!='source_units'} for g,s in m['groups'].items()} for a,m in result['models'].items()}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
