"""Summarize only an explicitly selected physical partition; standard library only."""
import argparse,csv,json,pathlib,statistics
from stage0_thresholds import decide
from seal_v3 import sha,verify_files,write_json
RULES=['native_union','coverage_constrained','last_anchor_fill']
def avg(v):return statistics.mean(v) if v else None
def mapping(r,rule):
 v=r['branches']['final'][rule];return dict(zip(v.get('pair_query',[]),v.get('pair_GT',[])))
def components(items):
 pending=set(items);out=[]
 while pending:
  comp={pending.pop()};changed=True
  while changed:
   changed=False
   for j in list(pending):
    if any(set(items[j]['source_ids'])&set(items[i]['source_ids']) for i in comp):
     comp.add(j);pending.remove(j);changed=True
  out.append(sorted(comp))
 return out
def summarize(folder):
 complete=json.loads((folder/'complete.json').read_text());verify_files(folder,complete)
 assert complete['state']=='complete' and complete['formal_inputs']==32
 entries={r['batch']:r for r in complete['files']};data={i:json.loads((folder/r['relative']).read_text()) for i,r in entries.items()}
 assert all(a['group']==complete['group'] for a in data.values())
 cases=[r for a in data.values() for r in a['inputs']];eligible=[r for r in cases if r['capacity_eligible']]
 counts={'M':sum(len(r['native_missing_GT']) for r in eligible),'T':sum(r['GT'] for r in eligible),
         'D':sum(sum(v!=mapping(r,'last_anchor_fill')[q] for q,v in mapping(r,'coverage_constrained').items()) for r in eligible),
         'B':sum(r['K'] for r in eligible)}
 units=[]
 for comp in components(entries):
  unit={'batches':comp,'quality':{},'direct_components':{}}
  for rule in RULES:
   image_means=[];lower=[];loss=[];terms={k:[] for k in ['bbox','giou','fgl']}
   for i in comp:
    image_means.append(avg([avg([v[rule]['mean_iou'] for v in r['branches'].values() if v[rule]['mean_iou'] is not None]) for r in data[i]['inputs'] if any(v[rule]['mean_iou'] is not None for v in r['branches'].values())]))
    lower.extend(v[rule]['q25_iou'] for r in data[i]['inputs'] for v in r['branches'].values() if v[rule]['q25_iou'] is not None)
    part=data[i]['batch_losses'][rule];loss.append(sum(part['losses'][k] for k in part['ordinary_direct_loss_keys']))
    for k in terms:terms[k].append(part['ordinary_direct_components'][k])
   unit['quality'][rule]={'mean_pair_IoU':avg([v for v in image_means if v is not None]),'mean_image_layer_q25_IoU':avg(lower),'weighted_direct_loss':avg(loss)}
   unit['direct_components'][rule]={k:avg(v) for k,v in terms.items()}
  units.append(unit)
 quality={rule:{key:avg([u['quality'][rule][key] for u in units if u['quality'][rule][key] is not None]) for key in ['mean_pair_IoU','mean_image_layer_q25_IoU','weighted_direct_loss']} for rule in RULES}
 conflicts={rule:{'positive_identity_conflicts':sum(v[rule].get('own_positive_identity_disagreements',0) for r in cases for v in r['branches'].values()),'regression_pairs_classified_background':sum(v[rule].get('classification_background_queries',0) for r in cases for v in r['branches'].values()),'changed_query_count':sum(r['changed_queries'][rule] for r in cases)} for rule in RULES}
 dn=[];dn_error=[]
 for a in data.values():
  assert a['DN_geometry']['present']
  for branch in a['DN_geometry']['branches'].values():
   for c in branch['full_native_sum_checks'].values():
    assert c['abs_error']<=1e-5*max(1,abs(c['native_total']));dn_error.append(c['abs_error'])
   dn.extend(g for image in branch['images'] for g in image['GT'] if g['omitted_by_U0'])
 for a in data.values():
  for part in a['batch_losses'].values():assert part['DDF_value_identical'] and part['DDF_gradient_max_abs_difference']==0
 assert any(part['gradient_checked'] for a in data.values() for part in a['batch_losses'].values())
 return {'seed':complete['seed'],'group':complete['group'],'inputs':32,'eligible_inputs':len(eligible),'counts':counts,
   'missing_rate':counts['M']/counts['T'] if counts['T'] else None,'mapping_difference_rate':counts['D']/counts['B'] if counts['B'] else None,
   'source_units':units,'quality':quality,'conflicts':conflicts,
   'direct_components':{rule:{term:avg([u['direct_components'][rule][term] for u in units]) for term in ['bbox','giou','fgl']} for rule in RULES},
   'overflow_inputs':sum(r['overflow'] for r in cases),'overflow_batches':sum(any(r['overflow'] for r in a['inputs']) for a in data.values()),
   'overflow_missing_GT':sum(len(r['native_missing_GT']) for r in cases if r['overflow']),
   'overflow_extra_missing_beyond_K':sum(r['excess_missing_beyond_K_capacity'] for r in cases if r['overflow']),
   'top_vote_tied_queries':sum(r['tie_sensitivity']['top_vote_tied_queries'] for r in cases),
   'native_differs_deterministic_min_GT_queries':sum(r['tie_sensitivity']['native_differs_min_GT_queries'] for r in cases),
   'deterministic_min_GT_eligible_missing_GT':sum(len(r['tie_sensitivity']['min_GT_union_missing_GT']) for r in eligible),
   'omitted_GT_own_layer_IoU':{name:avg([v['IoU'] for r in cases for v in r['missing_GT_own_layer_IoU'][name] if v['IoU'] is not None]) for name in next(iter(data.values()))['branch_names']},
   'omitted_GT_DN':{'GT_layer_appearances':len(dn),'zero_positive_slots':sum(g['positive_DN_slots']==0 for g in dn),'mean_GT_layer_IoU':avg([g['mean_iou'] for g in dn if g['mean_iou'] is not None]),'mean_GT_layer_q25_IoU':avg([g['q25_iou'] for g in dn if g['q25_iou'] is not None]),'weighted_batch_contribution_per_batch':{term:sum(g['weighted_batch_contribution'].get(term,0) for g in dn)/8 for term in ['bbox','giou','fgl']}},
   'DN_max_sum_abs_error':max(dn_error,default=0),'DDF_checks_passed':True,'source_partition_manifest_sha256':sha(folder/'complete.json')}
def main(base,group):
 if group=='heldout':
  release=json.loads((base/'heldout_access_receipt.json').read_text())
  assert sha(base/'local_frozen_analysis.json')==release['local_freeze_sha256']
  freeze=json.loads((base/'local_frozen_analysis.json').read_text());assert freeze['development_dry_run_passed']
  for row in freeze['local_code_files']:assert sha(pathlib.Path(row['absolute_path']))==row['sha256']
 models={seed:summarize(base/a/group) for a,seed in [('A',1),('B',2)]}
 assert [u['batches'] for u in models[1]['source_units']]==[u['batches'] for u in models[2]['source_units']]
 threshold=decide({s:m['counts'] for s,m in models.items()});reasons=list(threshold['stop_reasons']);eps=1e-6
 valid=all(v['mean_pair_IoU'] is not None and v['weighted_direct_loss'] is not None for m in models.values() for v in m['quality'].values())
 joint_worse=valid and all(m['quality']['coverage_constrained']['mean_pair_IoU']<m['quality']['native_union']['mean_pair_IoU']-eps and m['quality']['coverage_constrained']['weighted_direct_loss']>m['quality']['native_union']['weighted_direct_loss']+eps for m in models.values())
 anchor_dominates=valid and all(m['quality']['last_anchor_fill']['mean_pair_IoU']>=m['quality']['coverage_constrained']['mean_pair_IoU']-eps and m['quality']['last_anchor_fill']['weighted_direct_loss']<=m['quality']['coverage_constrained']['weighted_direct_loss']+eps and (m['quality']['last_anchor_fill']['mean_pair_IoU']>m['quality']['coverage_constrained']['mean_pair_IoU']+eps or m['quality']['last_anchor_fill']['weighted_direct_loss']<m['quality']['coverage_constrained']['weighted_direct_loss']-eps) for m in models.values())
 if joint_worse:reasons.append('both_models_lower_IoU_and_higher_direct_loss_than_U0')
 if anchor_dominates:reasons.append('simple_anchor_dominates_in_both_models')
 report={'group':group,'models':models,'thresholds':threshold,'stop_reasons':reasons,'stop':bool(reasons),
         'joint_worse':joint_worse,'anchor_dominates':anchor_dominates,'epsilon':eps,
         'status':'development_dry_run_only' if group=='development' else 'stop_current_candidate' if reasons else threshold['status'],
         'automatic_training':False,'no_AP_gain_claim':True}
 dest=base/(group+'_summary.json');write_json(dest,report)
 with (base/(group+'_整数分子分母.csv')).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['seed','M','T','200M_less_T','D','B','100D_less_B','missing_rate','mapping_difference_rate']);w.writeheader()
  for s,m in models.items():
   c=m['counts'];w.writerow({'seed':s,**c,'200M_less_T':200*c['M']<c['T'] if c['T'] else 'NA','100D_less_B':100*c['D']<c['B'] if c['B'] else 'NA','missing_rate':m['missing_rate'],'mapping_difference_rate':m['mapping_difference_rate']})
 print(json.dumps({'saved':str(dest),'group':group,'counts':{s:m['counts'] for s,m in models.items()},'stop_reasons':reasons,'interpretation':report['status']},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--group',choices=['development','heldout'],required=True);a=p.parse_args();main(a.base,a.group)
