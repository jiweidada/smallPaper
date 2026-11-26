"""Post-primary, explicitly exploratory clip decomposition from cached heads only."""
import datetime
from common import *

def main():
 resource_limits()
 save_json(OUT/'04_裁剪分解补充口径.json',{
  'written_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'reason':'Primary completed diagnostic shows large changes in original global clipping factors. Separate the last-layer DN gradient term from global clipping coupling before causal interpretation.',
  'status':'exploratory supplementary decomposition, specified after primary results, does not replace them',
  'scope':'all four preselected cross-batch cases and both existing groups; cached tensors only, zero new model forwards/backwards',
  'definitions':{'direct_given_original_clip':'L(include gradient, original include clip) - L(exclude gradient, same original include clip)',
    'clip_coupling':'L(exclude gradient, original include clip) - L(exclude gradient, recomputed exclude clip)',
    'identity':'primary total effect = direct_given_original_clip + clip_coupling'},
  'same_batch_cosine':'descriptive original ordinary vs DN head gradients on the generator; not a detector benefit test',
  'script_sha256':sha(__file__),
 })
 primary=json.loads((OUT/'03_全部局部反事实原始结果.json').read_text(encoding='utf8'))
 recipes=json.loads((OLD/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
 rows=[]
 for seed in [1,2]:
  criterion=build_criterion(recipes[str(seed)])
  caches={b:torch.load(OUT/f'seed{seed}/batch{b:02d}_frozen_head.pt',map_location='cpu',weights_only=False) for b in [0,1]}
  for generator,evaluation in [(0,1),(1,0)]:
   c=caches[generator];v=caches[evaluation];weights=c['weights']
   prior=next(r for r in primary['rows'] if (r['seed'],r['generator_batch'],r['evaluation_batch'])==(seed,generator,evaluation))
   common=adam_virtual(weights,c['gradient_total_without_DN_last'],c['optimizer_bindings'],prior['clip_include_DN'])
   row={'seed':seed,'generator_batch':generator,'evaluation_batch':evaluation,'generator_same_batch_ordinary_dot_DN':dot(c['gradient_ordinary'],c['gradient_DN']),
    'generator_same_batch_ordinary_DN_cosine':dot(c['gradient_ordinary'],c['gradient_DN'])/(norm(c['gradient_ordinary'])*norm(c['gradient_DN'])),
    'DN_gradient_norm':norm(c['gradient_DN']),'ordinary_gradient_norm':norm(c['gradient_ordinary']),
    'clip_include':prior['clip_include_DN'],'clip_exclude':prior['clip_exclude_DN_last'],'groups':{}}
   for hard in [False,True]:
    group='input_anchor_IoU_lt_0_5' if hard else 'all_ordinary';p=prior['groups'][group]
    if p['before']['pairs']==0:row['groups'][group]={'pairs':0};continue
    middle=metrics(v,common,criterion,hard)
    direct=p['include_DN']['loss_contribution']-middle['loss_contribution']
    coupling=middle['loss_contribution']-p['exclude_DN_last']['loss_contribution']
    assert abs(direct+coupling-p['DN_effect'])<1e-14
    row['groups'][group]={'total_primary_effect':p['DN_effect'],'direct_given_original_clip':direct,'clip_coupling':coupling,'exclude_gradient_same_include_clip_loss':middle['loss_contribution'],'include_clip_divided_by_exclude_clip':prior['clip_include_DN']/prior['clip_exclude_DN_last']}
   rows.append(row)
 save_json(OUT/'05_裁剪联动分解.json',{'status':'complete','rows':rows,'new_model_forwards':0,'new_training_steps':0})
 print(json.dumps({'rows':len(rows),'state':'complete'}))

if __name__=='__main__':main()
