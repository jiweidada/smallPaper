"""Clarify low-IoU PR losses and condition reference-to-final descriptions on same GT."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
first=ROOT/'outputs/warm_stem_failure_review12_20261011_v1'
for n,h in read(first/'manifest.json')['files'].items():assert sha(first/n)==h,n
r=read(first/'report.json')
tools=module('warm12_clarify_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
p,geometry,_,_,readers=tools.prepared();base=tools.cached_baseline(2,p,readers['full'])
warm=npz(ROOT/'outputs/warm_stem_primary_analysis12_20261011_v1/seed2/warm_stage_per_GT.npz');bv=base['values']
for n in ('annotation_id','image_id','category_id','original_area'):assert np.array_equal(warm[n],bv[n])
mask=warm['original_area']<=1024;paired={}
for t,ti in ((.5,0),(.75,5)):
 bi=bv['max_iou'][:,0]>=t;wi=warm['max_iou'][:,0]>=t
 bf=bv['max_iou'][:,2]>=t;wf=warm['max_iou'][:,2]>=t
 strata={'shared_initial_support':mask&bi&wi,'warm_new_initial_support':mask&~bi&wi,'warm_lost_initial_support':mask&bi&~wi}
 rows={}
 for name,m in strata.items():
  final=np.zeros((2,2),dtype=np.int64);TP=np.zeros((2,2),dtype=np.int64);query=np.zeros((2,2),dtype=np.int64)
  bq=bv['initial_best_at_final']>=t;wq=warm['initial_best_at_final']>=t
  np.add.at(final,(bf[m].astype(int),wf[m].astype(int)),1)
  np.add.at(TP,(bv['native_TP'][m,ti].astype(int),warm['native_TP'][m,ti].astype(int)),1)
  np.add.at(query,(bq[m].astype(int),wq[m].astype(int)),1)
  assert final.sum()==TP.sum()==query.sum()==int(m.sum())
  rows[name]=dict(GT=int(m.sum()),BASE_rows_warm_columns_bool0_1_final_any_query_support=final.tolist(),
   BASE_rows_warm_columns_bool0_1_native_all_area_TP=TP.tolist(),
   separately_chosen_initial_best_queries_final_support=query.tolist(),
   baseline_final_already_supported=int((m&bf).sum()),warm_final_supported=int((m&wf).sum()),
   initial_best_at_final_query_comparison_is_not_cross_run_same_query_identity=True)
 assert rows['warm_new_initial_support']['GT']-rows['warm_lost_initial_support']['GT']==r['existing_same_GT_outcomes'][str(t)]['COCO_small']['supported_GT_by_boundary']['encoder_initial_boxes']['delta']
 paired[str(t)]=rows
d=r['APs_and_area_specific_TP_delta_by_IoU']
low=sum(x['APs_delta_points'] for x in d[:5])/10;high=sum(x['APs_delta_points'] for x in d[5:])/10
assert abs(low+high-r['delta_points']['APs'])<1e-10
assert d[0]['small_area_specific_TP_delta']==51 and d[0]['APs_delta_points']<0
assert d[5]['APs_delta_points']>0 and d[5]['small_area_specific_TP_delta']==-2
v=read(ROOT/'outputs/warm_stem_primary_collection12_20261011_v1/seed2_warm_stem_v1/primary_verification_v1.json')
packet=Path(v['downloaded_packet']);bp=npz(base['packet']/'baseline_eval/endpoint_epoch12/precision_recall.npz')['precision']
wp=npz(packet/'run/endpoint_epoch12/precision_recall.npz')['precision']
class50=[]
for c in r['class_APs']:
 k=c['class_id'];a=bp[0,:,k,1,2];b=wp[0,:,k,1,2]
 assert (a>=0).all() and (b>=0).all()
 diff=(b-a)*100
 class50.append(dict(class_id=k,name=c['name'],APs50_BASE=float(a.mean()*100),APs50_warm=float(b.mean()*100),
  APs50_delta_points=float(diff.mean()),APs50_macro_contribution=float(diff.mean()/10),
  small_area_specific_TP_delta=c['small_area_specific_TP_delta_by_IoU'][0],
  native_interpolated_precision_recall_cells_lower=int((diff<-1e-12).sum()),cells_higher=int((diff>1e-12).sum()),
  BASE_native_interpolated_precision=a.tolist(),warm_native_interpolated_precision=b.tolist()))
assert abs(sum(c['APs50_macro_contribution'] for c in class50)-d[0]['APs_delta_points'])<1e-9
dest=ROOT/'outputs/warm_stem_failure_review12_20261011_v2';dest.mkdir(exist_ok=False)
result=dict(status='WARM_SEED2_LOW_IOU_NATIVE_PR_LOSS_CLARIFIED_WITH_SAME_GT_SUPPORT_STRATA',seed=2,
 supersedes_interpretation_only='../warm_stem_failure_review12_20261011_v1/report.json',
 BASE12=r['BASE12'],warm12=r['warm12'],delta_points=r['delta_points'],native_gate_passed=False,
 exploratory_IoU_range_contributions_to_total_APs_points={'0.50_to0.70':low,'0.75_to0.95':high},
 native_small_area_AP50_delta=d[0]['APs_delta_points'],native_small_area_AP75_delta=d[5]['APs_delta_points'],
 native_small_area_TP50_delta=51,native_small_area_TP75_delta=-2,class_APs50=class50,same_GT_initial_support_strata=paired,
 correction='The overall APs decrease is concentrated in the observed0.50–0.70 precision-recall metrics; the0.75–0.95 segment contributes positively. Thus fewer supported boxes at0.75 does not establish worse AP75, and positive TP counts do not establish better AP50. Neither overall failure nor its cause can be reduced to reference coverage or final localization counts.',
 changed_next_action='Prioritize class-conditioned precision-recall and final true-class selection evidence alongside original-user fixed12 controls. Treat decoder geometry propagation as a descriptive secondary question, not the established cause of the APs failure. Await the existing native-location diagnostic for initial selection versus reference regression.',
 boundaries='Exploratory post-hoc single-seed description; no significance, causal mediation or innovation claim. The same GT can be joined across runs; queries are followed only within runs and are separately selected. Native small-area PR arrays are exact; raw small-area dtMatches/dtIgnore score-prefix records were not saved, so an exact small-area score-prefix reconstruction or unique classification-versus-ordering cause is not claimed.',
 source_sha256=sha(__file__),evidence=dict(prior_review_manifest_sha256=sha(first/'manifest.json'),
  primary_verification_sha256=sha(ROOT/'outputs/warm_stem_primary_collection12_20261011_v1/seed2_warm_stem_v1/primary_verification_v1.json'),
  analysis_source_sha256=r['source_sha256'],baseline_stage_cache_manifest_sha256=base['cache_stage_manifest_sha256']),
 independent_saved_EMA_replay_verified=False,other_seed_endpoint_pending=True,root_proven=False,novelty_proven=False,
 new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,goal_complete=False)
write(dest/'report.json',result)
text=['Stem种子2失败分析：纠正只看覆盖数量的解释','',
 '固定12轮：AP %.4f（基线%.4f）；APs %.4f（基线%.4f）。'%(r['warm12']['AP'],r['BASE12']['AP'],r['warm12']['APs'],r['BASE12']['APs']),
 'IoU0.50：原生小面积TP净增51，但小面积AP下降0.5391个百分点。',
 'IoU0.75：原生小面积TP减少2，小面积AP反而提高0.0435个百分点。',
 '整体APs差值%.5f中，IoU0.50–0.70贡献%.5f，IoU0.75–0.95贡献%+.5f。这里是保存的原生PR数组的等权算术，并非因果分解。'%(r['delta_points']['APs'],low,high),'',
 '因此，最终高IoU几何覆盖少13个不能解释为高IoU AP必然更差；早期覆盖多347个也不能保证最终AP。问题还涉及类别之间的精度—召回差异。',
 '前一版逐目标报告保留；其几何和TP数字仍正确，本版进一步明确整体APs下降主要来自低IoU段的原生PR曲线，不把定位传播认定为失败根因。','',
 '按同一GT分组，区别“之前最终框已经覆盖”和“新增加最终覆盖”：']
for t,ss in paired.items():
 text.append('IoU '+t)
 for name,row in ss.items():
  text.append(name+': '+json.dumps({k:x for k,x in row.items() if not k.startswith('separately_') and not k.startswith('initial_best_at')},ensure_ascii=False))
text.extend(['','各类IoU0.50的小面积AP变化：'])
for c in class50:text.append('%s：AP差值%+.5f，原生小面积TP差值 %+d，101个原生插值PR点中下降%d点。'%(c['name'],c['APs50_delta_points'],c['small_area_specific_TP_delta'],c['native_interpolated_precision_recall_cells_lower']))
text.extend(['','下一步：继续完成原始P2、SiLU、注意力双种子固定12轮，重点检查类别条件的PR曲线和最终真实类别保留。选中参考框之前的具体筛选/回归问题仍待既有诊断队列。',
 '这条Warm配置不延长、不重跑。当前只有种子2完成该分析，另一种子与独立EMA核验仍待完成。',
 '目前缺少原生小面积逐检测分数前缀的匹配/忽略记录；只有原生小面积PR数组和全面积匹配记录，不能声称精确分数前缀解释了APs或证明某个分类头是根因。'])
(dest/'Stem失败分析与解释修正.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},source_sha256=sha(__file__)))
assert 'torch' not in sys.modules
print(json.dumps(dict(status=result['status'],IoU_contributions=result['exploratory_IoU_range_contributions_to_total_APs_points'],support_strata=paired),ensure_ascii=False))
