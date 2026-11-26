"""Summarize exact native small-area AP arithmetic and within-run query trajectories."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path.cwd()
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
analysis=ROOT/'outputs/warm_stem_primary_analysis12_20261011_v1/seed2'
for n,h in read(analysis/'manifest.json')['files'].items():assert sha(analysis/n)==h,n
r=read(analysis/'report.json')
assert r['analysis_sha256']==sha(ROOT/'英文小论文/code/warm_stem_primary_analysis12_20261011_v1/analyze_v2.py')
assert r['protocol_sha256']==sha(ROOT/'英文小论文/code/warm_stem_primary_analysis12_20261011_v1/protocol_v2.json')
assert r['native_TP_proxy_conflicts']==0 and not r['score_mismatch_images']
tools=module('warm12_summary_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
p,geometry,_,_,readers=tools.prepared();base=tools.cached_baseline(2,p,readers['full'])
folder=ROOT/'outputs/warm_stem_primary_collection12_20261011_v1/seed2_warm_stem_v1'
v=read(folder/'primary_verification_v1.json');packet=Path(v['downloaded_packet'])
assert v['verification_script_sha256']==sha(ROOT/'英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py')
warm=npz(analysis/'warm_stage_per_GT.npz');bv=base['values']
for n in ('annotation_id','image_id','category_id','original_area'):assert np.array_equal(warm[n],bv[n])
masks=tools.groups(base['annotation'],warm)
for n in ('input_under4','input_4_to_under8','input_8_to_under16','input_ge16'):masks['COCO_small_'+n]=masks['COCO_small']&masks[n]
for cid in range(10):masks['COCO_small_class_'+str(cid)]=masks['COCO_small']&(warm['category_id']==cid)
trajectories={}
for t in (.5,.75):
 trajectories[str(t)]={}
 for name,mask in masks.items():
  pair={}
  for tag,values in (('BASE12',bv),('warm12',warm)):
   support=values['max_iou']>=t;i=support[:,0]
   pre=values['initial_best_at_pre']>=t;final=values['initial_best_at_final']>=t
   final_any=support[:,2]
   counts=dict(GT=int(mask.sum()),initial_supported=int((mask&i).sum()),
    initial_best_same_query_retains_at_pre=int((mask&i&pre).sum()),
    initial_best_same_query_retains_at_final=int((mask&i&final).sum()),
    initial_best_same_query_lost_after_pre=int((mask&i&pre&~final).sum()),
    initial_best_same_query_lost_before_pre_recovers_final=int((mask&i&~pre&final).sum()),
    initial_best_query_lost_but_other_final_query_supports=int((mask&i&~final&final_any).sum()),
    initial_supported_all_final_query_support_lost=int((mask&i&~final_any).sum()))
   assert counts['initial_supported']==counts['initial_best_same_query_retains_at_final']+counts['initial_best_query_lost_but_other_final_query_supports']+counts['initial_supported_all_final_query_support_lost']
   pair[tag]=counts
  pair['warm_minus_BASE']={n:pair['warm12'][n]-pair['BASE12'][n] for n in pair['BASE12']}
  trajectories[str(t)][name]=pair
small_counts=np.bincount(warm['category_id'][masks['COCO_small']],minlength=10)
assert small_counts.sum()==26586 and (small_counts>0).all()
pr={}
for tag,path in [('BASE12',base['packet']/'baseline_eval/endpoint_epoch12'),('warm12',packet/'run/endpoint_epoch12')]:
 arrays=npz(path/'precision_recall.npz');P=arrays['precision'];R=arrays['recall']
 assert P.shape==(10,101,10,4,3) and R.shape==(10,10,4,3)
 counts=R[:,:,1,2]*small_counts[None,:]
 assert np.isfinite(counts).all() and np.max(np.abs(counts-np.rint(counts)))<1e-8
 TP=np.rint(counts).astype(np.int64)
 perIoU=[]
 for ti,t in enumerate(np.linspace(.5,.95,10)):
  v=P[ti,:,:,1,2];good=v[v>-1];assert len(good)==101*10
  perIoU.append(dict(IoU=float(t),APs_percent=float(good.mean()*100),small_area_specific_TP=int(TP[ti].sum()),
                     class_small_area_specific_TP=TP[ti].tolist()))
 mean=sum(x['APs_percent'] for x in perIoU)/10
 assert abs(mean-r[tag]['APs'])<1e-9
 pr[tag]=dict(APs=mean,perIoU=perIoU,per_class_small_GT_counts=small_counts.tolist())
for cls in r['class_APs_arithmetic']:
 cid=cls['class_id'];cls['small_area_specific_TP_delta_by_IoU']=[pr['warm12']['perIoU'][i]['class_small_area_specific_TP'][cid]-pr['BASE12']['perIoU'][i]['class_small_area_specific_TP'][cid] for i in range(10)]
deltas=[dict(IoU=a['IoU'],APs_delta_points=b['APs_percent']-a['APs_percent'],
              small_area_specific_TP_delta=b['small_area_specific_TP']-a['small_area_specific_TP']) for a,b in zip(pr['BASE12']['perIoU'],pr['warm12']['perIoU'])]
assert abs(sum(d['APs_delta_points'] for d in deltas)/10-r['delta_points']['APs'])<1e-9
worst=sorted(r['class_APs_arithmetic'],key=lambda c:c['contribution_to_total_APs_points'])
result=dict(status='WARM_SEED2_NATIVE_SMALL_AP_ARITHMETIC_AND_WITHIN_QUERY_TRAJECTORY_DESCRIPTION',seed=2,
 BASE12=r['BASE12'],warm12=r['warm12'],delta_points=r['delta_points'],native_gate_passed=False,
 class_APs=r['class_APs_arithmetic'],APs_by_IoU=pr,APs_and_area_specific_TP_delta_by_IoU=deltas,within_run_initial_best_query_trajectories=trajectories,
 existing_same_GT_outcomes=r['outcomes'],ranking_ambiguous_images=r['ranking_ambiguous_images'],native_score_list_max_abs_error=r['native_score_list_max_abs_error'],
 direct_findings=[
 'At IoU0.50 small initial support increases573, final support increases159 and all-area native TP on small GT increases49, while APs decreases0.14576 across the full10-IoU macro metric.',
 'At IoU0.75 small initial support increases347 and first-layer-pre support303, while final geometric support decreases13 and native all-area TP on small GT decreases2.',
 'The input-shortside-under4 cohort gains89 native all-area TP at IoU0.50 and19 at0.75; these are counts, not subgroup AP values.',
 'Class APs changes are heterogeneous; people has the largest negative contribution and van the largest positive contribution. Early-support gains do not establish effective final ranking or consistent high-IoU localization.',
 ],
 changed_next_action='Check the actual within-run reference-to-pre-to-final query path and class/IoU losses instead of inferring uniform candidate scarcity from AP. Retain this failed fixed12 endpoint, do not extend/repeat Warm. Complete original user P2/SiLU/attention controls; the dense-location selection diagnostic remains required to distinguish initial selection from regressed reference quality.',
 boundaries='Initial-best means first argmax in the preserved native query order, not a uniquely best identity. Queries are followed only within each run; no cross-run query identity assumed. The pre boundary and final boundary cannot identify a unique decoder layer1/2 or a loss root. Native small-area TP counts come from closed-small GT counts times native small-area recall; all-area TP flags are reported separately. No AP causal decomposition.',
 saved_EMA_independent_replay_verified=False,formal_terminal_reader_passed=False,other_seed_primary_endpoint_pending=True,
 new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_training_epochs=0,new_optimizer_updates=0,
 root_proven=False,novelty_proven=False,goal_complete=False,source_sha256=sha(__file__),
 evidence=dict(primary_analysis_manifest_sha256=sha(analysis/'manifest.json'),primary_verification_sha256=sha(folder/'primary_verification_v1.json'),
               baseline_stage_cache_manifest_sha256=base['cache_stage_manifest_sha256'],baseline_rank_cache_manifest_sha256=base['cache_rank_manifest_sha256']))
dest=ROOT/'outputs/warm_stem_failure_review12_20261011_v1';dest.mkdir(exist_ok=False)
write(dest/'report.json',result)
lines=['保留预训练Stem＋增强支路：种子2固定12轮失败分析','',
 'AP %.4f → %.4f，APs %.4f → %.4f；差值 %.4f / %.4f。'%(r['BASE12']['AP'],r['warm12']['AP'],r['BASE12']['APs'],r['warm12']['APs'],r['delta_points']['AP'],r['delta_points']['APs']),
 '615份已下载文件和全部548张阶段数组通过校验；完整原生精度/召回数组、GT命中与匹配记录均核对。3张图的排序近边界歧义保留；未用这些描述替代独立EMA核验。','',
 '小目标（26586个GT）的几何覆盖与原生命中：']
for t in ('0.5','0.75'):
 row=r['outcomes'][t]['COCO_small'];lines.append('IoU '+t+'：'+ '，'.join(n+' '+str(v['baseline'])+'→'+str(v['warm'])+'（'+str(v['delta'])+'）' for n,v in row['supported_GT_by_boundary'].items())+'；原生全面积匹配在小GT上的TP净变化 '+str(row['native_all_area_TP_net']))
lines.extend(['','IoU0.75：追踪本次运行中初始IoU最大的同一查询，不能用不同查询的“各阶段最好框”冒充同一框的改善。'])
for tag,vals in trajectories['0.75']['COCO_small'].items():
 lines.append(tag+': '+json.dumps(vals,ensure_ascii=False))
lines.extend(['','最小尺寸子群：输入短边<4像素的5096个小GT，IoU0.50命中净增89、IoU0.75净增19。这不是子群AP。','',
 '各类APs变化（个百分点，整体APs是10类等权均值）：'])
for c in r['class_APs_arithmetic']:
 lines.append('%s：%.4f→%.4f，差值%+.4f；贡献到总体APs %+.5f'%(c['name'],c['BASE_APs'],c['warm_APs'],c['delta_APs_points'],c['contribution_to_total_APs_points']))
lines.extend(['','各IoU的原生小面积AP与命中变化：'])
for d in deltas:lines.append('IoU %.2f：APs差值%+.5f，小面积专用匹配TP差值 %+d'%(d['IoU'],d['APs_delta_points'],d['small_area_specific_TP_delta']))
lines.extend(['','判断与下一步：早期参考框覆盖有所增加，但没有稳定保留到最终高IoU定位和各类AP。不能把失败统一解释为“看不到小目标”，也不能断言Stem特征完全没有作用。需要结合原始P2/SiLU/注意力双种子固定12轮结果，继续区分定位传播和类别排序损失。Warm这条路线不延长、不重跑。',
 '目前只定位到选中300个回归参考框、首层pre输出、最终输出三个边界。没有8400个已回归框，也不能独立确认TopK筛选、Decoder第1/2层或监督分配是根因。',
 '另一种子仍训练；独立EMA与正式終点读取待现有GPU作业完成。'])
(dest/'Stem未提升的逐目标分析.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},source_sha256=sha(__file__)))
assert 'torch' not in sys.modules
print(json.dumps(dict(status=result['status'],delta=r['delta_points'],IoU_deltas=deltas,tracked_query75=trajectories['0.75']['COCO_small']),ensure_ascii=False))
