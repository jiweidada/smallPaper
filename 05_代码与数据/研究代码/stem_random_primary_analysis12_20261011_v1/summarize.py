"""Summarize both cold-Stem controls and exact native small-area AP arithmetic."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/stem_random_primary_collection12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def mean(a):
    a=a[a>=0];assert a.size;return float(a.mean())*100

tools=module('random_summary_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
protocol,geometry,onset,rank,readers=tools.prepared()
rows=[];provenance={}
for seed in (1,2):
    folder=ROOT/f'outputs/stem_random_primary_analysis12_20261011_v1/seed{seed}_native_stem_scratch'
    for n,h in read(folder/'manifest.json')['files'].items():assert sha(folder/n)==h,n
    d=read(folder/'report.json');assert d['analysis_source_sha256']==sha(Path(__file__).parent/'analyze.py')
    assert d['status']=='SAME_GT_PRIMARY12_RANDOM_NATIVE_VS_ENHANCED_DESCRIPTION_REPLAY_PENDING'
    assert d['all548_images'] and d['all38759_GT'] and d['all10_categories_retained']
    assert not d['score_mismatch_images'] and d['native_TP_proxy_conflicts']==0
    base=tools.cached_baseline(seed,protocol,readers['full']);annotation=base['annotation']
    assert all(not a.get('ignore',0) and not a.get('iscrowd',0) for a in annotation['annotations'])
    categories=sorted(annotation['categories'],key=lambda c:c['id']);assert [c['id'] for c in categories]==list(range(10))
    small_counts=np.array([sum(a['category_id']==c['id'] and a['area']<=1024 for a in annotation['annotations']) for c in categories])
    assert (small_counts>0).all() and small_counts.sum()==26586
    endpoints={};pr={};rec={};initialization={};sources={}
    for label,arm,kind in (('random_native','native_stem_scratch','reference'),('enhanced','original_stem','full')):
        primary=ROOT/f'outputs/stem_primary_collection_20261011_v1/seed{seed}_{arm}_v1'
        v=read(primary/'primary_verification_v1.json');packet=Path(v['downloaded_packet'])
        assert v['actual_stage_images']==548 and not v['saved_EMA_independent_replay_verified']
        expected='PASS_RANDOM_STEM_PRIMARY12_ALL_FILES_AND548_STAGE_ARRAYS' if kind=='reference' else 'PASS_DOWNLOADED_EXACT12_PRIMARY_ARTIFACTS_ONLY'
        assert v['status']==expected
        verifier=Path(__file__).parents[1]/'stem_random_primary_collection12_20261011_v1/verify_primary.py' if kind=='reference' else ROOT/f'outputs/verify_stem_primary_seed{seed}_20261011_v1.py'
        assert v['verification_script_sha256']==sha(verifier)
        manifest=read(packet/'primary_manifest.json');assert sha(packet/'primary_manifest.json')==v['primary_manifest_sha256']
        for n,i in manifest['files'].items():assert (packet/n).stat().st_size==i['bytes'] and sha(packet/n)==i['sha256'],n
        s,flags,_=readers[kind].native_endpoint(packet/'run/endpoint_epoch12',annotation)
        assert s['identity']==v['identity'] and s['native_stats']==v['native_stats']
        endpoints[label]=s
        with np.load(packet/'run/endpoint_epoch12/precision_recall.npz',allow_pickle=False) as z:
            pr[label]=z['precision'][:,:,:,1,2].copy();rec[label]=z['recall'][:,:,1,2].copy()
        assert pr[label].shape==(10,101,10) and (pr[label]>=0).all()
        assert abs(mean(pr[label])-100*s['native_stats'][3])<1e-9
        initialization[label]=read(packet/'screen/preflight.json')['initialization']
        assert initialization[label]['exact_shared_tensors']==754 and initialization[label]['only_missing_prefix']=='backbone.stem.'
        sources[label]=dict(primary_verification_sha256=sha(primary/'primary_verification_v1.json'),primary_manifest_sha256=v['primary_manifest_sha256'],
            native_endpoint_manifest_sha256=sha(packet/'run/endpoint_epoch12/manifest.json'),checkpoint_sha256=v['last_checkpoint_sha256'],initialization=initialization[label])
    classes=[]
    for cid,c in enumerate(categories):
        delta=mean(pr['enhanced'][:,:,cid])-mean(pr['random_native'][:,:,cid])
        classes.append(dict(category_id=cid,name=c['name'],small_GT=int(small_counts[cid]),enhanced_minus_random_APs_points=delta,
            contribution_to_total_APs_points=delta/10,
            by_IoU_delta_points=[mean(pr['enhanced'][ti,:,cid])-mean(pr['random_native'][ti,:,cid]) for ti in range(10)]))
    assert abs(sum(c['contribution_to_total_APs_points'] for c in classes)-d['full_minus_control_pp']['APs'])<1e-9
    ious=[]
    for ti,t in enumerate(endpoints['enhanced']['protocol']['iou_thresholds']):
        tp={}
        for label in rec:
            a=rec[label][ti]*small_counts;assert np.max(np.abs(a-np.rint(a)))<1e-7
            tp[label]=int(np.rint(a).sum())
        ious.append(dict(IoU=t,enhanced_minus_random_small_AP_points=mean(pr['enhanced'][ti])-mean(pr['random_native'][ti]),
            native_small_area_specific_TP=tp,small_area_specific_TP_delta=tp['enhanced']-tp['random_native']))
    assert abs(np.mean([r['enhanced_minus_random_small_AP_points'] for r in ious])-d['full_minus_control_pp']['APs'])<1e-9
    masks={}
    for name in ('COCO_small','COCO_small_input_under4'):
        masks[name]={}
        for t in ('0.5','0.75'):
            o=d['outcomes'][t][name]
            masks[name][t]=dict(GT=o['GT'],supported_GT_by_boundary=o['supported_GT_by_boundary'],
                native_all_area_TP={label:o[key]['counts']['native_TP'] for label,key in (('random_native','control'),('enhanced','full_enhanced'))},
                native_all_area_TP_gained=o['native_all_area_TP_gained'],native_all_area_TP_lost=o['native_all_area_TP_lost'],
                native_all_area_TP_net=o['native_all_area_TP_net'],gained_TP_control_outcomes=o['gained_TP_control_outcomes'])
    metric={label:{n:s['native_stats'][i]*100 for n,i in (('AP',0),('APs',3))} for label,s in endpoints.items()}
    metric['cached_pretrained_BASE12']={n:base['summary']['native_stats'][i]*100 for n,i in (('AP',0),('APs',3))}
    rows.append(dict(seed=seed,primary_fixed12_metrics=metric,enhanced_minus_random_points=d['full_minus_control_pp'],
        enhanced_minus_pretrained_BASE_points={n:metric['enhanced'][n]-metric['cached_pretrained_BASE12'][n] for n in ('AP','APs')},
        classes=classes,by_IoU=ious,same_GT_cohorts=masks,ranking_ambiguous_images=d['ambiguous_images']))
    provenance[str(seed)]=dict(primary_sources=sources,derived_analysis_manifest_sha256=sha(folder/'manifest.json'),
        cached_BASE_source_report_sha256=base['source_report_sha256'],cached_BASE_stage_manifest_sha256=base['cache_stage_manifest_sha256'])
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
report=dict(status='BOTH_RANDOM_STEM_PRIMARY12_VERIFIED_SAME_GT_AND_NATIVE_SMALL_AP_ARITHMETIC_COMPLETED',seeds=rows,provenance=provenance,
    source_sha256=sha(__file__),prepared_protocol_sha256=sha(tools.PROTOCOL),verified_new_primary_files_each=606,actual_stage_images_each=548,
    all10_classes_and_all10_IoUs_retained=True,small_closed_area_at1024=True,matched_unit='GT ID, not cross-run query or box identity',
    independent_saved_EMA_replay_pending=True,formal_terminal_reader_passed=False,unique_cause_proven=False,novelty_proven=False,
    both_seed_APs_over_random_positive=True,both_seed_total_AP_nonnegative_over_random=False,pretrained_BASE_gate_passed=False,
    changed_next_action='Candidate coverage is not a consistent explanation across seeds. Inspect category/IoU PR and native selected-location-versus-regression diagnostics; assess the native-pretrained-preserving Warm run before selecting a new mechanism.',
    scope='Primary validation-native EMA epoch12 only. Random native versus enhanced removes the native Stem pretraining advantage from both arms; it estimates a net architecture-plus-own-initialization early-learning effect, not a unique parameter-matched mechanism. Native small-area TP derived from bound recall tensors is recorded separately from all-area TP on small GT. AP arithmetic is not model-cause attribution.',
    new_BASE_forwards=0,new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,goal_complete=False)
write(OUT/'analysis_summary.json',report)
lines=['随机原始 Stem 对照：两种子第12轮结果与定位','',
    '两个已停止的随机原始 Stem 结果包均下载并核验：每个606个文件、548张验证图像。以下使用训练期间第12轮 EMA 验证产物；保存权重的独立复验仍待完成。',
    '随机原始与增强结构均移除原生 Stem 预训练，均保留其余754个共享状态。增强模型总参数多5325个。比较包含结构及各自随机初始化的12轮学习效应，不能直接命名为某个唯一机制。','']
for r in rows:
    s=r['seed'];m=r['primary_fixed12_metrics'];lines += [f'种子{s}：',*[f"  {n}: AP={v['AP']:.6f}, APs={v['APs']:.6f}" for n,v in m.items()],
        f"  增强−随机：AP={r['enhanced_minus_random_points']['AP']:+.6f}, APs={r['enhanced_minus_random_points']['APs']:+.6f}"]
    for name,cohort in r['same_GT_cohorts'].items():
        for t,c in cohort.items():
            lines.append(f"  {name}, IoU{t}, GT={c['GT']}: 随机→增强 TP={c['native_all_area_TP']['random_native']}→{c['native_all_area_TP']['enhanced']}; 阶段覆盖="+str(c['supported_GT_by_boundary']))
    lines.append('  各类别对总APs差值的贡献（精确算术，非网络因果分解）：')
    lines += [f"    {c['name']}: {c['contribution_to_total_APs_points']:+.6f}" for c in r['classes']]
    lines.append('  原生small面积专用匹配的TP和各IoU的AP差值见analysis_summary.json；不能把上面的all-area小目标TP代称APs。')
    lines.append('')
lines += ['判断：两个种子的APs都高于随机原始Stem，但种子1总体AP下降；两个种子仍低于预训练BASE12。尚未通过目标验收。',
    '种子1小目标IoU0.5覆盖/TP下降，种子2上升，因此不能把统一原因写成“Stem保留更多细节、候选覆盖提升”。',
    '下一步：对照保留预训练的Warm版本，检查按类别和IoU的PR变化；等待真实TopK位置与选中框回归诊断，区分选位置和回归框的问题。',
    '全程读取已有产物，没有新模型/基线前向、训练、优化更新或COCO/Hungarian重匹配。']
text=OUT/'随机原始Stem对照与机制审查.txt'
with text.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
write(OUT/'analysis_summary_manifest.json',dict(files={p.name:sha(p) for p in (OUT/'analysis_summary.json',text)},source_sha256=sha(__file__)))
print(json.dumps(dict(status=report['status'],seeds=[dict(seed=r['seed'],delta=r['enhanced_minus_random_points'],
    under4_TP={t:c['native_all_area_TP'] for t,c in r['same_GT_cohorts']['COCO_small_input_under4'].items()},
    APs_contributions=[dict(name=c['name'],delta=c['contribution_to_total_APs_points']) for c in r['classes']],
    small_area_TP_by_IoU=[x for x in r['by_IoU'] if abs(x['IoU']-.5)<1e-10 or abs(x['IoU']-.75)<1e-10]) for r in rows])))
