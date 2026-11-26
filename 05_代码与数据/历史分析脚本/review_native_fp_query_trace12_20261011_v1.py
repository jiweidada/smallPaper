"""Bind eight actual saved-array runs and test a naive top-one deletion without rematching."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def verified(p):
    m=read(p/'manifest.json')
    for n,h in m['files'].items():assert sha(p/n)==h,n
    return read(p/'report.json')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
dest=ROOT/'outputs/native_fp_query_trace_review12_20261011_v1';dest.mkdir(exist_ok=False)
sources={};runs={};cells=0;all_tables=[];deletion_tables=[]
for seed in (1,2):
    for tag in ('BASE12','Warm12'):
        label=f'seed{seed}_{tag}'
        gp=ROOT/f'outputs/native_fp_geometry12_20261011_v1/{label}'
        tp=ROOT/f'outputs/native_fp_query_trace12_20261011_v1/{label}'
        g,t=verified(gp),verified(tp)
        assert g['identity']==t['identity'] and g['seed']==t['seed']==seed and g['tag']==t['tag']==tag
        assert sha(gp/'manifest.json')==t['native_FP_geometry_manifest_sha256']
        assert g['native_TP_counts_exact'] and g['native_FP_counts_exact'] and g['native_stable_global_and_local_score_ties_preserved']
        for r,family in ((g,'native_fp_geometry12_20261011_v1'),(t,'native_fp_query_trace12_20261011_v1')):
            code=ROOT/'英文小论文/code'/family
            assert sha(code/'analyze.py')==r['analysis_sha256'] and sha(code/'protocol.json')==r['protocol_sha256']
            for n,h in read(code/'protocol.json')['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
            for field in ('new_model_forwards','new_BASE_forwards','new_evaluator_calls','new_native_matching_calls','new_optimizer_updates','new_training_epochs'):assert r[field]==0
        if tag=='BASE12':
            packet=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1/extracted'
            run=packet/'baseline_eval';ann=read(packet/'reference/val_annotations.json')
        else:
            vpath=ROOT/f'outputs/warm_stem_primary_collection12_20261011_v1/seed{seed}_warm_stem_v1/primary_verification_v1.json'
            v=read(vpath);assert sha(vpath)==g['proof']['source_verification_sha256']
            assert v['status']=='PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
            packet=Path(v['downloaded_packet']);run=packet/'run'
            ann=read(ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1/extracted/reference/val_annotations.json')
        endpoint=run/'endpoint_epoch12';stages=run/'stages_epoch12'
        assert sha(endpoint/'manifest.json')==g['proof']['endpoint_manifest_sha256']
        assert sha(stages/'manifest.json')==g['proof']['stage_manifest_sha256']
        for n,h in read(endpoint/'manifest.json')['files'].items():assert sha(endpoint/n)==h,n
        summary=read(endpoint/'summary.json');assert summary['identity']==g['identity'] and summary['epoch_zero_based']==11 and summary['evaluated_weights']=='EMA'
        native=npz(endpoint/'precision_recall.npz')['precision'][:,:,:,0,2]
        ga=npz(gp/'native_FP_categories.npz');ta=npz(tp/'within_run_query_FP_trace.npz')
        stage_manifest=read(stages/'manifest.json');maxclass={};uncertain={};margin=1e-5
        for n,h in stage_manifest['files'].items():assert sha(stages/n)==h,n
        for iid in range(548):
            with np.load(stages/f'image_{iid:06d}.npz',allow_pickle=False) as z:logits=z['final_logits'].astype(np.float64)
            assert logits.shape==(300,10) and np.isfinite(logits).all()
            order=np.argsort(-logits,axis=1,kind='stable');maxclass[iid]=order[:,0]
            uncertain[iid]=(logits[np.arange(300),order[:,0]]-logits[np.arange(300),order[:,1]])<=margin
        pr=np.zeros_like(native);top1_status={};class_top1=[];fp_all=[]
        for cid in range(10):
            codes=ga[f'class{cid}_codes'];assert codes.shape[0]==10
            scores=ga[f'class{cid}_native_scores'];assert np.all(np.diff(scores)<=0)
            qids=ta[f'class{cid}_known_query_index'];images=ga[f'class{cid}_image_id']
            assert len(qids)==len(images)==codes.shape[1]
            status=np.full(len(qids),2,np.int8)
            for iid in np.unique(images):
                rows=np.flatnonzero((images==iid)&(qids>=0));query=qids[rows]
                confident=~uncertain[int(iid)][query];rr=rows[confident];qq=query[confident]
                status[rr]=(maxclass[int(iid)][qq]!=cid).astype(np.int8)
            top1_status[f'class{cid}_top1_deletion_status']=status
            gt=sum(a['category_id']==cid for a in ann['annotations'])
            tp_cum=np.cumsum(codes==0,axis=1).astype(float)
            fp_cum=np.cumsum((codes!=0)&(codes!=9),axis=1).astype(float)
            rc=tp_cum/gt;raw_pr=tp_cum/(tp_cum+fp_cum+np.spacing(1))
            envelope=np.maximum.accumulate(raw_pr[:,::-1],axis=1)[:,::-1]
            by_iou=[]
            for ti in range(10):
                indices=np.searchsorted(rc[ti],summary['protocol']['recall_thresholds'],side='left');valid=indices<len(scores)
                pr[ti,valid,cid]=envelope[ti,indices[valid]]
                assert np.array_equal(pr[ti,:,cid],native[ti,:,cid]),(seed,tag,cid,ti)
                for p in g['classes'][cid]['by_IoU'][ti]['prefixes']:
                    n=p['rank'];assert p['TP']==int(tp_cum[ti,n-1]) and p['FP']==int(fp_cum[ti,n-1])
                prefix=[]
                for p in g['classes'][cid]['by_IoU'][ti]['prefixes']:
                    n=p['rank'];c=codes[ti,:n];d=status[:n]==1;u=status[:n]==2
                    prefix.append(dict(rank=n,currently_native_TP_entries_deleted=int(((c==0)&d).sum()),currently_native_FP_entries_deleted=int(((c!=0)&(c!=9)&d).sum()),entries_with_query_or_argmax_margin_uncertain=int(u.sum())))
                by_iou.append(dict(IoU=summary['protocol']['iou_thresholds'][ti],prefixes=prefix))
            cells+=int(native[:,:,cid].size)
            class_top1.append(dict(class_id=cid,name=g['classes'][cid]['name'],by_IoU=by_iou))
            fp_all.append(g['classes'][cid]['by_IoU'][0]['prefixes'][-1])
        assert np.array_equal(pr,native)
        pe=next(p for p in g['classes'][1]['by_IoU'][0]['prefixes'] if p['rank']==1000)
        tr=next(p for p in t['classes'][1]['by_IoU'][0]['prefixes'] if p['rank']==1000)
        de=next(p for p in class_top1[1]['by_IoU'][0]['prefixes'] if p['rank']==1000)
        all_tables.append(dict(seed=seed,tag=tag,geometry=pe,trace=tr));deletion_tables.append(dict(seed=seed,tag=tag,**de))
        np.savez_compressed(dest/(label+'_top1_deletion_status.npz'),**top1_status)
        runs[label]=dict(identity=g['identity'],native_AP=100*summary['native_stats'][0],native_APs=100*summary['native_stats'][3],geometry_manifest_sha256=sha(gp/'manifest.json'),trace_manifest_sha256=sha(tp/'manifest.json'),native_all_area_precision_cells_verified=int(native.size),top1_deletion_description_all_classes_and_IoUs=class_top1,all_class_IoU50_complete_prefixes=fp_all,unique_query_uncertainty_retained=True)
        sources[gp.relative_to(ROOT).as_posix()+'/manifest.json']=sha(gp/'manifest.json');sources[tp.relative_to(ROOT).as_posix()+'/manifest.json']=sha(tp/'manifest.json')
        print(json.dumps(dict(run=label,people1000=de,cells=int(native.size))),flush=True)
assert cells==40400
warm_review=ROOT/'outputs/warm_stem_both_seeds_review12_20261011_v1/report.json'
wr=read(warm_review);assert not wr['two_seed_native_gate_passed'] if 'two_seed_native_gate_passed' in wr else True
native_root=ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed1_target_protect_v1/extracted/native'
lqe=native_root/'engine/deim/dfine_decoder.py';post=native_root/'engine/deim/postprocessor.py'
lqe_text=lqe.read_text(encoding='utf8');assert 'self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)' in lqe_text and 'return scores + quality_score' in lqe_text
report=dict(status='PASS_EIGHT_FP_TRACE_RUNS_BOUND_EXACT40400_NATIVE_PR_CELLS_AND_TOP1_DELETION_RISK_DESCRIBED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=runs,people_IoU50_prefix1000_examples=all_tables,people_top1_deletion_examples=deletion_tables,native_all_area_PR_cells_verified=cells,native_PR_max_abs_error=0,source_manifests=sources,warm_two_seed_result_source_sha256=sha(warm_review),warm_two_seed_gate_passed=False,analysis_source_sha256=sha(__file__),source_LQE_sha256=sha(lqe),source_postprocessor_sha256=sha(post),LQE_mathematical_property='One scalar is added to every class logit of one query. It preserves class logit differences and the query argmax, but can change absolute sigmoid scores and cross-query class-entry ordering. Raw pre-LQE logits and correction are not saved here; their learned causal effect is unresolved.',top1_scope='Delete known non-argmax query/category entries from the already emitted native list, only where the largest-to-second logit margin exceeds1e-5. Counts describe currently native-matched entries removed. No refill of flattenedTop300, no new matching, no counterfactual AP or APs, and no inference that each deleted current TP becomes a finally missed GT.',inference='Some final same-class occupied-target false positives already overlap the same target at selected regressed references; some other-class false positives share a query with a native TP in a different category. Neither proves a unique faulty module. Uniform Stem coverage improvement does not explain both seeds. Inspect selected8400-to300 and regression separately; quantify class-entry competition before deciding a model intervention.',new_model_forwards=0,new_BASE_forwards=0,new_training_epochs=0,new_optimizer_updates=0,new_evaluator_calls=0,new_native_matching_calls=0,root_proven=False,novelty_proven=False)
write(dest/'report.json',report)
lines=['误检与查询轨迹审查（已保存的固定第12轮EMA，全部10类×10个IoU）','已绑定8份实际分析产物；由误检编码独立还原40,400个原生全尺度PR单元，误差0。','', '示例：people类、IoU0.50、原生分数前1000条。此表不是APs分解，也不是根因标签。','配置 | TP/FP | 同类已占用目标FP | 异类重叠FP | 低重叠FP | 同类FP最早观察到相同最终目标：初始/首层pre/最终']
for e in all_tables:
    c=e['geometry']['counts'];s=e['trace']['first_FP_same_occupied_GT_support']
    lines.append(f"seed{e['seed']} {e['tag']} | {e['geometry']['TP']}/{e['geometry']['FP']} | {c['same_class_eligible_GT_all_already_native_matched_earlier']} | {c['other_class_overlap_at_threshold']} | {c['overlap_below0_1_any_class']} | {s['selected_regressed_encoder_reference']}/{s['first_layer_pre']}/{s['final_only']}")
lines+=['','每查询仅保留最高类的删除检查（不补选、不重匹配、不计算替代AP）：']
for d in deletion_tables:lines.append(f"seed{d['seed']} {d['tag']} people前1000：会删掉当前TP {d['currently_native_TP_entries_deleted']}条、当前FP {d['currently_native_FP_entries_deleted']}条、不确定 {d['entries_with_query_or_argmax_margin_uncertain']}条。")
lines+=['','原生LQE对一个查询的所有类别加同一个质量修正，因此不能改变该查询内部的类别排序；它可能改变查询间分数排序。现有产物没有分离LQE前的同层logits，不能断言LQE导致这些错误。','初始参考框已经经过Top300选择及回归；不能由当前三阶段轨迹把两者归因拆开。多个初始查询覆盖同一目标也是正常的候选现象，不能直接当成编码器错误。','Warm Stem种子1提升、种子2下降，双种子门槛失败。保留负结果，不延长或重跑。后续以原始用户模块实验和已部署的8400→300诊断补齐证据。']
(dest/'误检查询轨迹与修改风险审查.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
print(json.dumps(dict(status=report['status'],cells=cells)),flush=True)
