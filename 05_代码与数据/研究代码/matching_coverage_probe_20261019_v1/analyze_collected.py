"""Descriptive paired source-group summaries; no training, inference, or p-values."""
import csv,json,pathlib,statistics
import networkx as nx
ROOT=pathlib.Path(__file__).resolve().parents[2]/'data/08_匹配聚合覆盖探针_20261019_v1'
RULES=['native_union','coverage_constrained','last_anchor_fill']
def avg(x):return statistics.mean(x) if x else None
def main():
    manifest=json.loads((ROOT/'shared_inputs/inputs/manifest.json').read_text(encoding='utf8'))
    graph=nx.Graph();graph.add_nodes_from(range(16));batches={r['batch']:r for r in manifest['batches']}
    for i in range(16):
        for j in range(i):
            if set(batches[i]['source_ids'])&set(batches[j]['source_ids']):graph.add_edge(i,j)
    components=[sorted(c) for c in nx.connected_components(graph)]
    assert all(len({batches[b]['group'] for b in c})==1 for c in components)
    report={'status':'descriptive_stage0_only','model_forwards_are_frozen':True,'optimizer_steps':0,'components':components,'groups':{},'models':{},'no_training_launched':True}
    csvrows=[]
    for account,seed in [('A',1),('B',2)]:
        folder=ROOT/account/f'results_seed{seed}';complete=json.loads((folder/'complete.json').read_text());assert complete['formal_inputs']==64 and complete['optimizer_steps']==0 and complete['BN_and_state_restored']
        analyses={i:json.loads((folder/f'batch{i:02d}_analysis.json').read_text()) for i in range(16)}
        model={'seed':seed,'completion':complete,'groups':{}}
        for group in ['development','heldout']:
            ids=[i for i in analyses if batches[i]['group']==group];cases=[r for i in ids for r in analyses[i]['inputs']];eligible=[r for r in cases if r['capacity_eligible']]
            missing=[r for r in eligible if r['native_missing_GT']];different=[r for r in eligible if r['candidate_differs_anchor']]
            units=[]
            for comp in [c for c in components if batches[c[0]]['group']==group]:
                q={};loss={}
                for rule in RULES:
                    batch_iou=[];batch_loss=[]
                    for i in comp:
                        image_iou=[avg([v[rule]['mean_iou'] for v in case['branches'].values() if v[rule]['mean_iou'] is not None]) for case in analyses[i]['inputs']]
                        batch_iou.append(avg([v for v in image_iou if v is not None]))
                        part=analyses[i]['batch_losses'][rule];batch_loss.append(sum(part['losses'][k] for k in part['ordinary_direct_loss_keys']))
                    q[rule]=avg([v for v in batch_iou if v is not None]);loss[rule]=avg(batch_loss)
                units.append({'batches':comp,'mean_pair_IoU':q,'weighted_direct_loss':loss})
            summary={'inputs':len(cases),'source_components':len(units),'GT_appearances':sum(r['GT'] for r in cases),'eligible_inputs':len(eligible),'eligible_GT_appearances':sum(r['GT'] for r in eligible),'overflow_inputs':sum(r['GT']>300 for r in cases),'native_missing_inputs':len(missing),'native_missing_GT_appearances':sum(len(r['native_missing_GT']) for r in missing),'candidate_differs_anchor_inputs':len(different),'candidate_added_covered_GT':sum(r['candidate_covered']-r['native_covered'] for r in eligible),'mean_vote_loss_ratio_missing_only':avg([r['vote_loss_ratio'] for r in missing]),'solver_seconds_sum':sum(r['solver_seconds'] for r in cases),'source_units':units,'mean_pair_IoU':{rule:avg([u['mean_pair_IoU'][rule] for u in units if u['mean_pair_IoU'][rule] is not None]) for rule in RULES},'weighted_direct_loss':{rule:avg([u['weighted_direct_loss'][rule] for u in units]) for rule in RULES}}
            model['groups'][group]=summary
            csvrows.append({'account':account,'seed':seed,'group':group,**{k:v for k,v in summary.items() if isinstance(v,(int,float)) or v is None},**{'IoU_'+rule:summary['mean_pair_IoU'][rule] for rule in RULES},**{'loss_'+rule:summary['weighted_direct_loss'][rule] for rule in RULES}})
        controls=[]
        for i,a in analyses.items():
            for rule,x in a['batch_losses'].items():
                assert x['DDF_value_identical'] and x['DDF_gradient_max_abs_difference']==0
                controls.append(x['DDF_value_identical'])
        model['DDF_component_isolation_all_passed']=all(controls);report['models'][account]=model
    e=1e-6;hs=[report['models'][a]['groups']['heldout'] for a in ['A','B']]
    has_problem=all(h['native_missing_inputs']>0 for h in hs)
    differs=all(h['candidate_differs_anchor_inputs']>0 for h in hs)
    joint_worse=all(h['mean_pair_IoU']['coverage_constrained']<h['mean_pair_IoU']['native_union']-e and h['weighted_direct_loss']['coverage_constrained']>h['weighted_direct_loss']['native_union']+e for h in hs)
    anchor_dominates=all(h['mean_pair_IoU']['last_anchor_fill']>=h['mean_pair_IoU']['coverage_constrained']-e and h['weighted_direct_loss']['last_anchor_fill']<=h['weighted_direct_loss']['coverage_constrained']+e and (h['mean_pair_IoU']['last_anchor_fill']>h['mean_pair_IoU']['coverage_constrained']+e or h['weighted_direct_loss']['last_anchor_fill']<h['weighted_direct_loss']['coverage_constrained']-e) for h in hs)
    report['gate']={'heldout_applicable_missing_in_both_models':has_problem,'candidate_differs_anchor_in_both_models':differs,'candidate_jointly_worse_than_native_in_both_models':joint_worse,'anchor_dominates_candidate_in_both_models':anchor_dominates,'recommendation':'stop_current_candidate' if not has_problem or not differs or joint_worse or anchor_dominates else 'not_rejected_by_this_probe_only; discuss_next_stage_no_auto_training','epsilon':e}
    (ROOT/'探针结果汇总.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    with (ROOT/'探针来源组汇总.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    lines=['64输入匹配聚合探针：结果说明','两种旧132配方baseline24 raw checkpoint，同一64输入；不更新模型权重。','下列是冻结预测的匹配/损失诊断，不是检测AP，不是历史发生率，不是完整训练的净收益。','']
    for a,m in report['models'].items():
        lines.append(f"账号{a} / 旧baseline seed{m['seed']}")
        for group,h in m['groups'].items():
            lines.append(f"{group}: {h['inputs']}输入/{h['source_components']}来源连通组；容量内适用{h['eligible_inputs']}；原生遗漏{h['native_missing_inputs']}输入/{h['native_missing_GT_appearances']}个增强GT实例；候选不同于简单锚定{h['candidate_differs_anchor_inputs']}输入。")
            for rule in RULES:lines.append(f"  {rule}: 组等权平均pair IoU={h['mean_pair_IoU'][rule]:.8f}；原系数/K归一的普通直接几何损失={h['weighted_direct_loss'][rule]:.8f}")
        lines.append('')
    lines += ['预先判据：',json.dumps(report['gate'],ensure_ascii=False,indent=2),'','边界：DDF直接复用U0分量，其同值及叶梯度一致是实现定义检查，不作为机制证据。','64个增强输入不是64个独立训练重复；源图互斥不保证场景/视频序列独立。不作p值/等效性/发表判断。','正式输入原字节在两端相同；PNG烟测不计入正式结果。GT覆盖增加和保票最优是约束保证，不是探针成功。','没有执行六次短训，没有恢复旧UAVDT，后续等待用户看结果。']
    (ROOT/'64输入探针结果与主线去留.txt').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'gate':report['gate'],'summary':str(ROOT/'64输入探针结果与主线去留.txt')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
