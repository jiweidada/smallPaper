from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
doc=ROOT/'01_会议论文/稿件/MPSMA-DEIM_conference_EN_revised.docx'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
qa=json.loads((OUT/'visual_qa.json').read_text(encoding='utf-8'))
qa['docx_sha256']=sha(doc);qa['revision']=2
qa['inspection']='All 9 final page PNGs opened and visually reviewed. New protocol text on page 5, two original source images and caption on page 6, tables on pages 6–8 and eleven references on page 9 are legible. No observed clipping, overlaps or isolated table headers.'
qa['corrections'].append('Replaced Figure 6 with two identified unmodified validation images and checked source/embed SHA256 equality.')
qa['page_png_sha256']={f'page-{i}.png':sha(OUT/'rendered'/f'page-{i}.png') for i in range(1,10)}
(OUT/'visual_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
runs=json.loads((OUT/'run_audit.json').read_text(encoding='utf-8'))
dataset=json.loads((OUT/'dataset_eval_audit/annotation_audit.json').read_text(encoding='utf-8'))
proposed=[]
for key in ['N0','N4','S1']:
    r=runs[key];p=Path(r['source_dir'])/'best_stg2.pth';h=sha(p)
    assert h==r['checkpoint']['sha256']
    proposed.append({'run_id':key,'checkpoint':str(p),'checkpoint_sha256':h,'configuration':str(OUT/'source_snapshot'/key/'args.json'),'reference_best_log_epoch':r['best_epoch'],'reference_best_AP':r['metrics']['AP']})
plan={'status':'PROPOSED_NOT_AUTHORIZED_NOT_RUN','question':'Do the retained N0/N4/S1 checkpoints reproduce their reported validation metrics under an explicitly recorded implementation and evaluator?','why_existing_evidence_is_insufficient':'The logs and saved evaluator objects do not bind the best metric row to a checkpoint hash; epoch metadata can be reset by the solver.','minimum_scope':proposed,'validation_annotation':dataset['val']['annotation_path'],'validation_annotation_sha256':dataset['val']['annotation_sha256'],'image_count':548,'evaluation':'One validation pass per checkpoint using its retained configuration and a frozen local COCO-style maxDets=100 evaluator. Save loading diagnostics, source and environment versions, predictions and complete metrics. Original-config validation batch differences must be recorded.','stop_conditions':['No compatible supported runtime/device: report and stop; do not rent or access a remote server.','Missing/unexpected model keys or incompatible structure: stop before inference; do not use strict=False to conceal mismatch.','Hash mismatch or missing images: stop.'],'not_authorized':['training','fine-tuning','hyperparameter search','remote access','paid resources','new timing/profiling benchmark','official-protocol re-evaluation'],'limits':'Even a successful re-evaluation will not fix the historical baseline/ablation confounds or prove the exact historical code version. Fair training controls remain separately necessary.'}
(OUT/'proposed_reevaluation.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['下一步最小核验方案（待明确授权，未运行）','回答问题：N0/N4/S1三份留存权重能否重现所报告的验证指标。','原因：日志和现存eval对象没有记录生成最佳行的checkpoint哈希；last_epoch可能被恢复逻辑改写。','最小范围：以下三份权重各验证一次，548张验证图，使用明确冻结的本地COCO-style评价器；不训练，不调参，不新测时延，不访问远程或新增付费资源。','严格加载不匹配、环境/设备不兼容或数据哈希不符时停止报告，不绕过检查。','成功重评也不能解决原基线batch/聚合结构的混杂；公平训练对照仍需另行授权。','']
for p in proposed:lines.extend([p['run_id']+' '+p['checkpoint'],'SHA256: '+p['checkpoint_sha256']])
lines.extend(['','详细机器可读范围：proposed_reevaluation.json。','用户此前明确要求新实验必须先得到授权；截至本记录生成，尚未获得授权。'])
(OUT/'下一步最小核验_待授权.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
(ROOT/'01_会议论文/当前修订入口.txt').write_text('当前内容修订稿：'+str(doc)+'\n修改说明与未解决问题：'+str(OUT/'修订说明与未解决问题.txt')+'\n状态：第二轮完成，补齐现存标注/评价口径及图6来源；尚不能投稿。没有新实验。会议未选定。\n待授权的三份留存权重核验范围：'+str(OUT/'下一步最小核验_待授权.txt')+'\n',encoding='utf-8-sig')
print(json.dumps({'revision':2,'docx_sha256':sha(doc),'pages_reviewed':9,'proposal':'NOT_AUTHORIZED_NOT_RUN'},ensure_ascii=True))
