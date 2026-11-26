/** Independent artifact audit and final report; no model execution or repair. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const out=__dirname,old=path.join(out,'..','c_axis_precheck_20261025_v1');
const get=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
const ok=(v,m)=>{if(!v)throw Error(m)};
(async()=>{
 const result=await get(path.join(out,'355b_result.json')),manifest=await get(path.join(out,'355b_delivery_SHA256.json'));
 for(const r of manifest.files)ok(await hash(path.join(out,r.relative))===r.sha256,'delivery SHA '+r.relative);
 const inputs=await get(path.join(out,'355b_input_manifest.json'));let verifiedInputs=0;
 for(const r of [...inputs.files,...inputs.calibration_images,...inputs.validation_images]){ok(await hash(r.path)===r.sha256,'input SHA '+r.path);verifiedInputs++}
 const mu=await get(path.join(out,'355b_calibration_statistics.json'));
 ok(await hash(path.join(out,'355b_training_means.npz'))===mu.training_means_sha256,'mu SHA');
 const frep=await get(path.join(out,'355b_F_reproduction.json'));ok(frep.passed&&Object.values(frep.raw_max_abs).every(v=>v===0)&&Object.values(frep.metrics).every(v=>v.difference===0),'F reproduction');
 const cohorts=await get(path.join(old,'352_cohorts.json')),FM=await get(path.join(out,'355b_F_gt_matches.json'));
 const arms=(await get(path.join(out,'355b_protocol_frozen.json'))).arms;
 const metrics={},sTP={},cacheCounts={},predictionCounts={},valid={},invalidRows={};
 for(const a of arms){
   const cache=await fs.readdir(path.join(out,'355b_'+a+'_batch_cache'));cacheCounts[a]=cache.filter(n=>n.endsWith('.pt')).length;ok(cacheCounts[a]===137,'raw count '+a);
   const rec=await get(path.join(out,'355b_'+a+'_inference_receipt.json'));valid[a]=rec.valid_predictions;
   const rows=await get(path.join(out,'355b_'+a+'_predictions.json')),counts=new Map(),bad=[];
   for(const r of rows){ok(Number.isFinite(r.score)&&r.score>0&&r.score<=1&&r.bbox.length===4&&r.bbox.every(Number.isFinite)&&Number.isInteger(r.category_id)&&r.category_id>=0&&r.category_id<10,'nonfinite or category '+a);if(r.bbox[2]<=0||r.bbox[3]<=0)bad.push(r);counts.set(r.image_id,(counts.get(r.image_id)||0)+1)}
   ok(rows.length===548*300&&counts.size===548&&[...counts.values()].every(n=>n===300),'row count '+a);predictionCounts[a]=rows.length;
   ok(rec.valid_predictions===(bad.length===0),'validity retally '+a);
   if(bad.length){invalidRows[a]=bad;continue}
   metrics[a]=(await get(path.join(out,'355b_'+a+'_metrics.json'))).metrics;
   const M=await get(path.join(out,'355b_'+a+'_gt_matches.json')),comp=await get(path.join(out,'355b_'+a+'_cohort_comparison.json'));
   sTP[a]=cohorts.S_ids.reduce((n,id)=>n+Number(M['0.5'][String(id)].matched),0);
   for(const t of ['0.5','0.75'])for(const g of ['T','S']){
     const ids=cohorts[g+'_ids'].map(String),base=ids.reduce((n,id)=>n+Number(FM[t][id].matched),0),cur=ids.reduce((n,id)=>n+Number(M[t][id].matched),0);
     const rescue=ids.filter(id=>M[t][id].matched&&!FM[t][id].matched).length,lost=ids.filter(id=>FM[t][id].matched&&!M[t][id].matched).length,c=comp[t][g];
     ok(c.GT===ids.length&&base===c.F_TP&&cur===c.arm_TP&&rescue===c.rescued&&lost===c.lost,'TP retally '+a+t+g)
   }
 }
 const losses={};for(const a of Object.keys(metrics)){losses[a]={};for(const k of Object.keys(metrics.F))losses[a][k]=100*(metrics.F[k]-metrics[a][k])}
 const loss=(a,k)=>losses[a]?.[k]??null;
 const acomp=metrics.D0_mean?{AP:loss('D0_mean','AP')<=.1,APs:loss('D0_mean','APs')<=.1,S:(sTP.F-sTP.D0_mean)/sTP.F<=.01}:null;
 const bcomp=Object.fromEntries(['Dx_mean','Dy_mean','Dd_mean'].map(a=>[a,metrics.D0_mean&&metrics[a]?loss('D0_mean','APs')<loss(a,'APs'):null]));
 const bvals=Object.values(bcomp),gates={a_prunable:acomp?Object.values(acomp).every(Boolean):null,b_direction:bvals.includes(false)?false:(bvals.includes(null)?null:true),c_branch_active:metrics.OFF_mean?loss('OFF_mean','APs')>.3:null};
 const primaryValid=arms.slice(0,6).every(a=>valid[a]),decision=!primaryValid?'invalid':(!gates.c_branch_active?'insufficient':(Object.values(gates).every(Boolean)?'pass':'mixed'));
 ok(decision===result.decision,'decision mismatch');
 const counts=await get(path.join(out,'355b_execution_counts.json'));
 ok(counts.full_model_batch_calls===137*11&&counts.full_model_image_forwards===548*11&&counts.calibration_backbone_batch_calls===64&&counts.calibration_backbone_image_forwards===256&&counts.optimizer_steps===0&&counts.model_loads===1&&counts.COCO_evaluations===Object.keys(metrics).length,'execution counts');
 const report={audit:'independent source/outputs SHA, predictions, matched TP and threshold arithmetic; no model execution',verified_input_records:verifiedInputs,verified_output_files:manifest.files.length,core_manifest_sha256:await hash(path.join(out,'355b_delivery_SHA256.json')),valid,invalid_predictions:invalidRows,cache_counts:cacheCounts,prediction_counts:predictionCounts,metrics,losses_pp:losses,S_TP_retally:sTP,a_components_recomputed:acomp,b_comparisons_recomputed:bcomp,gates_recomputed:gates,primary_validity_gate:primaryValid,decision_recomputed:decision,all_integrity_checks_passed:true,actual_execution_counts:counts,optimizer_steps:0};
 const p=path.join(out,'355b_独立封存复核.json');await fs.writeFile(p,JSON.stringify(report,null,2)+'\n',{flag:'wx'});await fs.writeFile(p+'.sha256',await hash(p)+'  '+path.basename(p)+'\n',{flag:'wx'});
 const lines=['355b｜最终核查报告｜客户端日期2026-10-25','Material Passport: run+artifact audit; detector optimizer steps=0.','总体未通过；冻结运行状态='+decision+'。无效主臂不能用于通过判定。','F与352：logits/boxes全量最大差0，12指标差全0；详见355b_F_reproduction.json。','', 'arm\tAP(%)\tAPs(%)\tAP下降pp\tAPs下降pp\tS TP\tS TP净下降%'];
 for(const a of arms){if(metrics[a])lines.push([a,(100*metrics[a].AP).toFixed(9),(100*metrics[a].APs).toFixed(9),loss(a,'AP').toFixed(9),loss(a,'APs').toFixed(9),sTP[a],(100*(sTP.F-sTP[a])/sTP.F).toFixed(9)].join('\t'));else lines.push(a+'\tN/A（非法框；原始输出完整保留）')}
 lines.push('','(a) 可裁剪：'+JSON.stringify(acomp),'(b) D0 APs损失严格小于每个细节臂：'+JSON.stringify(bcomp)+'；一个已知比较为false就不满足合取条件；N/A不可补值。','(c) OFF-mean APs损失>0.3pp：'+(gates.c_branch_active===null?'N/A（OFF-mean非法框）':JSON.stringify(gates.c_branch_active))+'；实际损失='+(loss('OFF_mean','APs')??'N/A'),'主臂合法性：'+primaryValid,'','非法框逐条（不删框、不截断、不重算指标）：');
 for(const[a,rows]of Object.entries(invalidRows))lines.push(a+': '+JSON.stringify(rows));
 lines.push('','计算次数：'+JSON.stringify(counts),'输入与输出完整性复核：全部通过。','UAVDT：VisDrone未通过，按协议未启动。重训：未启动。','结果只说明固定权重中DC随图像及空间位置变化的部分无法在冻结容限内被训练均值替代；不否定所有重训高频方法，不证明信息独有性或新颖性。','工程记录：运行器在存在无效主臂时跳过综合判据计算；其自动355b_结果与判读.txt中的loss列因此打印了默认0，该列不能作为实际差值。本报告全部由保存的metrics与逐GT匹配独立重算，N/A不补值；冻结脚本和原始报告均未改动。');
 await fs.writeFile(path.join(out,'355b_最终核查报告.txt'),lines.join('\n')+'\n',{flag:'wx'});
 await fs.writeFile(path.join(out,'355b_读我_报告优先级.txt'),'主报告：355b_最终核查报告.txt；机器可核对版：355b_独立封存复核.json。\n运行器遇到无效主臂时，自动355b_结果与判读.txt的loss列为默认0，不能视为实际差值；所有可用臂的正确差值在主报告中独立重算。冻结脚本、协议、原始输出及该自动报告均保持原件。\n',{flag:'wx'});
 console.log(JSON.stringify({decision,gates,valid,S_TP:sTP,invalid_prediction_counts:Object.fromEntries(Object.entries(invalidRows).map(([a,r])=>[a,r.length])),verified_inputs:verifiedInputs,verified_output_files:manifest.files.length}));
})().catch(e=>{console.error(e);process.exitCode=1});
