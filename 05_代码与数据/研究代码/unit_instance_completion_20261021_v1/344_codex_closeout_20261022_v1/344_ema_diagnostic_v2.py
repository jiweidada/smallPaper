"""EMA val-only CPU diagnosis. Predictions live only in RAM, never target caches."""
import importlib,json,time,hashlib,gc,traceback
C=importlib.import_module('344_context')
import torch
from PIL import Image
S=importlib.import_module('stage1_cache_v3');F=importlib.import_module('343_fusion');E=importlib.import_module('343_evaluator')
torch.set_num_threads(2)
def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def run():
    idx=json.loads((C.SPEC/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
    ar=next(x for x in idx['annotations'] if x['split']=='val');ann=C.VAL/'annotations/val.json'
    assert C.sha(ann)==ar['expected_SHA256_from_existing_audit'];dataset=json.loads(ann.read_text(encoding='utf-8'))
    assert len(dataset['images'])==548
    raw=json.loads((C.S343/'343_result_阶段4全量评估.json').read_text(encoding='utf-8'))
    history=json.loads((C.S343/'343_result_阶段4历史AP精确对照.json').read_text(encoding='utf-8'))
    counts={'checkpoint_safe_loads_for_EMA':0,'model_constructions_via_raw_loader':0,'EMA_strict_loads':0,
        'global_forward_calls':0,'raw_forward_calls':0,'head_forwards':0,'image_decodes':0,'COCO_evaluate_calls':0,
        'prediction_cache_writes':0,'optimizer_steps':0}
    results=[];start=time.perf_counter()
    try:
        for seed in (1,2):
            cp=next(x for x in idx['checkpoints'] if x['seed']==seed);ch=C.sha(cp['path']);assert ch==cp['expected_SHA256_from_288']
            checkpoint=torch.load(cp['path'],map_location='cpu',weights_only=True);counts['checkpoint_safe_loads_for_EMA']+=1
            assert checkpoint['last_epoch']==23 and set(checkpoint['ema'])=={'module','updates'}
            ema=checkpoint['ema']['module'];assert all(torch.is_tensor(v) for v in ema.values())
            model=S.load_real_model(cp['path']);counts['model_constructions_via_raw_loader']+=1
            status=model.load_state_dict(ema,strict=True);counts['EMA_strict_loads']+=1
            assert not status.missing_keys and not status.unexpected_keys
            parameter_count=sum(p.numel() for p in model.parameters())
            assert parameter_count==sum(ema[k].numel() for k,_ in model.named_parameters())
            print(f'EMA seed={seed} strict-loaded tensors={len(ema)} parameters={parameter_count}',flush=True)
            assert all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters())
            before=state_sha(model);updates=int(checkpoint['ema']['updates']);del checkpoint,ema;gc.collect()
            rows=[];per_image=[];s0=time.perf_counter()
            for position,im in enumerate(dataset['images'],1):
                src=C.VAL/'images'/im['file_name']
                with Image.open(src) as f:pil=f.convert('RGB');counts['image_decodes']+=1
                assert pil.size==(im['width'],im['height']);t0=time.perf_counter()
                with torch.no_grad():pb,logits=S.global_forward_640(model,pil)
                counts['global_forward_calls']+=1
                assert torch.isfinite(pb).all() and torch.isfinite(logits).all()
                out,_=F.fuse({'final300_boxes':pb,'final300_logits':logits,'orig_wh':list(pil.size),'window_xyxy_640':torch.tensor([0.,0.,64.,64.])})
                detections=F.coco_rows(im['id'],out);rows.extend(detections)
                per_image.append({'image_id':im['id'],'seconds':time.perf_counter()-t0,'output_detections':len(detections)})
                if position%16==0 or position==548:print(f'EMA seed={seed} {position}/548 forwards={counts["global_forward_calls"]} elapsed={time.perf_counter()-start:.1f}s',flush=True)
            pred_sha=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            _,metrics,log,seconds=E.evaluate(dataset,rows);counts['COCO_evaluate_calls']+=1
            C.text(f'344_EMA_seed{seed}_COCO_v2.txt',log)
            after=state_sha(model);assert after==before and all(p.grad is None for p in model.parameters())
            rawstats=raw['per_seed'][seed-1]['cases']['baseline']['stats12']
            logged=history['per_seed'][seed-1]['training_log_stats12'];logstats=dict(zip(E.NAMES,logged))
            one={'seed':seed,'EMA_key_path':"checkpoint['ema']['module']",'parameter_count':parameter_count,'EMA_state_tensor_count':794,'EMA_updates':updates,'checkpoint_path':cp['path'],
                'checkpoint_sha256':ch,'strict_load_missing':status.missing_keys,'strict_load_unexpected':status.unexpected_keys,
                'raw_stats12':rawstats,'EMA_stats12':metrics,'training_log_stats12':logstats,
                'EMA_minus_log':{k:metrics[k]-logstats[k] for k in E.NAMES},
                'raw_minus_log':{k:rawstats[k]-logstats[k] for k in E.NAMES},
                'AP_matches_history_round5':round(metrics['AP'],5)==round(logstats['AP'],5),
                'all12_matches_history_round5':all(round(metrics[k],5)==round(logstats[k],5) for k in E.NAMES),
                'max_abs_EMA_minus_log':max(abs(metrics[k]-logstats[k]) for k in E.NAMES),
                'prediction_payload_sha256_only':pred_sha,'state_sha_before':before,'state_sha_after':after,'all_grad_none':True,
                'per_image':per_image,'seed_wall_seconds':time.perf_counter()-s0,'COCO_wall_seconds':seconds}
            results.append(one)
            C.write(f'344_result_EMA_seed{seed}_v2.json',{'status':'已执行；诊断结果','result':one,'已执行':dict(counts),
                '未执行':['EMA预测落盘','训练缓存/M0写入','head评估','真实训练','GPU','阶段5'],'未核':['主控独立复核'],
                'evaluator':E.evidence(),'source_sha256':C.sources()})
            print(json.dumps({'seed':seed,'EMA_AP':metrics['AP'],'log_AP':logstats['AP'],'max12diff':one['max_abs_EMA_minus_log']},ensure_ascii=False),flush=True)
            del model,rows,out,pb,logits;gc.collect()
        assert counts['global_forward_calls']==1096
        reproduced=all(r['all12_matches_history_round5'] for r in results)
        conclusion='历史日志评的是EMA' if reproduced else 'EMA未逐项复现日志至小数5位；差异原因尚未完全核定'
        C.write('344_result_EMA诊断_v2.json',{'status':'已验证（CPU诊断）','conclusion':conclusion,'per_seed':results,
            'comparison_precision':'逐项输出完整值与差；复现判定为12项均按小数5位与日志相同，不调整配置。',
            'remaining_possible_causes_if_not_reproduced':[] if reproduced else ['历史预处理/尺寸','Windows.pyd与历史Linux.so','CPU浮点与历史运行差异','ignore区域处理或历史源码身份'],
            'mainline':'继续只用checkpoint[model] raw；EMA框/logits仅在内存供本次评价，未存入任何预测缓存/M0/训练输入。',
            '已执行':counts,'wall_seconds':time.perf_counter()-start,'未执行':['真实训练','GPU','SSH','队列','阶段5'],
            '未核':['主控独立复核','跨平台二进制逐位一致'],'evaluator':E.evidence(),'source_sha256':C.sources()})
    except Exception as ex:
        C.write('344_result_EMA诊断_实现失败_v2.json',{'status':'实现失败','exception':repr(ex),'traceback':traceback.format_exc(),
            '已执行':counts,'未执行':['真实训练','阶段5'],'未核':['EMA诊断完成'],'source_sha256':C.sources()});raise
if __name__=='__main__':run()
