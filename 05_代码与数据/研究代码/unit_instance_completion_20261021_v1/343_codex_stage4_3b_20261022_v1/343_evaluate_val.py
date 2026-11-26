"""Cached raw baselines and synthetic additions only; no model loading/forward."""
import importlib,json,time
C=importlib.import_module('343_context')
import torch,numpy as np
F=importlib.import_module('343_fusion');E=importlib.import_module('343_evaluator')
torch.set_num_threads(2)

def main():
    ann=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
    data=json.loads(ann.read_text(encoding='utf-8'));annsha=C.sha(ann)
    assert len(data['images'])==548
    results=[];cfgs=[];all_equal=True
    for seed in (1,2):
        cfg=C.DATA/f'本次24轮双基线模型下载_20261009_v1/seed{seed}/bundle/resolved.yml'
        config=json.loads(cfg.read_text(encoding='utf-8'))
        assert config['evaluator']=={'type':'CocoEvaluator','iou_types':['bbox'],'using_tiny_metrice':False,'using_vtiny_metrice':False,'max_dets':100}
        cfgs.append({'path':str(cfg),'sha256':C.sha(cfg),'evaluator':config['evaluator']})
        rng=np.random.default_rng(2026102207+seed)
        modes={k:[] for k in ('baseline','low_new4','random_new4')}
        selected=[]
        for iid in range(548):
            e=torch.load(C.S3/f'342_cache_val_v2_seed{seed}'/f'342_{iid:06d}.pt',map_location='cpu',weights_only=True)
            assert e['annotation_sha256']==annsha
            # Synthetic valid center/width/height; no head is executed.
            local=torch.tensor(rng.uniform(.05,.95,(4,4)),dtype=torch.float32)
            baseline,_=F.fuse(e)
            low,_=F.fuse(e,local,torch.full((4,10),-12.))
            random,_=F.fuse(e,local,torch.tensor(rng.normal(0,2,(4,10)),dtype=torch.float32))
            selected.append({'image_id':iid,'low_new_selected':int(low['is_new'].sum()),'random_new_selected':int(random['is_new'].sum())})
            for name,out in [('baseline',baseline),('low_new4',low),('random_new4',random)]:
                modes[name].extend(F.coco_rows(iid,out))
        localresults={}
        for name,rows in modes.items():
            predname=f'343_val_seed{seed}_{name}_predictions.json';C.write(predname,rows)
            _,stats,log,seconds=E.evaluate(data,rows)
            C.text(f'343_val_seed{seed}_{name}_COCO.txt',log)
            localresults[name]={'stats12':stats,'detections':len(rows),'wall_seconds':seconds,
                'prediction_file':str(C.HERE/predname),'prediction_file_sha256':C.sha(C.HERE/predname)}
            print(f'seed={seed} {name}: AP={stats["AP"]:.12f} APs={stats["APs"]:.12f}; {seconds:.1f}s',flush=True)
        eq=localresults['baseline']['stats12']==localresults['low_new4']['stats12'];all_equal &= eq
        recorded={1:.16603378,2:.1640869}[seed]
        results.append({'seed':seed,'cases':localresults,'low_new4_stats12_bit_equal':eq,
            'training_log_AP':recorded,'AP_difference_from_training_log':localresults['baseline']['stats12']['AP']-recorded,
            'synthetic_selected_per_image':selected})
    C.write('343_result_阶段4全量评估.json',{'status':'已验证' if all_equal else '实现失败',
        'low_new4_stats12_bit_equal_both_seeds':all_equal,'per_seed':results,'evaluator':E.evidence(),
        'config_evidence':cfgs,'annotation_path':str(ann),'annotation_sha256':annsha,
        'baseline_identity':'阶段3 val v2的BASE24 checkpoint[model] raw缓存；不执行任何模型或训练产物。',
        'baseline_mismatch_policy':'不改评估设置。差异可能涉及本机.pyd与Linux.so、原始训练运行或浮点/排序；仅列可能原因，未核因果。',
        '已执行':{'cache_loads':1096,'COCO_evaluate_calls':6,'COCO_accumulate_calls':6,'COCO_summarize_calls':6,
            'fusion_calls':3288,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['训练后八run评估','真实训练','GPU','阶段5'],
        '未核':['主控独立核对','Windows与Linux二进制一致性'],'source_sha256':C.sources()})

if __name__=='__main__':main()
