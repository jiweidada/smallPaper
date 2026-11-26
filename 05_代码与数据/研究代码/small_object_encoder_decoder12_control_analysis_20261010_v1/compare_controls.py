"""Apply the original two-seed fixed12 gate to verified fine/control packets."""
from pathlib import Path
import argparse, csv, datetime, json, sys
import numpy as np
from results import read, write, load_endpoint, pair
from evaluator_binding import same_result_evaluator
from control_proofs import ROOT, FINE, PLAN, PLAN_SHA, fine_local, verify_promotion_packet, sha
import geometry_core as geometry

OUT = ROOT/'outputs/small_object_encoder_decoder12_20261010_v5'
STATUS = 'PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'


def load_arrays(directory):
    manifest = read(directory/'manifest.json')
    for name,h in manifest['files'].items(): assert sha(directory/name)==h,name
    report=read(directory/'report.json')
    assert report['native_TP_geometry_disagreement_cells']==0
    with np.load(directory/'candidate_per_GT.npz',allow_pickle=False) as z:
        values={n:z[n].copy() for n in z.files}
    assert len(values['annotation_id'])==38759
    return values,report


def compare_seed(seed):
    fine,fine_report,fine_done,fine_report_path=fine_local(seed)
    collection=OUT/f'terminal_collection/seed{seed}'
    packet=collection/'extracted';report_path=collection/'verification_report.json'
    report=read(report_path)
    assert report['status']==STATUS and report['seed']==seed and report['variant']=='degraded'
    assert report['all548_saved_weight_stage_arrays_exact'] and report['stage_geometry_TP_conflicts']==0
    assert report['model_forwards']==report['evaluator_calls']==report['training_updates']==0
    assert report['cost']['candidate_parameters']==fine_report['cost']['candidate_parameters']==13806625
    done=read(packet/'run/training_completion.json')
    assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
    assert done['identity']['arm']=='degraded' and done['identity']['seed']==seed
    assert sha(packet/'run/last.pth')==done['last_checkpoint_sha256']
    assert report['final_training_counts']['attempts_total']==report['final_training_counts']['EMA_updates']==19404
    promotion=verify_promotion_packet(packet)
    fs,fh,_=load_endpoint(fine/'run/endpoint_epoch12')
    ds,dh,_=load_endpoint(packet/'run/endpoint_epoch12')
    fb,_,_=load_endpoint(fine/'baseline_eval/endpoint_epoch12')
    db,_,_=load_endpoint(packet/'baseline_eval/endpoint_epoch12')
    assert fb['native_stats']==db['native_stats']
    assert fs['overall_percent']==fine_report['candidate_overall_percent']
    assert ds['overall_percent']==report['candidate_overall_percent']
    same_result_evaluator(fs,ds);same_result_evaluator(fb,db)
    native_gain=fs['overall_percent']['APs']>fb['overall_percent']['APs'] and fs['overall_percent']['AP']>=fb['overall_percent']['AP']
    assert native_gain and fine_report['native_gate_passed_this_seed']
    compared=pair(ds,fs,dh,fh)  # Fixed direction: fine minus degraded.
    fine_values,fine_stages=load_arrays(FINE/f'paired_seed{seed}')
    degraded_values,degraded_stages=load_arrays(OUT/f'paired_seed{seed}')
    for k in ('annotation_id','image_id','category_id','original_area'):
        assert np.array_equal(fine_values[k],degraded_values[k]),k
    scale=FINE/f'paired_seed{seed}/input_scale'
    scale_manifest=read(scale/'manifest.json')
    for name,h in scale_manifest['files'].items():assert sha(scale/name)==h
    with np.load(scale/'input_sizes_per_GT.npz',allow_pickle=False) as z:
        assert np.array_equal(z['annotation_id'],fine_values['annotation_id'])
        min_side=z['min_side640'].copy()
    scopes={'all':np.ones(38759,dtype=bool),'COCO_small':fine_values['original_area']<=1024,
            **{f'class_{c}':fine_values['category_id']==c for c in range(10)}}
    changes={}
    for label,mask in scopes.items():
        metric=geometry.pair_metrics(degraded_values,fine_values,mask)
        metric.pop('transition_annotation_ids');changes[label]=metric
    assert changes['all']['new_native_TP']==compared['GT_new_TP_by_IoU']
    assert changes['all']['lost_native_TP']==compared['GT_lost_TP_by_IoU']
    bands={name:(min_side>=a)&(min_side<b) for name,a,b in
           [('under2',0,2),('2_to_under4',2,4),('4_to_under8',4,8),
            ('8_to_under16',8,16),('16_to_under32',16,32),('32_and_above',32,float('inf'))]}
    band_changes={}
    for label,mask in bands.items():
        mask=mask&scopes['COCO_small']
        metric=geometry.pair_metrics(degraded_values,fine_values,mask)
        metric.pop('transition_annotation_ids');band_changes[label]=metric
    return {'seed':seed,'fixed_epoch':12,'EMA':True,'baseline':fb['overall_percent'],
        'fine':fs['overall_percent'],'degraded':ds['overall_percent'],
        'fine_minus_baseline':fine_report['delta_AP_metrics_pp'],
        'degraded_minus_baseline':report['delta_AP_metrics_pp'],
        'fine_minus_degraded':compared['delta_AP_metrics_pp'],
        'native_fine_gate_passed':native_gain,
        'fine_detail_gate_passed':compared['delta_AP_metrics_pp']['APs']>0,
        'native_matching_fine_minus_degraded':compared,
        'descriptive_stage_changes_fine_minus_degraded':changes,
        'COCO_small_GT_micro_changes_by_640_input_min_side':band_changes,
        'cost':{'fine':fine_report['cost'],'degraded':report['cost']},
        'final_training_counts':{'fine':fine_report['final_training_counts'],'degraded':report['final_training_counts']},
        'provenance':{'fine_report':str(fine_report_path),'fine_report_sha256':sha(fine_report_path),
            'degraded_report':str(report_path),'degraded_report_sha256':sha(report_path),
            'fine_checkpoint_sha256':fine_done['last_checkpoint_sha256'],
            'degraded_checkpoint_sha256':done['last_checkpoint_sha256'],
            'fine_archive_sha256':fine_report['archive_sha256'],'degraded_archive_sha256':report['archive_sha256'],
            'promotion_binding':promotion}}


def main(output):
    output=Path(output);assert not output.exists()
    assert sha(PLAN)==PLAN_SHA
    seeds=[compare_seed(n) for n in (1,2)]
    assert all(s['native_fine_gate_passed'] for s in seeds)
    passed=all(s['fine_detail_gate_passed'] for s in seeds)
    report={'status':'FIXED12_FOUR_CONFIGURATION_COMPARISON_VERIFIED',
        'completed_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scientific_plan_sha256':PLAN_SHA,'seeds':seeds,
        'both_native_fine_gates_passed':True,'both_fine_detail_gates_passed':passed,
        'disposition':('EXPLORATORY_FINE_SPATIAL_FIELD_SCREEN_SUPPORTED_NO_EXTENSION' if passed else
                       'STOP_FINE_DETAIL_ATTRIBUTION_ROUTE_NOT_SUPPORTED_BOTH_SEEDS'),
        'mechanism_gate':'Fixed12 EMA APs_fine>APs_degraded for BOTH same-seed capacity controls.',
        'claim_allowed':('At this fixed12 setting, both same-seed tests favor preserving fine P2 spatial information over the predeclared pool/restore control.' if passed else
                        'Fine has small native gains, but the paired controls do not support attributing them to preserved fine P2 spatial detail in both seeds.'),
        'claim_boundary':['Exploratory12-epoch prefix of native132 schedule; no full-schedule efficacy or statistical stability established.',
                          'No independent isolation of dual queries, geometry chart, initial proposal correction or gradient conflict.',
                          'No learned-manifold, true image-phase attention, first/novel or guaranteed publication claim.',
                          'Pooling and restoration alter spatial content and statistics; this is a controlled ablation, not proof of the unique root cause.',
                          'GT-micro native all-area matching and geometry support are descriptive, not APs or COCO area-specific recall.'],
        'automatic_retry':False,'automatic_extension':False,'automatic_third_seed':False,
        'new_model_forwards':0,'new_baseline_forwards':0,'new_optimizer_updates':0,
        'nearest_prior_evidence':str(FINE/'nearest_prior_supplement_20261010.json')}
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    output.mkdir(parents=True,exist_ok=False);write(output/'report.json',report)
    with (output/'固定12轮指标.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','variant','AP','APs','AP75','small_AP75','APm','APl'])
        writer.writeheader()
        for s in seeds:
            for arm in ('baseline','fine','degraded'):
                writer.writerow({'seed':s['seed'],'variant':arm,**{k:s[arm][k] for k in ('AP','APs','AP75','small_AP75','APm','APl')}})
    lines=['两组种子：固定第12轮 EMA；数值为百分制，差值为百分点。','']
    for s in seeds:
        lines.append(f"Seed {s['seed']}")
        for arm in ('baseline','fine','degraded'):
            v=s[arm];lines.append(f"  {arm}: AP={v['AP']:.6f}, APs={v['APs']:.6f}, AP75={v['AP75']:.6f}")
        v=s['fine_minus_degraded'];lines.append(f"  fine−degraded: ΔAP={v['AP']:+.6f}, ΔAPs={v['APs']:+.6f}, ΔAP75={v['AP75']:+.6f}")
        lines.append(f"  细节机制预设判据：{'通过' if s['fine_detail_gate_passed'] else '未通过'}")
    lines += ['',('两组均支持细节分支的探索性贡献。' if passed else '两组未同时支持细节分支归因；结束此路线，不延长或追加调参。'),
              '模型参数由10,231,997增至13,806,625，增加34.9358%；fine与degraded参数量相同。',
              '两组固定12轮结果仅为探索筛选，不能宣称长期稳定、首创、学习到流形或已消除梯度冲突。',
              '完整新增/损伤GT统计、逐类变化、阶段和输入像素分解、成本、来源见同目录report.json。']
    (output/'两个种子固定12轮对照结论.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(output/'manifest.json',{'reader_files':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()},'scientific_plan_sha256':PLAN_SHA})
    print(json.dumps({'status':report['status'],'both_fine_detail_gates_passed':passed,
                      'disposition':report['disposition'],'new_model_forwards':0},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    main(p.parse_args().output)
