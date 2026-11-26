"""Independent coverage recomputation from saved indices and canonical COCO."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/native_augmented_supervision_20261010_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def pair(data,e,b):return data[e['prefix']+f'.batch{b}.query_index'],data[e['prefix']+f'.batch{b}.GT_row_index']
def main():
    target=OUT/'local_analysis_verified';target.mkdir(exist_ok=False)
    ann_path=ROOT/'数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-train/annotations/train.json'
    assert sha(ann_path)=='32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883'
    raw=read(ann_path);ann={r['id']:r for r in raw['annotations']};images={r['id']:r for r in raw['images']}
    old=ROOT/'英文小论文/code/assignment_mechanism_diagnostic_20261010_v1/analyze_saved.py';spec=importlib.util.spec_from_file_location('prior_verified_geometry',old);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    per_gt=[];batches=[];stage_rows=[];arrays_verified=0;normal_losses_verified=0;max_loss_error=0.;totals=[];augment_summaries=[]
    for seed in (1,2):
        collection=OUT/f'collection_seed{seed}';capture=collection/'capture_v1';identity=read(capture/'augmentation_identity.json');source=read(capture/'source_identity.json');complete=read(capture/'completion.json')
        assert complete['model_forward_calls']==34 and complete['observer_plain_outputs_losses_BN_RNG_exact']
        for key in ['engine/deim/deim_criterion.py','engine/deim/matcher.py']:
            previous=ROOT/f'outputs/assignment_mechanism_diagnostic_20261010_v1/collection_seed{seed}/collected_native_sources'/key
            assert sha(previous)==source['native_files_sha256'][key]
        assert len(identity['batch_identity'])==16 and identity['primary_positions']==[round(i*6470/63) for i in range(64)]
        assert identity['primary_image_ids']==[sorted(images)[p] for p in identity['primary_positions']]
        assert all(r['ordinary_and_tagged_outputs_exact'] for r in identity['batch_identity'])
        ordinary_summaries={(r['endpoint'],r['batch']):r for r in read(capture/'summary.json')['samples']}
        augment_summaries.append(dict(seed=seed,primary_images=64,pre_collate_GT=sum(sum(r['pre_collate_counts']) for r in identity['batch_identity']),post_collate_GT=sum(sum(r['post_collate_counts']) for r in identity['batch_identity']),pre_images_gt300=sum(n>300 for r in identity['batch_identity'] for n in r['pre_collate_counts']),post_images_gt300=sum(n>300 for r in identity['batch_identity'] for n in r['post_collate_counts']),post_max_GT=max(n for r in identity['batch_identity'] for n in r['post_collate_counts']),mixup_batches=sum(r['mixup'][0] is not None for r in identity['batch_identity'])))
        for endpoint in (12,132):
            endpoint_rows=[]
            for b in range(16):
                p=capture/f'endpoint{endpoint}/batch_{b:03d}';evidence=read(p/'evidence.json');assert evidence['status']=='CAPTURED' and not evidence['errors'];assert evidence['actual_optimizer_steps_in_sample']==0
                with np.load(p/'arrays.npz',allow_pickle=False) as a:data={k:a[k] for k in a.files}
                assert len(data)==evidence['arrays'] and all(np.isfinite(v).all() for v in data.values());arrays_verified+=len(data)
                normal=[e for e in evidence['events'] if e['kind']=='loss_call' and not e['is_DN']];dn=[e for e in evidence['events'] if e['kind']=='loss_call' and e['is_DN']];matchers=[e for e in evidence['events'] if e['kind']=='matcher'];assert len(matchers)==5
                cls=[e for e in normal if e['loss']=='mal'];reg=[e for e in normal if e['loss']=='boxes'];assert len(cls)==len(reg)==5
                n_groups=int(data['dn_meta.num_groups']);assert n_groups>=1;input_size=evidence['metadata']['input_shape'][-1];same_identity=identity['batch_identity'][b]
                for j in range(4):
                    t=data[f'target.{j}.boxes'];labels=data[f'target.{j}.labels'];ids=data[f'target.{j}.verified_source_annotation_ids'];n=len(t);assert t.shape==(n,4) and (t[:,2:]>0).all()
                    assert n==same_identity['post_collate_counts'][j] and ids.tolist()==same_identity['source_annotation_ids'][j]
                    assert digest(t)==same_identity['boxes'][j] and labels.tolist()==same_identity['labels'][j]
                    assert labels.tolist()==[ann[int(i)]['category_id'] for i in ids]
                    mixed=f'target.{j}.mixup' in data;weights=data[f'target.{j}.mixup'] if mixed else np.ones(n,dtype=np.float32)
                    assert mixed==(same_identity['mixup'][j] is not None)
                    if mixed:assert digest(weights)==same_identity['mixup'][j]
                    votes=Counter();winning=defaultdict(list)
                    for e in matchers:
                        q,g=pair(data,e,j);assert len(q)==min(n,300) and len(set(q))==len(q)==len(set(g));assert (q>=0).all() and (q<300).all() and (g>=0).all() and (g<n).all();votes.update(zip(q.tolist(),g.tolist()))
                    for (q,g),count in votes.items():winning[q].append((g,count))
                    for e in cls:
                        me=next(x for x in matchers if x['stage']==e['stage']);q,g=pair(data,e,j);mq,mg=pair(data,me,j);assert np.array_equal(q,mq) and np.array_equal(g,mg)
                        stage_rows.append(dict(seed=seed,endpoint=endpoint,batch=b,item=j,stage=e['stage'],GT_rows=n,normal_classification_pairs=len(q),classification_GT_covered=len(set(g))))
                    union_pair=None
                    for e in reg:
                        q,g=pair(data,e,j);assert set(q)==set(winning) and len(set(q))==len(q)
                        for qq,gg in zip(q,g):assert votes[int(qq),int(gg)]==max(v for _,v in winning[int(qq)])
                        current=(q,g)
                        if union_pair is None:union_pair=current
                        else:assert np.array_equal(q,union_pair[0]) and np.array_equal(g,union_pair[1])
                    expected_dn_gt=np.tile(np.arange(n),n_groups);expected_dn_q=data[f'dn_meta.batch{j}.positive_queries'];assert len(expected_dn_gt)==len(expected_dn_q)
                    for e in dn:
                        q,g=pair(data,e,j);assert np.array_equal(q,expected_dn_q) and np.array_equal(g,expected_dn_gt)
                    def count(events):
                        result=np.zeros(n,dtype=np.int64)
                        for e in events:np.add.at(result,pair(data,e,j)[1],1)
                        return result
                    final=count([e for e in cls if e['stage']=='model']);any_cls=count(cls);normal_reg=count(reg);dn_reg=count([e for e in dn if e['loss']=='boxes']);dn_cls=count([e for e in dn if e['loss']=='mal']);all_covered=(any_cls+normal_reg+dn_reg+dn_cls)>0;normal_covered=(any_cls+normal_reg)>0
                    recomputed=dict(batch_item=j,GT_rows=n,final_cls=int((final>0).sum()),any_normal_cls=int((any_cls>0).sum()),normal_regression_union=int((normal_reg>0).sum()),DN=int((dn_reg>0).sum()),no_any_supervision=int((~all_covered).sum()),no_normal_supervision=int((~normal_covered).sum()))
                    assert recomputed==ordinary_summaries[endpoint,b]['counts'][j]
                    batches.append(dict(seed=seed,endpoint=endpoint,batch=b,item=j,input_size=input_size,mixup=mixed,**recomputed))
                    wh=t[:,2:].astype(np.float64)*input_size;short=wh.min(1)
                    for g in range(n):
                        r=dict(seed=seed,endpoint=endpoint,batch=b,item=j,GT_row=g,primary_image_id=identity['primary_image_ids'][4*b+j],source_annotation_id=int(ids[g]),source_image_id=ann[int(ids[g])]['image_id'],category=int(labels[g]),original_COCO_small=bool(ann[int(ids[g])]['area']<1024),original_area=float(ann[int(ids[g])]['area']),input_size=input_size,input_short_pixels=float(short[g]),input_size_bin='lt4' if short[g]<4 else '4to8' if short[g]<8 else '8to16' if short[g]<16 else 'ge16',mixup=mixed,mixup_weight=float(weights[g]),GT_count_in_composite=n,dense_gt300=n>300,final_cls=bool(final[g]),any_normal_cls=bool(any_cls[g]),normal_regression=bool(normal_reg[g]),DN_cls=bool(dn_cls[g]),DN_regression=bool(dn_reg[g]),no_normal_supervision=not bool(normal_covered[g]),no_any_supervision=not bool(all_covered[g]),normal_cls_positive_pairs=int(any_cls[g]),DN_cls_positive_pairs=int(dn_cls[g]))
                        per_gt.append(r);endpoint_rows.append(r)
                # Independently recompute raw L1/GIoU for each complete normal batch head.
                for e in reg:
                    raw_expected={'loss_bbox':0.,'loss_giou':0.}
                    for j in range(4):
                        q,g=pair(data,e,j);v=module.raw_loss(data[e['stage']+'.pred_boxes'][j,q],data[f'target.{j}.boxes'][g],e['num_boxes_normalizer'])
                        for k,value in v.items():raw_expected[k]+=value
                    for k,value in raw_expected.items():
                        error=abs(value-e['native_unweighted_loss_scalars'][k]);max_loss_error=max(max_loss_error,error);assert error<2e-5,(seed,endpoint,b,e['stage'],k,error)
                    normal_losses_verified+=1
            def summarize(rows):
                count=len(rows);small=[r for r in rows if r['original_COCO_small']]
                return dict(seed=seed,endpoint=endpoint,GT_rows=count,distinct_source_annotations=len({r['source_annotation_id'] for r in rows}),original_COCO_small_rows=len(small),original_area_equal1024_rows=sum(r['original_area']==1024 for r in rows),final_cls_covered=sum(r['final_cls'] for r in rows),any_normal_cls_covered=sum(r['any_normal_cls'] for r in rows),normal_regression_covered=sum(r['normal_regression'] for r in rows),DN_cls_covered=sum(r['DN_cls'] for r in rows),DN_regression_covered=sum(r['DN_regression'] for r in rows),no_normal_supervision=sum(r['no_normal_supervision'] for r in rows),no_any_supervision=sum(r['no_any_supervision'] for r in rows),small_no_final_cls=sum(not r['final_cls'] for r in small),small_no_normal_supervision=sum(r['no_normal_supervision'] for r in small),small_no_normal_fraction=sum(r['no_normal_supervision'] for r in small)/len(small),dense_image_rows=sum(r['dense_gt300'] for r in rows))
            totals.append(summarize(endpoint_rows))
    def csv_write(name,rows):
        with (target/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_write('per_augmented_GT_coverage.csv',per_gt);csv_write('per_composite_image_coverage.csv',batches);csv_write('per_stage_classification.csv',stage_rows)
    strata=[]
    for seed in (1,2):
        for endpoint in (12,132):
            rows=[r for r in per_gt if r['seed']==seed and r['endpoint']==endpoint]
            for dimension in ['input_size_bin','original_COCO_small','mixup','dense_gt300','category']:
                for value in sorted({r[dimension] for r in rows},key=str):
                    subset=[r for r in rows if r[dimension]==value];strata.append(dict(seed=seed,endpoint=endpoint,dimension=dimension,value=value,GT_rows=len(subset),no_final_cls=sum(not r['final_cls'] for r in subset),no_normal_supervision=sum(r['no_normal_supervision'] for r in subset),no_normal_fraction=sum(r['no_normal_supervision'] for r in subset)/len(subset),DN_covered=sum(r['DN_cls'] and r['DN_regression'] for r in subset),no_any_supervision=sum(r['no_any_supervision'] for r in subset)))
    save(target/'summary.json',dict(status='PASS_INDEPENDENT_SOURCE_AND_COVERAGE_RECOMPUTATION',small_definition='Original canonical source area strictly<1024, as listed in the frozen protocol; exactly1024 rows reported separately. This is not a COCO AP computation.',GT_unit='Post-augmentation target-row exposures; distinct original source annotations reported separately, not independent observations.',augmentation=augment_summaries,endpoint_totals=totals,strata=strata))
    mature=[r for r in totals if r['endpoint']==132];eligible=all(r['small_no_normal_fraction']>=.10 for r in mature)
    decision=dict(predeclared_mature_both_seed_small_no_normal_ge10percent=eligible,measured_fractions=[r['small_no_normal_fraction'] for r in mature],DN_covers_every_saved_GT=all(r['no_any_supervision']==0 and r['DN_cls_covered']==r['GT_rows'] and r['DN_regression_covered']==r['GT_rows'] for r in totals),priority='NORMAL_QUERY_CAPACITY_ELIGIBLE_FOR_CONTROLLED_INVESTIGATION' if eligible else 'NORMAL_QUERY_CAPACITY_LOWER_PRIORITY_BELOW_PREDECLARED_BOUNDARY',strong_no_any_supervision_hypothesis_supported=False if all(r['no_any_supervision']==0 for r in totals) else True,historical_root_cause_established=False,detector_AP_evidence=False,detector_training_launched=False)
    save(target/'decision.json',decision)
    verification=dict(status='PASS_NATIVE_PAIR_DN_METADATA_SOURCE_IDS_MAJORITY_UNION_COVERAGE_AND_NORMAL_BOX_LOSS_RECOMPUTATION',samples=64,composite_images=256,augmented_GT_rows=len(per_gt),arrays_verified=arrays_verified,normal_batch_head_losses_verified=normal_losses_verified,maximum_raw_box_loss_error=max_loss_error,actual_matcher_optimality_independently_solved=False,native_assignment_observer_parity=True,train_annotation_sha256=sha(ann_path),analysis_code_sha256=sha(Path(__file__)),prior_geometry_helper_sha256=sha(old),model_forwards_added=0,optimizer_steps=0,training_epochs_added=0)
    save(target/'verification.json',verification);print(json.dumps(dict(totals=totals,augmentation=augment_summaries,decision=decision,verification=verification)))
if __name__=='__main__':main()
