from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, sys
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent; B=D.parent
L=B.parent/'文献_相关原始论文_20261004'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as output:
        json.dump(x,output,ensure_ascii=False,indent=2); output.write('\n')
now=datetime.now(timezone.utc).isoformat()
prior=read(B/'83_A36原生分类头与独立指标核验回执.json')
for path,digest in prior['files_sha256'].items(): assert sha(path)==digest,path
save(D/'continuation_revalidation.json',dict(recorded_utc=now,
    previous_goal_turn_classification='NO_PROGRESS_STATUS_EXPLANATION_ONLY',
    previous_turn_action='Checked authoritative state and answered current-mainline question; no new research evidence was produced in that turn.',
    prior_A36_files_verified=len(prior['files_sha256']),
    current_action='Completed new closest-work/source audit and prepared an outcome-blind data sample; no model execution allocated.',
    goal_status='active',true_blocker=False))

verified=[]
for manifest in ['selected_source_verification.json','entry_dependency_verification.json','final_selected_source_verification.json']:
    for row in read(D/manifest)['files']:
        slug='UnmaskingTiny' if row.get('repo','').startswith('AngstCJ/') else 'PiDiViT'
        p=D/'source'/slug/row['path']; raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==row['git_blob_sha']
        verified.append(dict(slug=slug,path=row['path'],sha256=row['sha256'],git_blob_sha=blob,bytes=len(raw)))
assert len(verified)==25 and len({(r['slug'],r['path']) for r in verified})==25
for row in read(D/'download_receipt.json')['rows']:
    if 'file' in row:
        assert sha(row['file'])==row['sha256']
        assert sha(row['text_file'])==row['text_sha256']
assert sha(D/'existing_dome_probe_loader.py')=='8dc66cf817d387733a895501dc160dad703eedad9d061ce8f4ffd95296b7c1ab'

def anchor(path,needle):
    p=D/'source'/'PiDiViT'/path
    found=[i+1 for i,line in enumerate(p.read_text(encoding='utf-8').splitlines()) if needle in line]
    assert found,(path,needle)
    return dict(path=path,matching_lines=found,needle=needle)
anchors={
    'selected_meta_arch':anchor('configs/few-shot/vitl_shot10.yaml','META_ARCHITECTURE:'),
    'registered_selected_class':anchor('detectron2/modeling/meta_arch/__init__.py','from .devit_update import'),
    'selected_module_import':anchor('detectron2/modeling/meta_arch/devit_update.py','from lib.regionprop_update import'),
    'frozen_backbone':anchor('detectron2/modeling/meta_arch/devit_update.py','for p in backbone.parameters(): p.requires_grad = False'),
    'roi_align':anchor('detectron2/modeling/meta_arch/devit_update.py','roi_features = self.roi_align(patch_tokens, aug_rois)'),
    'roi_shape_before_normalization':anchor('detectron2/modeling/meta_arch/devit_update.py','roi_features = roi_features.flatten(2)'),
    'roi_normalization_axis':anchor('detectron2/modeling/meta_arch/devit_update.py','roi_features = F.normalize(roi_features, dim=-1)'),
    'prototype_projection':anchor('detectron2/modeling/meta_arch/devit_update.py','feats = roi_features.transpose(-2, -1) @ class_weights_norm.T'),
    'foreground_propagation':anchor('detectron2/modeling/meta_arch/devit_update.py','fg_output = self.rpropnet('),
    'background_propagation':anchor('detectron2/modeling/meta_arch/devit_update.py','bg_output = self.rpropnet_bg('),
    'bbox_derived_region_target':anchor('lib/regionprop_update.py','mask_w = (x >= gt_rois[:, 0].unsqueeze(1))'),
    'five_scale_fusion':anchor('lib/regionprop_update.py','multi_scale_features = torch.cat([x1, x2, x3, x4, x5]'),
    'scale_gate':anchor('lib/regionprop_update.py','scale_weights = self.scale_attention('),
    'difference_after_multiscale':anchor('lib/regionprop_update.py','diff_cd = self.pdc_cd(x)'),
    'difference_strength':anchor('lib/regionprop_update.py','createConvFunc(pdc_type , 0.875)'),
    'central_difference':anchor('lib/ops_theta.py','weights.sum(dim=[2, 3], keepdim=True) * theta'),
    'attention_readout':anchor('lib/regionprop_update.py','latent = (fused_features * region_weights).sum'),
    'class_readout':anchor('lib/regionprop_update.py','class_score = self.linear(latent)'),
    'background_prototype_source':anchor('Tools.md','coco_2017_train_panoptic_stuffonly'),
    'actual_training_entry':anchor('scripts/train.sh','python3 tools/train_net.py'),
    'selected_rpn_config':anchor('scripts/train.sh','DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml'),
}
audit=dict(recorded_utc=now,status='25_SELECTED_AUTHOR_BLOBS_REHASHED_STATIC_ONLY',
    source_blobs=len(verified),PiDiViT_source_blobs=24,UnmaskingTiny_source_blobs=1,
    author_code_imported_or_executed=False,new_model_calls=0,new_probe_fits=0,
    verified_files=verified,anchors=anchors,
    PiDiViT=dict(commit='c59a130ed357cef0a2ef859a25455a597bd37294',
        method_scope='Main abstract, introduction framing, complete selected Method3.1-3.3 text; selected experiment setup,Tables4-5 and CPDC-strength discussion; supplement table1/2 and accompanying text. Not all experiments,references or runtime.',
        location='Frozen DINOv2-L patch features -> expanded proposal ROIAlign -> prototype/feature embeddings and box-region maps -> three propagation layers; NOT a replacement global ViT encoder.',
        operators='Selected code first fuses1/3/5/7/dilated3 kernels with sigmoid scale weights, then standard/central-difference branches with softmax2 attention; theta0.875; region-weighted pooled feature -> linear class score; region-to-box.',
        supervision='Selected training region maps are rectangles rasterized from matched boxes, not required instance pixel masks. Author background-prototype recipe additionally uses COCO panoptic stuff-only data; exact recipe is not bbox-only VisDrone reproduction.',
        paper_code_caution='Paper calls projection cosine similarity; selected code normalizes [N,C,S] ROI features along S before transpose, rather than each C-dimensional spatial vector. Static axis finding, no full reproduction or effect measured.',
        cost_scope='Selected script uses frozen ViT-L/14, two-stage proposals, ROI branching. Supplement time comparison0.83->0.93sec/image is author-reported, not our benchmark or a24GB training guarantee.'),
    UnmaskingTiny=dict(commit='e9d531f227ddd577052077f3cfcc577d212a585f',
        journal='Image and Vision Computing172(2026),106026',doi='10.1016/j.imavis.2026.106026',
        read_scope='Publisher indexed abstract,highlights,introduction and selected Method/Experimental settings/Limitation excerpts only. Final full method unavailable. Anonymous OpenReview candidate same title was discovered but neither PDF nor public API could be retrieved; version equivalence unverified.',
        method_claim='YOLOX-based sparse token selection and classification-semantic-guided foreground refinement; VisDrone/UAVDT.',
        repository_state='Pinned recursive public tree has exactly one blob:README.md; README promises future full code/weights. No runnable implementation or weights verified at this commit.',
        consequence='Generic recover low-score small foreground using retained semantic information is already prior art; exact operator overlap remains unresolved without full text.'),
    structural_pdf_scope='No new PDF page anchors are claimed; sections/tables and source line anchors only. Earlier environment preflight was UNAVAILABLE, which is not PASS.')
save(D/'source_operator_audit.json',audit)

closest=[
 dict(work='PiDiViT',site='Expanded proposal ROI/spatial propagation',detail='Central difference plus multikernel fusion',conditioning='Proposal,prototype class and region maps',supervision='Boxes plus author foreground/background prototypes',boundary='Known local differential/region semantic readout; moving the same blocks to UAV is insufficient by itself.'),
 dict(work='Unmasking the Tiny',site='YOLOX foreground scores/tokens,selected publisher excerpts',detail='Sparse tokens plus class-semantic attention aggregation',conditioning='Collective classification semantics',supervision='Full details not acquired',boundary='Already recovers suppressed small-target foreground on VisDrone/UAVDT; cannot claim that generic motivation as gap.'),
 dict(work='DEIMv2 STA',site='Input-image detail CNN plus selected ViT blocks',detail='Resized ViT tokens concatenate stride8/16/32CNN features then projection',conditioning='Image-dependent,not explicit per-final-query readout in selected adapter',supervision='Original detection recipe',boundary='Generic shallow-detail supplementation already covered.'),
 dict(work='EdgeCrafter ECDet',site='Overlapping stem,late ViT block mean and resized pyramid',detail='Mean blocks10/11,resize to8/16/32,hybrid encoder',conditioning='Image-level feature pyramid before decoder',supervision='Task-specialized distillation and detection recipe',boundary='Generic compactViT/late fusion already covered; task-adapted VisDrone weights and24GB full training not verified.'),
 dict(work='SPAR-Det',site='Segmentation-guided fusion and cascade ROI expert routing',detail='Selected implementation learned flattened ROI top4-of8gate',conditioning='Region-dependent experts',supervision='Auxiliary segmentation/task and detection details in prior selected read',boundary='Region-conditioned routing not new; paper8-statistic gate differs from inspected selected source entry,not proof about every configuration.'),
 dict(work='QueryDet',site='Coarse query -> sparse high-resolution detection',detail='Fine-feature query reread and shared heads',conditioning='Sparse candidate locations',supervision='Detection',boundary='Generic coarse-to-fine query reread already covered.')]
save(D/'closest_work_comparison.json',dict(recorded_utc=now,rows=closest,
    prior_A32_audit=str(B/'A32_轻量新编码器与专家路由原文筛查_v1'/'source_operator_audit.json'),
    prior_A32_audit_sha256=sha(B/'A32_轻量新编码器与专家路由原文筛查_v1'/'source_operator_audit.json'),
    claim='These comparisons delimit known components; they do not prove a novel algorithm or a new architecture-specific failure.'))

data_root=Path(r'D:\数据集\visdrone2019')
old_val=read(B/'A35_新架构与查询生存最近工作筛查_v1'/'native_prefix_protocol_FROZEN.json')['cases']
old_val_names={r['file'] for r in old_val}; old_val_prefixes={r['prefix'] for r in old_val}
prior_train=set()
for folder in ['A33_固定可见内容的采样相位诊断_v1','A34_相位干预剩余76图复核_v1']:
    prior_train.update(r['file_name'] for r in read(B/folder/'phase_protocol_FROZEN.json')['cases'])
inventory={}; pools={}
for split in ['train','val','test-dev']:
    root=data_root/('VisDrone2019-DET-'+split)
    paths=sorted((root/'images').glob('*.jpg'))
    groups={}
    for p in paths: groups.setdefault(p.stem.split('_')[0],[]).append(p)
    anns=list((root/'annotations').glob('*.txt'))
    inventory[split]=dict(images=len(paths),annotations=len(anns),prefix_groups=len(groups),
                          image_path=str(root/'images'),annotation_path=str(root/'annotations'))
    pools[split]=groups
assert inventory['train']['images']==6471 and inventory['val']['images']==548
seed='A37-semantic-paired-20261031-v1'
def key(value): return hashlib.sha256((seed+'|'+value).encode()).hexdigest()
val_groups={g:ps for g,ps in pools['val'].items() if g not in old_val_prefixes}
train_groups={g:[p for p in ps if p.name not in prior_train] for g,ps in pools['train'].items() if g not in pools['val']}
train_groups={g:ps for g,ps in train_groups.items() if ps}
selected=[]
for split,groups,count in [('train',train_groups,64),('val',val_groups,48)]:
    assert len(groups)>=count,(split,len(groups))
    for prefix in sorted(groups,key=key)[:count]:
        image=sorted(groups[prefix],key=lambda p:key(p.name))[0]
        annotation=image.parent.parent/'annotations'/(image.stem+'.txt')
        with Image.open(image) as im: width,height=im.size
        selected.append(dict(index=len(selected),split=split,prefix=prefix,file=image.name,
                             image_path=str(image),annotation_path=str(annotation),width=width,height=height,
                             image_bytes=image.stat().st_size,image_sha256=sha(image),annotation_sha256=sha(annotation)))
train_prefixes={r['prefix'] for r in selected if r['split']=='train'}
val_prefixes={r['prefix'] for r in selected if r['split']=='val'}
assert not train_prefixes & val_prefixes
assert not val_prefixes & old_val_prefixes
assert len(selected)==112
sample=dict(recorded_utc=now,status='SAMPLE_FROZEN_BEFORE_NEW_FEATURE_OUTCOMES_NOT_AN_EXECUTION_PROTOCOL',
    seed=seed,selection='SHA256-ranked filename prefix groups,one SHA256-ranked image per group,no annotations or model scores used in sample selection.',
    split_counts={'train':64,'val':48},cases=selected,
    exclusions=dict(old_val_images=sorted(old_val_names),old_val_prefixes=sorted(old_val_prefixes),
        previously_phase_tested_train_images=len(prior_train),train_excludes_all_official_val_prefixes=True),
    blind_scope='Held out from future probe fitting and excludes the12reused diagnostic val prefixes. Not claimed globally unseen across old project work,author checkpoint selection,or all past research.',
    prefix_scope='Filename-prefix grouping is a conservative duplicate/context control,not verified flight/video identity.',
    inventory=inventory,official_train_val_prefix_overlap=len(set(pools['train'])&set(pools['val'])),
    remote_existing_train_images=6471,selected_val_images_to_upload=48,
    selected_val_upload_bytes=sum(r['image_bytes'] for r in selected if r['split']=='val'),
    new_model_call_allocation=0,new_detector_training_allocation=0,new_probe_fit_allocation=0)
save(D/'semantic_probe_sample_FROZEN.json',sample)
draft=dict(recorded_utc=now,status='DESIGN_DRAFT_REQUIRES_INTERFACE_AND_ANALYTIC_FIXTURES_BEFORE_ALLOCATION',
    question='For the same predicted boxes,do post-fusion fine-memory features discriminate missed small foreground from matched hard background more accessibly than final query features?',
    sample_manifest=str(D/'semantic_probe_sample_FROZEN.json'),sample_sha256=sha(D/'semantic_probe_sample_FROZEN.json'),
    host='Frozen original Dome-S,HGNetv2-B0CNN+hybridencoder+nativevariable250..500querydecoder; no new ViT failure inference.',
    representations={'primary_query':'Final query feature before classification Linear,256D',
        'primary_fine':'Fine memory/value ROI sampled at the SAME predicted box,7x7bilinear mean,256D; exact memory tap not yet frozen',
        'secondary_contrast':'Same-box inner mean minus fixed surrounding-ring mean,known control not an innovation',
        'query_capacity_control':'Fixed-seed random Fourier feature query ridge512D,no model optimizer',
        'geometry_score_control':'Predicted box geometry and native class score only'},
    sampling='Primary valid classes1..10,original area<=1024,unoccluded/untruncated; one-to-one geometry query assignmentIoU>=0.5. Exclude every real annotation class1..11 and ignored areas from background candidates. Match predicted area/aspect/native score before inspecting representations. Final exact bins/matching and native-miss definition pending freeze.',
    primary_analysis='Train-only standardization and fixed-regularization weighted ridge; primary fine-average minus query AUC on heldout native-missed small foreground/background pairs. No choose-best of fine/contrast on val.',
    controls=['Query nonlinear/capacity control','Positive-only true-class prediction check; foreground discrimination alone does not establish correct-class recovery.','Paired image bootstrap with frozen heldout scores; not training-seed uncertainty.','One predetermined repeated model input and independent interpolation fixture.'],
    interpretation=['Positive linear gap means accessibility under the specified probes,not information-theoretic erasure or proven attention causality.',
        'Query capacity control closes gap -> investigate readout objective/capacity before fine-feature bypass.',
        'Both poor -> no encoder-loss attribution; raw backbone/visibility evidence still needed.',
        'Query better -> local context aggregation may help; do not force detail compensation.'],
    gates_not_yet_frozen=True,possible_later_cap={'whole_detector_forwards':113,'probe_fits_upper_bound':7,'detector_optimizer_updates':0},
    allocated_model_calls=0,allocated_probe_fits=0,allocated_detector_training_updates=0,
    unresolved_before_execution=['Exact memory and final query feature taps plus source fidelity','Matching/ignore and analytic interpolation/AUC fixtures','Fully fixed metrics,gates,bootstrap and stopping rules','Execution file hashes and remote input verification'],
    novelty_boundary='Diagnostic only. PiDiViT/UnmaskingTiny/STA/ECDet/SPAR/QueryDet must remain explicit controls. A concrete useful new constraint/coupling need not use a new primitive, but a renamed stack or low-score motivation alone is insufficient.')
save(D/'paired_semantic_probe_DRAFT.json',draft)
print(json.dumps(dict(status='SOURCE_EVIDENCE_AND_OUTCOME_BLIND_SAMPLE_READY',verified_author_blobs=25,
    train_images=64,heldout_probe_images=48,train_val_prefix_overlap=sample['official_train_val_prefix_overlap'],
    new_model_calls=0,new_probe_fits=0,new_optimizer_updates=0),ensure_ascii=False))
