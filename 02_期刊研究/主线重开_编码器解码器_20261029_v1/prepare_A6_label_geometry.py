"""Freeze a source-bound annotation-only coverage audit before target-count outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A6_稀疏编码器前景监督覆盖审计_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    assert not (OUT/'protocol.json').exists()
    meta=OUT/'dataset_geometry_metadata.json'
    assert sha(meta)=='e2f84f8fca118fabbfc870138b9f142645945fb6204cf332445b2a16b32f5e7f'
    source=ROOT/'comparison_sources_v1/Focus_DETR_operators_v1'
    receipt=json.loads((source/'source_verification.json').read_text(encoding='utf-8'))
    assert receipt['status']=='ALL7_OPERATOR_BLOBS_VERIFIED'
    for row in receipt['files']:
        assert sha(source/row['path'])==row['sha256']
    fg=source/'projects/focus_detr/modeling/foreground_supervision.py'
    protocol=dict(
        material_passport=dict(origin_skill='academic-research-suite',origin_mode='new sparse-encoder label-coverage hypothesis,annotation-only source-bound audit',
                               origin_date=datetime.now(timezone.utc).isoformat(),verification_status='FROZEN_BEFORE_A6_LABEL_COUNTS',version_label='A6_v1'),
        status='FROZEN_BEFORE_A6_LABEL_COUNTS',
        source_archive=str(ROOT/'A0_冻结模型诊断结果_v1/backup/A0_A0b_bounded_diagnostics.tar.gz'),
        source_archive_sha256='4a265de93e5b4af230b0020c07c066d02602ea8bcd2d10216ea2493e79a8887d',
        trace_manifest=json.loads((ROOT/'A3_中心修正一对一约束核验_v1/input_manifest.json').read_text(encoding='utf-8')),
        geometry_metadata_sha256=sha(meta),author_commit=receipt['commit'],author_foreground_source_sha256=sha(fg),
        source_scope='Seven official Focus-DETR operator/config files inspected;GenTargets strict-inside/offmax masks and active select_loss call verified. Isolated CPU target-definition equivalence precedes count analysis.',
        question='At a fixed sparse-encoder budget,do tiny objects receive no own foreground-positive token labels,and does this remain after nearby union support and an aspect-preserving input reference?',
        scenarios=['square640_source','aspect800_reference_source','square640_paper_overlap_sensitivity'],
        input_contract='640 is our existing diagnostic setup. Aspect reference resizes shortest edge toward800,caps longest at1333,and rounds dimensions to nearest integer. It is a conventional reference,NOT a verified reproduction of the full Focus-DETR input recipe.',
        source_target_contract='float32 XYXY -> normalized CXCYWH -> pixel XYXY roundtrip;stride center s//2+i*s;strict min(l,t,r,b)>0 and low<max(l,t,r,b)<=high.',
        source_ranges=[[-1,64],[64,128],[128,256],[256,999999]],
        paper_overlap_sensitivity=[[-1,64],[64,256],[128,512],[256,999999]],
        paper_source_difference='Paper appendix reports overlap but inspected criterion hardcodes non-overlap. Counts must not silently swap the two.',
        primary_cohort='All valid100 A0 annotation sets,4491GT;primary<8px cohort fixed by the original640 sqrt-area,not detector errors.',
        nearby_support='For every GT,check whether any binary-union P3 foreground target occurs in its nearest3x3 stride8 cells;not a proof that these tokens represent that particular instance.',
        inference='Image means with filename-prefix-group bootstrap2000,seed20261030;report pooled counts separately;no multiple-endpoint pivot.',
        gates=dict(min_primary_gt=200,min_groups=15,
                   point_void='640<8 mean>=.20 and95%lower>.10',
                   void_without_nearby='640<8 joint mean>=.10 and95%lower>.05',
                   resolution_sensitivity='Same physical640<8 cohort still has>=.10 mean void at the aspect800 reference',
                   pass_rule='All gates only prioritize a bounded sparse-encoder coverage test;do not establish novel algorithm or detector effect.'),
        novelty_controls=['RFLA already studies missing positive samples;nearest-point/RFLA transfer alone cannot be claimed new.',
                          'Dome-DETR already uses Gaussian density masks for sparse window processing;simple Gaussian foreground supervision alone is not a new encoder.',
                          'DQs/density-driven query counts already exist;this audit concerns foreground selection supervision,not increasing decoder query count.'],
        limitations=['No Focus-DETR detector inference/training or AP measured;TinyFormer prediction errors are not used as proof of Focus-DETR errors.',
                     'Unselected Focus tokens remain in full memory;no foreground-positive label does not imply feature deletion or no detection supervision.',
                     'A valid receptive field may contain the target even if the token center lies outside its box.',
                     'The100-image annotation cohort is exploratory,not a full-dataset label census.',
                     'Nearest-cell and cell-intersection rescue are mathematical controls,not evidence of novel or effective learned detection.',
                     'No implication that a mere foreground-label change is sufficient to support a paper.'],
        new_full_detector_forwards=0,new_fits=0,detector_updates=0,global_full_forwards_unchanged=185,
        scientific_mainline_validated=False)
    write(OUT/'protocol.json',protocol)
    shutil.copyfile(fg,OUT/'foreground_supervision_author.py')
    for name in ('a6_label_geometry.py','a6_author_target_equivalence.py'):
        shutil.copyfile(ROOT/name,OUT/name)
    files=[dict(file=name,sha256=sha(OUT/name),bytes=(OUT/name).stat().st_size) for name in
           ('protocol.json','a6_label_geometry.py','a6_author_target_equivalence.py','foreground_supervision_author.py','dataset_geometry_metadata.json')]
    write(OUT/'freeze_manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),files=files))
    print(json.dumps(dict(status='A6_FROZEN_BEFORE_LABEL_COUNTS',images=100,author_commit=receipt['commit'],
                          protocol_sha256=sha(OUT/'protocol.json'),new_fits=0,new_full_detector_forwards=0)))

if __name__=='__main__':
    main()
