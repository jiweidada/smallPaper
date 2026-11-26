"""Preserve the invalid reader and correct the observed validation GT format."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
p=HERE/'analyze_frozen_tail_delta_v1.py'
q=HERE/'analyze_frozen_tail_delta_v2.py'
assert not q.exists()
s=p.read_text(encoding='utf8')
s=s.replace('frozen_final_head_damage_seed2_v1','frozen_final_head_damage_seed2_v2')
s=s.replace('frozen_final_head_tail_delta_v1','frozen_final_head_tail_delta_v2')
s=s.replace("gt=load(target['boxes']).astype(np.float64)",
    "gt_pixels=load(target['boxes']).astype(np.float64)\n"
    "                assert (gt_pixels[:,2:]>gt_pixels[:,:2]).all() and gt_pixels.min()>=0 and gt_pixels.max()<=640.0001\n"
    "                gt=gt_pixels/640.")
s=s.replace('iou(xyxy(gt),old)','iou(gt,old)').replace('iou(xyxy(gt),new[valid])','iou(gt,new[valid])')
s=s.replace('gt[:,2:].min(1)*640','(gt[:,2:]-gt[:,:2]).min(1)*640')
s=s.replace('gt[index,2:].min()*640','(gt[index,2:]-gt[index,:2]).min()*640')
s=s.replace('_native_targets_cxcywh','_native_targets_xyxy_normalized')
s=s.replace("scale=4.0,old_final_distance_is_saved_native_Integral=True,",
    "scale=4.0,old_final_distance_is_saved_native_Integral=True,\n"
    "        native_val_target_format='absolute input-pixel XYXY, verified from raw data and resolved val transform',")
ast.parse(s)
q.write_text(s,encoding='utf8')
out=ROOT/'outputs/sod_fdr_tail_support_20261010_v2/frozen_final_head_damage_seed2_v1/withdrawn_reader_receipt.json'
with out.open('x',encoding='utf8') as f:
    json.dump(dict(status='WITHDRAWN_INVALID_VAL_TARGET_FORMAT_ASSUMPTION',reader_process_exit_code=0,
        reason='Native validation targets are absolute input XYXY, not training normalized CXCYWH. The allzero result was caught before training dispatch. Integral/box reconstruction passed; GT counts are invalid and must not be used.',
        original_files_preserved=True,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        failed_PowerShell_repair_attempt_made_no_files=True,candidate_forwards_added=0,epochs_added=0),f,indent=2)
print('SEPARATE_GT_FORMAT_READER_V2_WRITTEN')
