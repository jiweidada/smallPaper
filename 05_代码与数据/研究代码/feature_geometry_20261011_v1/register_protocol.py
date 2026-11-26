"""Freeze the descriptive analysis before reading any selected feature values."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).parent
OUT=ROOT/'outputs/feature_geometry_20261011_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    stages=[['stem','backbone.stem','NCHW',160,160],['stage0','backbone.stages.0','NCHW',160,160],
        ['backbone_P3','backbone.stages.1','NCHW',80,80],['encoder_proj_P3','encoder.input_proj.0','NCHW',80,80],
        ['FPN_P3','encoder.fpn_blocks.1','NCHW',80,80],['decoder_input_P3_identity','decoder.input_proj.0','NCHW',80,80],
        ['encoder_memory_P3','decoder.enc_output','P3_first6400',80,80],
        ['backbone_P4','backbone.stages.2','NCHW',40,40],['encoder_proj_P4','encoder.input_proj.1','NCHW',40,40],
        ['FPN_P4','encoder.fpn_blocks.0','NCHW',40,40],['PAN_P4','encoder.pan_blocks.0','NCHW',40,40],
        ['backbone_P5','backbone.stages.3','NCHW',20,20],['encoder_proj_P5','encoder.input_proj.2','NCHW',20,20],
        ['attention_P5','encoder.encoder.0','Tokens400',20,20],['PAN_P5','encoder.pan_blocks.1','NCHW',20,20]]
    protocol=dict(schema='saved_feature_local_geometry_v1',seeds=[1,2],source_panel_images=8,
        native_batches=[25,105],input_size=[640,640],stages=stages,
        primary_same_resolution_comparisons=[['backbone_P3','encoder_proj_P3'],['encoder_proj_P3','FPN_P3'],['FPN_P3','encoder_memory_P3']],
        cohorts=['under4','4_to8','8_to16','16_to32','at_least32'],threshold_definition='short side in saved native XYXY input-pixel GT',
        background_radii_input_px=[16,32],angle_count=16,clearance_input_px=2,
        points='GT center and first SHA256-ranked valid point on each fixed radius circle; point must be outside all retained GT boxes dilated2, within[0.5,639.5]. Both radii required; no fallback/search.',
        background_means_uncovered_by_retained_GT_not_verified_true_background=True,
        same_points_shared_across_stages_and_seeds=True,
        sampling='Bilinear with coordinate input_position/640*feature_size-0.5, clamp edges; native token-anchor convention, not exact convolution receptive-field mapping.',
        metrics=['raw_cosine_distance','per_image_channel_STD_standardized_RMS_distance','same_nearest_anchor_cell','leave_image_out_5NN_foreground_vs_selected_GT_free_point'],
        nearest_neighbor_radius=32,k=5,kNN_max_pairs_per_image_per_cohort=64,
        neighbor_selection='SHA256-ranked GT indices, independent of values/errors; no same-image labeled neighbors; equal foreground/background labeled point counts. Per-image unlabelled channel statistics; normalize standardized feature vector to unit norm. Boundary similarity ties within1e-12 flagged/excluded.',
        summary='GT medians plus mean of per-image medians; no significance/causal or detection AP claims.',
        affine_projection='EMA Conv1x1+evalBN matrix; sampled GPU FP32 output vs CPU FP64 reconstruction tolerance2e-4+1e-3*maxabs_output, actual error reported. FP64 SVD rank uses eps64*256*sigma_max; an invertible affine map can change cosine distances without losing exact-arithmetic information.',
        exploratory_join_final_geometry='Existing saved within-forward IoU50/75 status; association only, not new matching or causal effect.',
        invalid_zero_vectors_reported=True,scalar_distance_not_information_loss_proof=True,
        dimensions_and_nonlinear_coordinate_changes_limit_cross_layer_distance_comparison=True,
        native_encoder_memory_includes_anchor_validity_mask_and_Linear_LayerNorm=True,
        no_retraining_no_new_detector_forwards=True,new_baseline_forwards=0,new_optimizer_updates=0,new_epochs=0,
        hypothetical_manifold_not_formally_identified=True,root_novelty_and_population_AP_not_claimed=True,
        code_sha256={p.name:sha(p) for p in CODE.glob('*.py')},
        raw_archive_sha256={str(s):json.loads((ROOT/f'outputs/candidate_all_module_capture_20261010_v1/seed{s}_download_v1/raw_verification_report_v1.json').read_text())['archive_sha256'] for s in (1,2)})
    OUT.mkdir(exist_ok=False)
    with (OUT/'protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status='REGISTERED_FIXED_DESCRIPTIVE_FEATURE_GEOMETRY',protocol_sha256=sha(OUT/'protocol.json'))))


if __name__=='__main__':main()
