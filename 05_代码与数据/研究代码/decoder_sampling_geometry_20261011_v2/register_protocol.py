"""Freeze a saved-output native decoder sampling diagnostic before execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))


def main():
    source=ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted/native'
    binding=read(ROOT/'英文小论文/code/fdr_tail12_20261011_v2/source_binding.json')['native_files_sha256']
    names=['engine/deim/dfine_decoder.py','engine/deim/dfine_utils.py','engine/deim/utils.py']
    for name in names:assert sha(source/name)==binding[name],name
    ast.parse((HERE/'analyze_saved_sampling.py').read_text(encoding='utf8'))
    evidence={}
    for seed in (1,2):
        packet=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1'
        report=read(packet/'verification_report_v1.json')
        plan=read(packet/'extracted/prepared/plan.json')
        recipe=plan['model_recipe']['DFINETransformer']
        assert recipe['num_points']==[3,6,3] and recipe['num_layers']==3 and recipe['hidden_dim']==256
        assert recipe['num_queries']==300 and recipe['cross_attn_method']=='default'
        raw=ROOT/f'outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1'
        capture=read(raw/f'module_candidate_panel_seed{seed}_raw_v1_metadata.json')
        raw_report=read(raw/'raw_verification_report_v1.json')
        assert raw_report['status']=='PASS_DOWNLOADED_ALL_CALLED_MODULE_RAW_OUTPUTS'
        assert raw_report['archive_sha256']==capture['sha256']
        evidence[str(seed)]=dict(target_report_sha256=sha(packet/'verification_report_v1.json'),
            raw_report_sha256=sha(raw/'raw_verification_report_v1.json'),metadata_sha256=sha(raw/f'module_candidate_panel_seed{seed}_raw_v1_metadata.json'),
            archive_sha256=capture['sha256'],archive_bytes=capture['bytes'],capture_manifest_sha256=capture['manifest_sha256'],
            checkpoint_sha256=report['identity'].get('candidate_checkpoint_sha256',read(packet/'extracted/run/training_completion.json')['last_checkpoint_sha256']))
    protocol=dict(schema='saved_native_decoder_sampling_geometry_v2',
        script_sha256=sha(HERE/'analyze_saved_sampling.py'),registration_script_sha256=sha(__file__),
        native_source_sha256={n:binding[n] for n in names},sources=evidence,
        arm='target_protect',epoch_zero_based=11,weights='EMA',
        seeds=[1,2],native_batches=[25,105],image_ids=[100,101,102,103,420,421,422,423],
        decoder_layers=[0,1,2],spatial_shapes=[[80,80],[40,40],[20,20]],num_heads=8,
        points_per_level=[3,6,3],offset_scale=0.5,reg_scale=4.0,
        references=['saved encoder_initial_boxes','FDR0 reconstructed from actual Integral0 and saved pre reference','FDR1 reconstructed from actual Integral1 and saved pre reference'],
        value_inputs='Actually called decoder.input_proj.0/1/2 outputs; native value_op has no projection or mask in this evaluation.',
        reconstruction_control=dict(final_box_abs_tolerance=1e-6,cross_attention_max_abs_tolerance=0.0025,
            cross_attention_relative_L2_tolerance=1e-5,arithmetic='NumPy FP64 reconstruction from actual GPU FP32 tensors, not a bitwise native replay; no tolerance changes after execution'),
        spatial_footprint='Lower-left index of bilinear stencil: floor(location*[W,H]-0.5). Same stencil alone does not mean constant values.',
        occupancy_boundary_ambiguity_cells=0.001,near_constant_relative_RMS=0.01,
        head_collapse_definition='All P3 points have same stencil, all stencils fully inside image, no point within0.001 cell of an integer stencil boundary, nonzero sampled energy and conditional-attention weighted RMS dispersion/value RMS <0.01.',
        query_link='One GT chooses its max-IoU pre box within this run; the fixed query slot is followed through three layers. Zero-IoU pre maxima are flagged and excluded from linked summaries. This is a GT-assisted descriptive link, not native matching.',
        failure_group='Final any-query IoU<0.5 against actual native validation XYXY target in input pixels. Not class-aware TP or AP.',
        repeated_GT_to_query_links='Per-GT descriptive records may share one query; all300-query summaries are also reported. No cross-model query alignment.',
        diagnostic_priority_rule='A concentration-based module intervention is eligible for further reasoning only if, in each seed, at least25% of eligible under4 final-failure GT links have >50% of P3 heads meeting the joint collapse definition in at least one layer. This is a precondition screen, not an AP gate or causal proof; a failed screen retires this concentration-led proposal without a parameter sweep.',
        source_models_are_previous_target_policy_not_new_FDR=True,
        no_population_manifold_information_loss_or_causal_root_claim=True,
        new_model_forwards=0,new_evaluator_calls=0,optimizer_updates=0,epochs_added=0)
    with (HERE/'protocol.json').open('x',encoding='utf8') as file:json.dump(protocol,file,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='REGISTERED_NATIVE_SAMPLING_DIAGNOSTIC_BEFORE_EXECUTION',protocol_sha256=sha(HERE/'protocol.json'),points_per_level=protocol['points_per_level'],model_forwards=0,epochs_added=0)))


if __name__=='__main__':main()
