"""Package reviewed code/source, never the credentials or redundant weights."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

D = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    spec = {
        'status': 'METHOD_FIXED_BEFORE_CAPACITY_AND_FORMAL_TRAINING',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'user_authorization': 'User explicitly requested setting a goal and running fresh native/regional-distillation/instance-query-supervision each12epochs; implementation, transfer, bounded capacity and sequential training authorized.',
        'skill_scope': 'ARS experiment-agent run, single assistant. User execution request overrides generic command-only or confirmation guidance; no third-party messaging.',
        'source_pin': 'd8752b75c9beaecfcdd0e5cd207bae3b6946ae23',
        'arms': ['B', 'M', 'Q'], 'epochs_per_group': 12, 'seed': 20261101,
        'microbatch': 8, 'gradient_accumulation': 1, 'input_hw': [640, 640],
        'common_init_sha256': '2c6e0f9c6765fc7ba069c2292e0884bb86924d2232a952499fcd251cc57b47e6',
        'teacher_checkpoint_sha256': 'bbd5712fe42bc7b11e3b01d46e930f7baa3992dd76832a913b367a58626a6ad6',
        'teacher_selection': 'A42 original B fixedepoch12 EMA.module; native10class ECDet-S, eval(), all requires_gradFalse, no_grad BF16 forward before student; no large-teacher retraining',
        'teacher_matching': 'Existing native final-output Hungarian matcher on the teacher final predictions; same teacher/input/GT-matched queries for M/Q. All matched real instances, no confidence/size filtering; DNqueries absent in teacher eval. GT beyond300query capacity remains unmatched and is counted.',
        'student_region': 'Existing stride4 stem convs[1] Bx48x160x160; GT3x3 ROIAlign sampling_ratio2 alignedTrue inFP32, meanpool, Linear48->192 then L2norm eps1e-6',
        'M_target': 'Same frozen teacher hybrid-encoder output[1] Bx192x40x40, sameGT3x3 ROIAlign mean, L2norm eps1e-6, stopgradient',
        'Q_target': 'Same frozen teacher matched final query = final classLinear input Bx300x192, L2norm eps1e-6, stopgradient',
        'distillation_loss': 'Mean1-cosine over matched instances in a batch; fixed coefficient1; M/Q both compute both frozen target types for complete paired-stream verification',
        'student_projection_initialization': 'M/Q identical nn.Linear48->192 defaultinit with seed20261102; not in native forward; all capacity weights discarded',
        'native_loss_weights': {'loss_mal': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_fgl': .15, 'loss_ddf': 1.5},
        'optimizer': {'type': 'AdamW', 'backbone_lr': 2.5e-5, 'other_and_projection_lr': 5e-4,
                      'weight_decay': 1e-4, 'norm_bn_bias_weight_decay': 0,
                      'clip_max_norm': .1, 'warmup_updates': 500,
                      'scheduler': '12epoch cosine floor0.1 after warmup'},
        'ema': {'decay': .9999, 'warmups': 1000},
        'augmentation': 'Deterministic image/epoch horizontal flip; fixed640stretch and ImageNetnorm; no mosaic/mixup',
        'precision': 'StudentBF16forward FP32nativecriterion/ROI/projector/loss; noGradScaler',
        'checkpoint_selection': 'fixed epoch12 EMA for every arm',
        'intermediate_validation_epochs': [4, 8, 12],
        'inference': 'Physically remove auxiliary module/hooks for all validation and latency; checkpoint model/ema.module contains native keys only, training auxiliary state separate',
        'metric': 'COCO-style original clippedGTarea small<=1024pixels^2; maxDet300; class0rectangle unionIoA>=.5 filtersGT/predictions; others11unscored; NOT officialVisDrone score',
        'primary': 'APs percentagepoints; APs75/overallAP also report',
        'practical_gates': {'Q_minus_B_APs_pp_min': .5, 'Q_minus_M_APs_pp_min': .3,
                            'overall_AP_drop_max_pp': .2},
        'latency': 'Native eager3090 batch1 BF16,10warm30repeats, no auxiliary hooks/projector; descriptive pairedmedian, not onboard/TensorRT',
        'capacity_plan': {'fixed_dense_batch': 8, 'no_batch_search_or_fallback': True,
                          'max_allocated_bytes': 22 * 2**30,
                          'student_forward_cap': 8, 'teacher_forward_cap': 4, 'optimizer_update_cap': 5,
                          'auxiliary_gradient_probe_cap': 2,
                          'steps': '3evaluation influencechecks, B1update, M2updates, Q2updates. NativeCPUexportstrictload and teacherfreeze checks. NoAPeval.'},
        'formal_expected_calls': {'student_forwards_each': 10159, 'teacher_forwards_B': 0,
                                  'teacher_forwards_each_M_Q': 9708, 'optimizer_updates_each': 9708},
        'data_pairing': 'All12samplerorderhashes and first8actualinputbatches per epoch across B/M/Q; rolling all-batch teacher predictions/instance assignment/MandQtargets hashes MvsQ. Does not claim every actual inputpixel byte-compared.',
        'failure_action': 'Stop on nonfinite loss/gradient or execution error, no automatic retry. If fixedendpoint practical gates fail, retire directquery-target claim with no layer/weight/teacher/threshold rescue sweep; prepare next candidate, no automatic new-direction training.',
        'research_status': 'Unverified hypothesis with modest-prior-overlap risk; no semantic bottleneck, novelty, significance or journal-readiness claim',
        'new_direction_training_after_queue': False,
    }
    (D / 'pre_capacity_spec.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    paths = list(D.glob('*.py')) + [D / n for n in ['pre_capacity_spec.json', 'local_source_preparation.json', 'train_data_manifest.json', 'val_data_manifest.json']]
    paths += [p for p in (D / 'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py', '.yml', '.yaml']]
    for p in paths:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
    manifest = {p.relative_to(D).as_posix(): sha(p) for p in sorted(set(paths))}
    archive = D / 'stage_code_source.zip'
    assert not archive.exists(), 'One-shot package; preserve existing payload'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            z.write(p, p.relative_to(D).as_posix())
        z.writestr('member_sha256.json', json.dumps(manifest, indent=2) + '\n')
    receipt = {'status': 'PACKAGED_LOCAL_NOT_RUN', 'bytes': archive.stat().st_size,
               'sha256': sha(archive), 'member_files': len(manifest), 'files_sha256': manifest,
               'redundant_weights_not_transferred': True, 'new_model_forwards': 0, 'new_optimizer_updates': 0}
    (D / 'stage_code_source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ['status', 'bytes', 'sha256', 'member_files']}))


if __name__ == '__main__':
    main()
