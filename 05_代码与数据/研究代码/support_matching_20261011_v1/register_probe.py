"""Freeze a new matching-cost probe before any candidate computation."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/support_matching_20261011_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
SOURCE = ROOT / 'outputs/scale_protection12_20261010_v1/terminal_collection/seed2_target_protect_v1/extracted'


def main():
    OUT.mkdir(exist_ok=False)
    inputs = {}
    for seed in (1, 2):
        base = ROOT / f'outputs/native_augmented_supervision_20261010_v1/collection_seed{seed}/capture_v1'
        for batch in range(16):
            folder = base / f'endpoint12/batch_{batch:03d}'
            m = read(folder/'manifest.json')
            for name, digest in m['files'].items():
                assert sha(folder/name) == digest
            inputs[f'{seed}/{batch}'] = dict(relative_folder=str(folder.relative_to(ROOT)),
                files=m['files'], manifest_sha256=sha(folder/'manifest.json'))
    protocol = dict(schema='support_matching_saved_probe_v1',
        frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_goal_turn='no_progress_explanation_only', previous_blocker=None,
        hypothesis='A bounded penalty for reference-incompatible assignments across all ordinary stages may route supervision to feasible queries without removing tiny-GT coverage.',
        intervention='Add cost_giou * min(1, max_edge_overflow/native_support_halfwidth) to all five original ordinary matching matrices, then use the original GO frequency union.',
        no_hyperparameter_search=True, coefficient=2, all_target_scales=True,
        reference='native pre_outputs.pred_boxes; same query order at all stages',
        sources=inputs, native_matcher_sha256=sha(SOURCE/'native/engine/deim/matcher.py'),
        native_criterion_sha256=sha(SOURCE/'native/engine/deim/deim_criterion.py'),
        matcher_config=read(SOURCE/'prepared/resolved.yml')['DEIMCriterion']['matcher'],
        source_code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        primary_cohort='All 64 saved augmented train-mode images per seed at BASE12; per-GT input shortside<4, no exclusion of images with >300GT.',
        entry_gate='Both seeds: smaller fraction of final GO under4 pair exposures outside reference support; no decrease in unique under4 GT rows receiving GO supervision. This is a manipulation/coverage gate, not efficacy.',
        secondary='Report final MAL and GO by input-size bands, original matching objective cost, current matched IoU, losses/gains of supervised GT, changes to previously high-IoU responsibilities, and GO tie sensitivity.',
        native_replay='Replay native matcher on same saved tensors. Report exact assignment identity and original CPU objective differences; CUDA or tie differences are not detector errors. No passing manipulation conclusion unless comparator replay is certified.',
        budget=dict(model_forwards=0, BASE_forwards=0, optimizer_steps=0, training_epochs=0, saved_batches_per_seed=16),
        literature=dict(title='DEIM: DETR with Improved Matching for Fast Convergence',venue='CVPR 2025',
            url='https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html',
            borrowed_reasoning='Separate matching quantity and quality; test the supervision mechanism with a counterfactual and then training.',
            this_penalty_not_claimed_from_paper=True, novelty_not_established=True),
        boundaries=['Saved frozen EMA augmented fixtures are not the actual historical optimizer trajectory.',
            'Reference-feasible does not imply correct class, attained IoU, AP, or unique root.',
            'Soft penalty can change other-size assignments, GT coverage, or favor larger reference boxes.',
            'No FDR range expansion, loss-floor change, final-layer-only reassignment, new queries, or sampling translation.',
            'Formal candidate training, if justified, is a fresh configuration from the common init, max12 epochs, fixed12 EMA, both seeds APs>BASE and AP>=BASE.'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='FROZEN_BEFORE_EXECUTION',protocol_sha256=sha(OUT/'protocol.json'),batches=32)))


if __name__ == '__main__':
    main()
