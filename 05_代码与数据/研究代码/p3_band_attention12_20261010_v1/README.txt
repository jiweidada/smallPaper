P3 band attention — isolated fixed-12 native experiment

This package does not launch itself. Four frozen plans cover wavelet/spatial and
seeds 1/2. Run wavelet screening first. Spatial is a matched mechanism control
and requires the separately reviewed wavelet screen acceptance; the package
never automatically launches another arm. Baseline is for zero-update engineering
checks and historical same-seed epoch12 comparison, not new baseline training.

Model contract: P3BandAttentionDEIM(recipe, arm), attention_arm, band_attention.
The confirmed attention_contract.json is copied into every plan and included in
each prepared identity. Haar and four spatial sampling phases share the same
19,232 trainable parameters, initialization and gate. Spatial phase is not
Fourier phase. The source code does not implement the prior stopped feedback.

Initialization and data pairing:
- Same bound common-init bytes (SHA256 85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91).
- Strict native state names/shapes/dtypes; only band_attention.* may be new.
- New module initialization uses isolated CPU fork_rng; native model construction
  order and every native tensor are checked by the zero-update model preflight.
- Native seed setup occurs once before constructing the model and native loader.
  No preflight runs inside the formal training process. New forward has no random
  draws. The first100_training_inputs.jsonl encoding matches archived BASE/P3
  audit encoding, allowing actual input/target/RNG comparison, not a seed-only claim.
- Native criterion, optimizer, augmentation and LR recipe stay at epoches=132.
  Training runs only the original prefix 0..11. Native GradScaler may skip steps;
  true optimizer steps and attempts are recorded separately. Resume, repeated
  epoch entry and whole-epoch AMP rollback are forbidden; failures are preserved.

The generic experiment.py check and prepare commands are static and import no
torch. train --bundle is the only explicit training entrypoint, refuses existing
output, and supports no resume option. Execution authorization and idle-device
checks belong to the separate reviewed controller. This package delivery alone
is not authorization to run it.

results.py captures the existing native final EMA evaluation without another
forward. It retains full precision/recall, native GT hits, and matching records.
stage_trace.py passively captures original encoder/pre/final tensors. Its compare
command expects all four completed attention endpoints, reports wavelet minus
spatial within each seed, and is not the primary wavelet-versus-historical-BASE
screen acceptance. Independent shared-protocol endpoint reload and cost review
remain external requirements; a file-complete run is not a scientific success.
