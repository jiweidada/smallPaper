# Abstract

Small objects in unmanned aerial vehicle (UAV) imagery provide limited visual evidence amid clutter and scale variation. We present MPSMA-DEIM, a DEIM-D-FINE adaptation that combines a Multi-branch Progressive Stem (MPS) with Multi-scale Context-aware Attention (MSCA). MPS fuses detail, downsampling and high-frequency paths into stride-four features; MSCA weights backbone features using pooled or dilated context and SimAM-derived energy gates. We evaluate the n configuration on 6,471 training images and all 548 VisDrone2019 validation images using a common training recipe and a frozen local Common Objects in Context (COCO)-style protocol. At the fixed round-320 exponential moving average (EMA) endpoint, the complete model (FULL) achieves average precision (AP) of 0.2255 and small-object AP (APs) of 0.1539, exceeding the original baseline (BASE) by 3.53 and 3.23 percentage points. FULL has 7.54 million trainable parameters versus 3.74 million for BASE. The conditional stem replacement (NO_MPS) yields AP/APs of 0.2099/0.1362; removing the three MSCA gates (NO_MSCA) yields 0.2222/0.1473. The latter comparison changes both the gates and model capacity, with 3.84 million fewer parameters in NO_MSCA. Results describe single-seed runs on one local validation split under documented checkpoint continuations.

# Body

1. Introduction

Unmanned aerial vehicle (UAV) imagery poses a challenging object-detection setting because many objects occupy few pixels, overlap with neighbouring objects, and appear against complex backgrounds [1], [2]. Small-object performance depends on both the features retained during early downsampling and the context available for classification and localization. We therefore study shallow feature paths and context-dependent gates within a DEIM-D-FINE backbone.

Existing approaches combine local and contextual information in different ways. Liang et al. integrate convolutional local-context enhancement with a Swin Transformer for UAV imagery [3]. For detection transformers, DEIM improves training through dense matching and a matchability-aware loss [4], while D-FINE refines bounding-box distributions across decoder layers [5]. These methods provide the host architecture and training framework for this work. We investigate a narrower design question: how to combine a shallow branch for fine detail with context-dependent feature weighting inside this backbone.

The design combines two backbone modifications. The Multi-branch Progressive Stem (MPS) aligns detail, main and high-frequency paths, applies residual spatial weighting, and fuses them before the final stem downsampling. Multi-scale Context-aware Attention (MSCA) applies pooled-context gates at stride 16 and a dilated-context gate at stride 32. The design contribution is the placement and combination of these paths and gates within the HGNetv2 backbone of DEIM-D-FINE, using established convolution, pooling and SimAM-style weighting operations. The complete configuration also uses full-width aggregation projections and the sigmoid-weighted linear unit (SiLU) in configurable blocks. We evaluate the complete n configuration and two conditional removals under a common frozen recipe, reporting fixed-round and validation-selected accuracy, parameter counts and the associated continuation histories.

2. Related Work

Real-time detectors seek a useful balance between detection accuracy and computation. YOLOv10 uses consistent dual assignments to support inference without non-maximum suppression (NMS) [6]. DEIM [4] and D-FINE [5] instead develop the training and localization of detection transformers. MPSMA-DEIM retains the encoder and decoder design of its corresponding DEIM-D-FINE configuration and modifies the backbone. The present evaluation focuses on detection accuracy and trainable parameter count.

Feature recalibration and multi-scale aggregation are established design tools. Effective squeeze-excitation (ESE), introduced with VoVNetV2 in CenterMask [7], recalibrates channels. SimAM [8] derives neuron-wise weights from an energy function without additional learned parameters. DMF-Net pairs multi-branch detail extraction with dual-attention fusion [9], while TF-DEIM-DFINE combines a Mamba backbone, attention and a feature pyramid [10]. MPSMA-DEIM places branch fusion in the stem and contextual gates after HG-block aggregation. The pooled-context gate applies a SimAM-style rule to a mean of aligned feature maps; the dilated-context gate computes the rule on learned contextual features and combines it with channel weights. The evaluation treats module removal as a comparison conditional on the remaining backbone configuration.

3. Proposed Method

3.1 Overall Architecture

MPSMA-DEIM uses an HGNetv2-B0 backbone, a hybrid feature encoder and a D-FINE decoder trained within the DEIM framework [4,5]. Figure 1 shows the implemented n and s configurations. Both use the same B0 stage widths and modified backbone. The evaluated n configuration supplies stride-16 and stride-32 features to an encoder of width 128. The encoder has one eight-head layer; the three-layer decoder uses 300 regular queries and six sampling points per feature level. The implementation also supports an s configuration with an additional stride-8 input and width 256; it is outside the present controlled comparison. MPS and MSCA modify the backbone, while the detection losses and decoder design are inherited.



Fig. 1. Backbone and scale-dependent encoder inputs at 640 × 640. Tensor labels use channels × height × width. The dashed connection is present only in the s configuration.

3.2 Multi-branch Progressive Stem

Given an input image X with even height and width, MPS replaces the standard stem with three paths (Fig. 2). In B0, the detail path uses two stride-one 3 × 3 convolution blocks with 8 channels, then 2 × 2 average pooling. The main path uses two 16-channel 3 × 3 blocks with strides two and one. The high-frequency path forms X − AvgPool3×3(X), using stride one and padding one, then applies a 1 × 1 projection and a depthwise 3 × 3 convolution with 4 channels. Bilinear interpolation with align_corners=False aligns this path to the main path. Concatenating the three outputs produces a 28-channel feature Z at half the input resolution.

A learned spatial gate maps the concatenation from 28 to 2 channels with a 1 × 1 convolution, batch normalization and the configured activation, then to one channel with a 7 × 7 convolution and sigmoid. The weighted feature is Z · [1 + A(Z)], where · denotes element-wise multiplication throughout. Fusion applies 1 × 1 and 3 × 3 convolution blocks with channel transitions 28 → 32 → 16, followed by a stride-two 3 × 3 block with 16 channels. The output stride is four, as in the original stem. The reported configurations enable the learnable affine block (LAB): each ConvBNAct in the detail, main, fusion and final downsampling paths applies a learned scalar scale and bias after activation. The high-frequency and spatial-gate paths do not contain this block. The name progressive refers to the two downsampling steps; it does not imply that the output has a higher spatial resolution than the baseline.



Fig. 2. MPS for HGNetv2-B0 with H = W = 640 for the reported inputs. The three branches are concatenated, followed by residual weighting, fusion convolutions and a final stride-two convolution. BN denotes batch normalization.

3.3 Multi-scale Context-aware Attention

The modified backbone uses ESE, SimAM, MSCA-Lite and MSCA-Dilated in its four stages (Fig. 3). Each HG block concatenates its features, projects them directly to its output width C using one 1 × 1 ConvBNAct, and then applies its gate. The baseline uses two 1 × 1 ConvBNAct blocks with intermediate width C/2 and no separate ESE or SimAM gates; the modified backbone also changes this projection path. B0 has one block in stages 1, 2 and 4 and two in stage 3, giving two MSCA-Lite instances. Stage outputs are 64, 256, 512 and 1024 channels at strides 4, 8, 16 and 32. Stage-2 SimAM uses variance denominator HW − 1 [8]; the two MSCA variants use denominator HW, so we describe them as SimAM-style gates.



Fig. 3. Stage-specific gates in the modified B0 backbone. Each HG block aggregates concatenated features with a full-width 1 × 1 ConvBNAct followed by its gate.

MSCA-Lite (Fig. 4) takes a feature map F with C channels and spatial size H × W. Let Pk denote k × k average pooling with stride k and Uk denote bilinear interpolation back to H × W with align_corners=False. The contextual feature is

Fms = [F + U₂(P₂(F)) + U₄(P₄(F))] / 3.                                      (1)

For each channel of a feature map T, define μ(T) as its spatial mean and v(T) as the spatial mean of [T − μ(T)]². With λ = 10⁻⁴, define the gate G and the MSCA-Lite output as

G(T) = sigmoid{[T − μ(T)]² / [4(v(T) + λ)] + 0.5},   Ylite = F · G(Fms).    (2)

The statistics are computed over spatial positions without collapsing the feature map before gating. G(Fms) has shape C × H × W and weights the original feature F. MSCA-Lite adds no learned parameters, but its pooling, interpolation and element-wise operations still incur computation. Its benefit for detection is a hypothesis to be separated from the other configuration changes.



Fig. 4. MSCA-Lite. The energy gate is computed on the aligned contextual feature, and the value path is the original input F.

MSCA-Dilated (Fig. 5) uses three branches Bd with dilation d ∈ {1, 2, 4}. Each branch applies a 1 × 1 convolution from C to C/8 channels, batch normalization and a rectified linear unit (ReLU), followed by a grouped 3 × 3 convolution from C/8 to C channels and batch normalization. The group count is C/8, so each input group produces eight output channels. This is a channel-multiplying grouped convolution, not a channel-preserving depthwise convolution. Concatenation and a learned 1 × 1 projection form the contextual feature; its gate uses the same G as Eq. (2):

Cctx = Conv₁×₁(Concat[B₁(F), B₂(F), B₄(F)]),   As = G(Cctx).                 (3)

A separate channel gate uses global average pooling (GAP) of F and two 1 × 1 convolutions, reducing C to C/8 and restoring C. Their weights and biases are W₁, W₂ and b₁, b₂, respectively:

Ac = sigmoid{W₂ ReLU[W₁ GAP(F) + b₁] + b₂}.                               (4)

The output weights the original input by both gates, with the channel gate broadcast over spatial positions:

Ydilated = F · As · Ac.                                                     (5)

For kernel size k and dilation d, the effective kernel size is

keff = k + (k − 1)(d − 1).                                                  (6)

With k = 3, the effective sizes are 3, 5 and 9 for d = 1, 2 and 4. All branches use stride one and padding d, preserving H × W. For stage 4, C = 1024 and the reduced width and group count are both 128. The contextual spatial gate remains channel-specific, with shape C × H × W; it is not a single H × W mask. In stage 4, the Dilated gate has 3,841,280 trainable parameters; its 3072-to-1024 context projection accounts for 3,146,752.



Fig. 5. MSCA-Dilated. The contextual and channel gates both modulate F. The reduction ratio is 8 and the dilation rates are 1, 2 and 4.

3.4 Backbone Activation

The full configuration selects SiLU, f(x) = x sigmoid(x) [11], for configurable convolution blocks in the stem and backbone stages. The internal branch and channel-gate activations of MSCA-Dilated remain ReLU. The original baseline uses ReLU in its configurable backbone blocks. Both configurations retain the same encoder and decoder activation settings. SiLU is a configuration choice, rather than a new activation mechanism.

4. Experiments

4.1 Dataset and Evaluation

We use the VisDrone2019 detection training and validation splits [2], with 6,471 and 548 images, respectively, across ten categories. Inputs are resized to 640 × 640. The frozen converted annotations contain 343,205 training boxes and 38,759 validation boxes. All validation images are retained, including two byte-identical images with different annotation records. The same data files are used in every arm.

Evaluation follows a local bounding-box protocol based on Common Objects in Context (COCO) [12]: intersection-over-union (IoU) thresholds 0.50–0.95 in steps of 0.05, 101 recall thresholds, class-aware matching and maxDets=[1, 10, 100] per image and category. The postprocessor returns 300 predictions per image across categories. Average precision (AP) averages the ten IoU thresholds; AP50 and AP75 use 0.50 and 0.75. Small-object AP (APs) uses original-image box area up to 1,024 square pixels. Average recall (AR) is reported at maxDets=100 as AR@100. Its large-object counterpart, ARl, uses original-image area of at least 9,216 square pixels.

The converted annotations have iscrowd=0 and no ignore field. The 1,378 ignored-region entries and 32 entries labelled others in the original validation annotations are absent. The official VisDrone toolkit filters detections using ignored regions and uses an AP detection limit of 500 [13]. The local protocol therefore differs from the official challenge evaluation, and the reported values are not used to rank against official test-dev results.

4.2 Training and Endpoint Protocol

BASE is the original n detector with its standard stem, two-convolution aggregation paths and ReLU backbone activations. FULL is the complete n configuration described in Section 3, including its aggregation projections and activation choices. Both use seed 0, 32-bit floating point (FP32) with automatic mixed precision and TensorFloat-32 (TF32) disabled, batch size 4, and one NVIDIA RTX 3090 graphics processing unit (GPU) per run. The software versions are PyTorch 2.5.1+cu124, CUDA 12.4 and faster-coco-eval 1.8.0. Pretrained-backbone loading and tuning are disabled. A predefined set of 554 shared encoder, decoder, stage-downsampling and HG-block tensors has identical initial values. The first 100 training input batches were checked for exact agreement.

Each retained trajectory contains 320 completed rounds with 1,617 optimizer update attempts per round, or 517,440 attempts. The training loader drops its final incomplete batch. We use one-based round numbers. AdamW uses betas (0.9, 0.999), base learning rates of 4 × 10⁻⁴ for the backbone and 8 × 10⁻⁴ otherwise, weight decay 10⁻⁴, and the frozen zero-decay normalization and encoder/decoder bias groups. The gradient norm is clipped at 0.1. Learning rates are assigned after each update: the first update uses the base rate divided by 500, followed by the selected quadratic warm-up over 2,000 steps. The rates remain at their base values after warm-up because lr_gamma=1.0. The exponential moving average (EMA) uses decay 0.9999 and a ramp time constant of 1,000 updates.

Mosaic and batch MixUp each have selection probability 0.5 in rounds 5–156. Photometric distortion, zoom-out and IoU cropping are eligible in rounds 5–296; Mosaic-selected samples skip zoom-out and IoU cropping. From round 297, these strong augmentations are disabled, while horizontal flipping and resizing remain. Batch multi-scale resizing is disabled. The native late-stage policy restores the best-AP checkpoint from the first 296 rounds at stage entry and after a later round that does not set a new global best. This policy is identical in all arms.

The primary endpoint is the EMA evaluated at actual round 320 before any subsequent restore. The secondary endpoint is the highest-AP round within each run, choosing the earliest round on ties. APs and all other metrics follow that same AP-selected checkpoint. Differences are computed from unrounded metrics. Stored endpoint weights, prediction files, evaluator arrays and metric rows are bound by hashes and were checked together.

4.3 Complete Configuration Results

Table 1. Fixed-round primary endpoints and validation-selected secondary endpoints. AP and recall values are fractions; rounds are one-based.

At the primary endpoint, FULL reaches AP 0.2255 and APs 0.1539, compared with BASE values of 0.1902 and 0.1215. The differences are 3.53 and 3.23 percentage points, respectively. The secondary best-AP rounds are 303 for BASE and 305 for FULL. These results describe the complete changed backbone configuration under the common local recipe.

FULL has 7,540,942 trainable parameters, versus 3,737,611 for BASE (2.02 times). Table 1 compares the complete configuration, including its ESE/SimAM gates, aggregation paths and activation choices. The gain cannot be assigned to MPS or MSCA alone.



Fig. 6. BASE and FULL validation curves over all 320 retained rounds, without smoothing. Circles mark round 320; stars mark the highest-AP checkpoint and its corresponding APs. Shading covers late-restoration rounds 297–320. BASE continued after rounds 214 and 308; FULL after round 1.

Figure 6 shows sharp validation drops in the observed BASE trajectory, while the observed FULL trajectory is smoother. These curves document this pair of runs. Different continuation histories and a single seed prevent an architectural claim about training stability or a prediction of run-to-run variability.

4.4 Conditional Module Removal

Two additional runs examine removals from FULL under the same frozen data, optimizer, augmentation, evaluation and 320-round schedule. NO_MPS replaces only MPS with the original BASE StemBlock using SiLU and retains the remaining FULL structure. NO_MSCA replaces the two MSCA-Lite gates and one MSCA-Dilated gate with identities, retaining their aggregation projections, ESE, SimAM, MPS and SiLU. All retained tensors start from their corresponding FULL initial values; the replacement stem starts from the original BASE stem tensors. The trainable parameter counts are 7,535,617 and 3,699,662, respectively.

Table 2. Conditional module-removal results at the fixed round-320 endpoint. NO_MPS replaces MPS; NO_MSCA removes the three MSCA gates. Metrics are AP/APs/AR@100/ARl fractions; trainable parameters are in millions.

At fixed round 320, FULL exceeds NO_MPS by 1.57 percentage points in AP and 1.77 in APs (Table 2). Round 320 is also the NO_MPS trajectory's AP-best checkpoint, so its primary and secondary endpoints coincide.

FULL has AR@100 of 0.3743 versus 0.3621 for NO_MPS. Its large-object recall is lower: ARl is 0.5446 versus 0.5615, a difference of −1.69 percentage points.

Replacing the stem removes 5,325 trainable parameters while retaining the other FULL components. The stem operator and realized training histories both differ, so this result describes the conditional comparison of these runs and provides no estimate of variability across seeds.

At round 320, FULL exceeds NO_MSCA by 0.33 percentage points in AP, 0.66 in APs and 0.28 in AR@100 (Table 2). This accuracy difference accompanies 3,841,280 additional trainable parameters in FULL, an increase from 3.70 million to 7.54 million. The comparison therefore combines gate removal with a substantial capacity change. The NO_MSCA AP-best secondary endpoint is round 302 (AP 0.2253), selected on the same validation split. BASE lacks the ESE/SimAM gates and differs in aggregation paths and activations, so it does not provide a joint-removal control. The design provides no MPS–MSCA interaction estimate.

4.5 Limitations

One seed and one validation split provide no estimate of run-to-run variability or independent test performance. The validation set guides restoration and secondary checkpoint selection. Model capacity differs among configurations, notably between FULL and NO_MSCA. Latency and onboard performance for the current n model remain unmeasured.

BASE resumed after round 214, excluding 11 later rows from the interrupted branch; FULL resumed after round 1. Both anchors lacked main-process random states and were reseeded with 0. BASE resumed again after round 308 with saved random states, rerunning a partial round 309. NO_MPS began at round zero. The FULL per-batch learning-rate record begins at round 2.

NO_MSCA resumed from complete checkpoints after rounds 29 and 164 on replacement RTX 3090 instances, restoring model and EMA weights, optimizer and warmup state, and all saved random states. Partial work in round 165 was discarded after the second interruption. The full amount of unretained work across the interruptions is unknown; retained round counts therefore do not establish equal physical training computation or bitwise equivalence to uninterrupted training.

5. Conclusion

MPSMA-DEIM integrates a three-path stem and stage-specific contextual gates into the DEIM-D-FINE backbone. At the fixed round-320 EMA endpoint, FULL achieves AP 0.2255 and APs 0.1539, exceeding BASE by 3.53 and 3.23 percentage points with 2.02 times as many trainable parameters. FULL exceeds NO_MPS by 1.57 and 1.77 points in AP and APs, while its ARl is 1.69 points lower. NO_MSCA has 3.84 million fewer parameters and AP/APs 0.33/0.66 points lower than FULL at the same endpoint. These are single-seed observations on one local validation split under documented continuation histories. Repeated seeds and controls for continuation histories and gate capacity are needed for stronger component attribution. Latency and onboard performance remain unmeasured.
