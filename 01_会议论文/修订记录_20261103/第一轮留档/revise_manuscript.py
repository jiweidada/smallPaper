from pathlib import Path
from copy import deepcopy
from datetime import datetime
import hashlib,json,shutil,re
from docx import Document
from docx.text.paragraph import Paragraph
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SRC=ROOT/'01_会议论文/稿件/MPSMA-DEIM_conference_EN.docx'
DEST=ROOT/'01_会议论文/稿件/MPSMA-DEIM_conference_EN_revised.docx'
EXPECTED='ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==EXPECTED,'Source changed; do not overwrite work'
backup=ROOT/'01_会议论文/稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
backup.parent.mkdir(exist_ok=True)
if not backup.exists():shutil.copy2(SRC,backup)
assert hashlib.sha256(backup.read_bytes()).hexdigest()==EXPECTED
d=Document(SRC);els=list(d.element.body);ops=[]
data=json.loads((OUT/'run_audit.json').read_text(encoding='utf-8'))
def text_of(el):return ''.join(t.text or '' for t in el.xpath('.//w:t | .//m:t'))
def fmt(p,size=10.5):
    for r in p.runs:
        r.font.name='Times New Roman';r.font.size=Pt(size);r.font.color.rgb=RGBColor(0,0,0)
    p.paragraph_format.space_after=Pt(5);p.paragraph_format.line_spacing=1.1
    p.paragraph_format.widow_control=True
def replace(i,text):
    p=Paragraph(els[i],d._body);old=text_of(els[i]);p.clear();p.add_run(text);fmt(p)
    ops.append({'source_block':i,'operation':'replace_paragraph','before':old,'after':text});return p
def after(i,text,style=None):
    e=OxmlElement('w:p');els[i].addnext(e);p=Paragraph(e,d._body)
    if style:p.style=style
    p.add_run(text);fmt(p);ops.append({'source_block':i,'operation':'insert_after','after':text});return p
def before_el(el,text,style=None):
    e=OxmlElement('w:p');el.addprevious(e);p=Paragraph(e,d._body)
    if style:p.style=style
    p.add_run(text);fmt(p);return p
def remove(i):
    ops.append({'source_block':i,'operation':'remove','before':text_of(els[i])});els[i].getparent().remove(els[i])
def comment(i,msg):
    p=Paragraph(els[i],d._body)
    if p.runs:d.add_comment(p.runs,text=msg,author='Revision evidence check',initials='EC')

changes={
0:'MPSMA-DEIM with a Multi-branch Progressive Stem and Multi-scale Context-aware Attention for UAV Small Object Detection',
5:'Small objects in unmanned aerial vehicle (UAV) images provide limited visual evidence and occur amid clutter and scale variation. We present MPSMA-DEIM, an adaptation of DEIM-D-FINE that combines a Multi-branch Progressive Stem (MPS) with Multi-scale Context-aware Attention (MSCA). The stem merges detail, downsampling and high-frequency branches before producing stride-four features. The attention design uses pooling-based context at an intermediate stage and dilated context at the deepest stage, with energy-based gates derived from SimAM. In the recorded VisDrone2019 validation runs, the complete n configuration achieves average precision (AP) of 0.2270 and small-object AP of 0.1518; the s configuration achieves AP of 0.2876. These results describe individual configurations rather than isolated component effects: the historical baseline uses a different batch size, and some variants also change the aggregation structure. An archived FP32 benchmark reports 22.572 ms for network inference plus post-processing on an RTX 3090 at 640 × 640 input. The study documents the architecture and its measured costs, while matched training controls remain necessary to establish the improvement attributable to the proposed combination.',
8:'Unmanned aerial vehicle (UAV) imagery poses a challenging object-detection setting because many objects occupy few pixels, overlap with neighbouring objects, and appear against complex backgrounds [1]. Small-object performance depends on both the features retained during early downsampling and the context available for classification and localization. These considerations motivate changes to feature extraction, but an architectural rationale alone does not establish that a particular component improves detection.',
9:'Existing approaches combine local and contextual information in different ways. Liang et al. integrate convolutional local-context enhancement with a Swin Transformer for UAV imagery [2]. For detection transformers, DEIM improves training through dense matching and a matchability-aware loss [3], while D-FINE refines bounding-box distributions across decoder layers [4]. These methods provide the host architecture and training framework for this work. We investigate a narrower design question: how to combine a shallow branch for fine detail with context-dependent feature weighting inside this backbone.',
10:'The architectural contribution of MPSMA-DEIM is the combination of two concrete modifications. The Multi-branch Progressive Stem (MPS) aligns three shallow feature paths, applies residual spatial weighting, and fuses them before the final stem downsampling. Multi-scale Context-aware Attention (MSCA) uses different gates across backbone stages, including a pooled-context gate and a dilated-context gate. The constituent operations and SimAM-style weighting are inherited techniques. We also use the sigmoid-weighted linear unit (SiLU), without claiming a new activation mechanism. We report traceable configuration-level results, distinguish the n and s protocols, and identify the comparisons that require matched controls before a causal performance claim can be made.',
12:'Real-time detectors seek a useful balance between detection accuracy and computation. YOLOv10 uses consistent dual assignments to support inference without non-maximum suppression (NMS) [5]. DEIM [3] and D-FINE [4] instead develop the training and localization of detection transformers. MPSMA-DEIM retains the encoder and decoder design of its corresponding DEIM-D-FINE scale and modifies the backbone. Results obtained with different data splits, initializations or inference implementations cannot by themselves establish a better accuracy-speed trade-off.',
13:'Feature recalibration and multi-scale aggregation are established design tools. Effective squeeze-excitation (ESE), introduced with VoVNetV2 in CenterMask [6], uses channel-wise recalibration. SimAM [7] derives neuron-wise weights from an energy function without additional learned parameters. Our pooled-context gate applies a related energy rule to a mean of aligned feature maps; the dilated-context gate computes the rule from learned contextual features and combines it with channel weights. These are adaptations of existing ideas. Their placement, value path and computational cost are the relevant architectural details, and their individual effects require controlled comparisons.',
16:'MPSMA-DEIM uses an HGNetv2-B0 backbone, a hybrid feature encoder and a D-FINE decoder trained within the DEIM framework [3,4]. Figure 1 distinguishes the two reported scales. Both use the same B0 stage widths and the same modified backbone. The n configuration supplies stride-16 and stride-32 features to an encoder of width 128, whereas the s configuration additionally supplies stride-8 features and uses width 256. Thus, the s result is a change in detector configuration, not evidence from a larger backbone alone. Within a given scale, MPS and MSCA are backbone modifications; the detection losses and decoder design are inherited.',
18:'Fig. 1. Backbone and scale-dependent encoder inputs at 640 × 640. Tensor labels use channels × height × width. The dashed connection is present only in the s configuration.',
20:'MPS replaces the standard stem with three paths (Fig. 2). For the B0 configuration, the detail path applies two stride-one 3 × 3 convolution blocks with 8 output channels, followed by 2 × 2 average pooling. The main path applies a stride-two 3 × 3 block followed by a stride-one block, both with 16 output channels. The high-frequency path forms X − AvgPool3×3(X), using stride one and padding one, then applies a 1 × 1 projection and a depthwise 3 × 3 convolution with 4 channels. Bilinear interpolation with align_corners=False aligns this path to the main path. Concatenating the three outputs produces a 28-channel feature Z at half the input resolution.',
22:'Fig. 2. MPS for HGNetv2-B0. The three branches are concatenated, followed by residual weighting, fusion convolutions and a final stride-two convolution. BN denotes batch normalization.',
24:'The four stages use ESE, SimAM, MSCA-Lite and MSCA-Dilated, respectively (Fig. 3). Each gate follows the aggregation projection inside an HG block. B0 contains one block in stages 1 and 2, two blocks in stage 3, and one block in stage 4; consequently, there are two MSCA-Lite instances. Stage outputs have 64, 256, 512 and 1024 channels at strides 4, 8, 16 and 32. The stage-2 SimAM implementation uses a variance denominator of HW − 1 [7]. The two MSCA variants below instead use the spatial mean of squared deviations, with denominator HW; they are therefore described as SimAM-style gates.',
26:'Fig. 3. Stage-specific gates within the B0 backbone. The gates operate inside feature-aggregation blocks, rather than replacing the stage convolutions.',
27:'MSCA-Lite (Fig. 4) takes a feature map F with C channels and spatial size H × W. Let Pk denote k × k average pooling with stride k and Uk denote bilinear interpolation back to H × W with align_corners=False. The contextual feature is',
28:'Fms = [F + U₂(P₂(F)) + U₄(P₄(F))] / 3.                                      (1)',
29:'For each channel of a feature map T, define μ(T) as its spatial mean and v(T) as the spatial mean of [T − μ(T)]². With λ = 10⁻⁴, define the gate G and the MSCA-Lite output as',
30:'G(T) = sigmoid{[T − μ(T)]² / [4(v(T) + λ)] + 0.5},   Ylite = F · G(Fms).    (2)',
31:'The statistics are computed over spatial positions without collapsing the feature map before gating. G(Fms) has shape C × H × W and weights the original feature F. MSCA-Lite adds no learned parameters, but its pooling, interpolation and element-wise operations still incur computation. Its benefit for detection is a hypothesis to be separated from the other configuration changes.',
33:'Fig. 4. MSCA-Lite. The energy gate is computed on the aligned contextual feature, and the value path is the original input F.',
34:'MSCA-Dilated (Fig. 5) uses three branches Bd with dilation d ∈ {1, 2, 4}. Each branch applies a 1 × 1 convolution from C to C/8 channels, batch normalization and a rectified linear unit (ReLU), followed by a grouped 3 × 3 convolution from C/8 to C channels and batch normalization. The group count is C/8, so each input group produces eight output channels. This is a channel-multiplying grouped convolution, not a channel-preserving depthwise convolution. Concatenation and a learned 1 × 1 projection form the contextual feature; its gate uses the same G as Eq. (2):',
35:'Cctx = Conv₁×₁(Concat[B₁(F), B₂(F), B₄(F)]),   As = G(Cctx).                 (3)',
36:'A separate channel gate uses global average pooling (GAP) of F and two 1 × 1 convolutions, reducing C to C/8 and restoring C:',
37:'Ac = sigmoid{W₂ ReLU[W₁ GAP(F)]}.                                           (4)',
38:'The output weights the original input by both gates, with the channel gate broadcast over spatial positions:',
39:'Ydilated = F · As · Ac.                                                     (5)',
40:'For kernel size k and dilation d, the effective kernel size is',
41:'keff = k + (k − 1)(d − 1).                                                  (6)',
42:'With k = 3, the effective sizes are 3, 5 and 9 for d = 1, 2 and 4. All branches use stride one and padding d, preserving H × W. For stage 4, C = 1024 and the reduced width and group count are both 128. The contextual spatial gate remains channel-specific, with shape C × H × W; it is not a single H × W mask.',
44:'Fig. 5. MSCA-Dilated. The contextual and channel gates both modulate F. The reduction ratio is 8 and the dilation rates are 1, 2 and 4.',
46:'The full configuration selects SiLU, f(x) = x sigmoid(x) [8], for the configurable convolution blocks in the stem and backbone stages. This does not replace every activation: the internal branch and channel-gate activations of MSCA-Dilated remain ReLU. The encoder and decoder keep their configured activations. SiLU is a configuration choice rather than a novel component, and the historical runs do not isolate its effect because the nominal SiLU variant also changes feature aggregation and batch size.',
49:'The experiments use the VisDrone2019 detection data [9], with 6,471 training images and 548 validation images covering ten categories. Validation results are reported separately from test-dev results. Figure 6 illustrates the imagery. The local training configurations load COCO-format annotations and resize images to 640 × 640. The resulting COCO-style average precision (AP) and average recall (AR) metrics [10] should not be equated with the VisDrone challenge protocol without verifying annotation conversion and ignored-region handling.',
51:'Fig. 6. Examples of VisDrone imagery.',
52:'The six recorded runs use seed 0 and contain 320 logged validation epochs, indexed 0–319. Their configurations disable pretrained-backbone loading and tuning; the nominal SiLU variant also records a resume path to its own last checkpoint. AdamW uses weight decay 10⁻⁴ and betas (0.9, 0.999). For the n runs, the configured backbone learning rate is 4 × 10⁻⁴ and the remaining rate is 8 × 10⁻⁴. The s configuration uses 2 × 10⁻⁴ for backbone parameters outside batch normalization and 4 × 10⁻⁴ otherwise; its parameter-group exclusions also differ. Most importantly, the historical n baseline uses training and validation batches of 16, whereas the other runs use batches of 4. Equal epoch counts therefore do not imply an equal number of optimizer updates.',
53:'The saved augmentation configuration includes Mosaic, photometric distortion, zoom-out, intersection-over-union (IoU) cropping, horizontal flipping and resizing, with stage boundaries [4, 156, 296]. Mosaic and mixup are configured with probability 0.5 during the main augmentation interval. Strong augmentation ends at epoch 296, and exponential moving average (EMA) is enabled. The accuracy tables use the complete metric vector from the epoch with the highest logged validation AP for each run, rather than independently selecting the best epoch for each metric. AP50 and AP75 denote AP at IoU thresholds of 0.50 and 0.75; APs denotes the reported small-object AP, and AR@100 denotes reported recall with at most 100 detections per image. This validation-selected, single-seed reporting does not provide an independent test estimate or uncertainty across training runs.',
54:'4.2 Configuration Comparisons and Attribution Limits',
55:'Table 1 identifies the five n configurations using explicit run identifiers. N0 uses the original two-convolution aggregation path (the code labels this path se). N1 and N3 instead use a full-width projection followed by ESE in every stage. N2 and N4 use the hierarchical attention arrangement of Section 3.3. Hence the labels MPS-only, MSCA-only and SiLU-only would conceal additional changes and are not used here.',
56:'Table 1. Recorded n configurations and training parameter counts. SE-path denotes the code-specific two-convolution aggregation path; it is not a claim that this path is the canonical SE attention module.',
58:'The checkpoint tensors explain the parameter discrepancy between N0 and N3. Their stem, encoder and decoder tensor shapes agree, while the four backbone-stage parameter totals increase by 4,094, 53,246, 557,052 and 983,038, respectively, for a total of 1,597,430. This matches the difference in the training logs and is consistent with replacing the aggregation path by ESE-based aggregation. An activation replacement alone would not account for this increase. N1 versus N3 also changes both the stem and activation; the available rows therefore do not isolate either factor.',
59:'Table 2 reports the logged accuracy. N4 has the highest AP, AP50, AP75, APs and AR@100 among these five configurations. Relative to N0, its observed differences are +3.75, +5.91, +3.87, +2.66 and +4.08 percentage points, respectively, calculated from the unrounded logs. These are configuration-level differences and cannot be assigned to MPS, MSCA or SiLU independently. They also do not establish synergy: the reference batch sizes differ, the aggregation paths differ, and no complete factorial comparison is available. N1 has higher large-object AP than N4 (0.3868 versus 0.3794), so the complete configuration does not lead on every metric.',
60:'4.3 Results for the n and s Configurations',
61:'Table 3 reports the complete n and s configurations on the validation split. Both use HGNetv2-B0, but the s configuration adds a stride-8 encoder input, increases the encoder and decoder width from 128 to 256, changes the decoder sampling-point allocation from [6, 6] to [3, 6, 3], and uses a different learning-rate configuration. Its higher AP therefore describes this combined scale-and-training change, rather than an isolated benefit of the proposed backbone modules.',
62:'Table 3. Validation results for the complete configurations. AP is selected within each single training run. Training parameters differ from deployed parameter counts in Table 4.',
64:'The s configuration reaches AP 0.2876 and APs 0.2123. A baseline trained under the same s protocol is needed to quantify its algorithmic gain. External test-dev results are excluded from these validation tables: differences in data split, initialization and evaluation prevent a controlled accuracy ranking. No comparison of parameter efficiency against those external accuracy values is inferred.',
66:'Table 4 retains the archived deployment-complexity measurements at 640 × 640 input and batch size one. The floating-point operation (FLOP) counts are expressed in billions (GFLOPs). These counts describe the benchmarked structures, while Tables 1 and 3 report the parameter counts logged during training. Conversion to a deployment model changes which parameters remain in the model.',
67:'Table 4. Archived complexity of the deployed s structures at 640 × 640.',
69:'The reported deployed parameter count increases from 10.18 M to 13.99 M, and the operation count from 24.86 to 29.63 GFLOPs. The archived module breakdown assigns approximately 3.84 M parameters to the single MSCA-Dilated module. MSCA-Lite and SimAM have no learned parameters; the stem increases from 6,474 to 11,799 parameters. These numbers identify where the additional capacity lies, but do not prove that this capacity is more effective than a matched alternative. Absolute FLOP counts are tool-dependent: a separate archived code-tree measurement reports 25.37 and 30.14 GFLOPs for the same two model labels.',
70:'Table 5 reports the archived FP32 benchmark on an NVIDIA RTX 3090 using 640 × 640 input, batch size one, 50 warm-up iterations and 200 timed iterations. The script uses CUDA events with synchronization and cudnn.benchmark=False. Its input is a synthetic tensor already on the GPU. The combined timing covers the network and configured postprocessor; it excludes image decoding, preprocessing, host-to-device transfer and application overhead.',
71:'Table 5. Archived RTX 3090 FP32 latency. Combined denotes network plus post-processing, not a complete image-to-result pipeline.',
73:'The reported combined latency is 22.572 ms (44.30 frames/s) for MPSMA-DEIM (s) and 19.011 ms (52.60 frames/s) for the benchmark baseline, an 18.7% increase. The baseline record specifies no checkpoint, whereas the MPSMA record specifies a checkpoint path. Because the script converts to deployment form before loading with strict=False and does not retain missing/unexpected-key counts in the JSON result, the checkpoint path alone does not establish an exact trained-model load. These are bounded structural timing measurements, not proof of onboard real-time performance or a trained-model accuracy-speed advantage. Differences between separate mean timings are not treated as direct postprocessor measurements.',
74:'4.5 Late-epoch Variation and Evidence Limits',
75:'For the s run, the last 20 logged validation AP values have a mean of 0.287065, a sample standard deviation of 0.000306 and a range of 0.001113. The best logged epoch is 302, and the final logged epoch is 319. These statistics summarize within-run variation only; they do not measure reproducibility across seeds or rule out validation-selection effects.',
78:'The available evidence supports a documented architectural design and single-run configuration results. It does not yet isolate shallow-detail preservation, contextual weighting or activation effects. The retained checkpoints also contain last_epoch metadata that differs from the best-AP log index in some runs. The available solver can reset last_epoch when restoring a stage-one checkpoint, so this field alone cannot identify the evaluation that produced a reported value. A complete reproduction requires the matching training-code version, annotation conversion, evaluator settings and checkpoint-to-evaluation linkage. Qualitative box overlays without matched thresholds and ground truth are insufficient to resolve these limitations.',
80:'MPSMA-DEIM combines a three-path shallow stem with stage-specific contextual attention in a DEIM-D-FINE detector. The implementation uses residual stem weighting, parameter-free pooled-context gates and a learned dilated-context gate, while retaining the host encoder and decoder design for each scale. The recorded n and s configurations reach validation AP of 0.2270 and 0.2876, respectively. These observations are promising enough to motivate controlled evaluation, but the historical baseline differs in batch size and aggregation structure, preventing isolated component attribution. The archived timing results describe GPU network-plus-postprocessor execution and do not establish an onboard deployment guarantee. The next empirical requirement is a matched baseline/full-model comparison with fixed data, initialization policy, training budget and evaluation, followed by targeted component controls if individual contribution claims are retained.'
}
for i,t in changes.items():replace(i,t)
after(20,'A learned spatial gate maps the concatenation from 28 to 2 channels with a 1 × 1 convolution, batch normalization and the configured activation, then to one channel with a 7 × 7 convolution and sigmoid. The weighted feature is Z · [1 + A(Z)], where · denotes element-wise multiplication throughout. Fusion applies 1 × 1 and 3 × 3 convolution blocks with channel transitions 28 → 32 → 16, followed by a stride-two 3 × 3 block with 16 channels. The output stride is four, as in the original stem. The name progressive refers to the two downsampling steps; it does not imply that the output has a higher spatial resolution than the baseline.')

for i in [76,77]:remove(i)
for i in range(82,101):remove(i)

def table_at(old,rows,widths):
    tbl=d.add_table(rows=len(rows),cols=len(rows[0]));tbl.alignment=WD_TABLE_ALIGNMENT.CENTER;tbl.autofit=False
    for col,w in zip(tbl.columns,widths):col.width=Inches(w)
    for j,row in enumerate(rows):
        for k,txt in enumerate(row):
            cell=tbl.cell(j,k);cell.width=Inches(widths[k]);cell.text=str(txt);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
            pr.append(b);m=OxmlElement('w:tcMar')
            for side,val in [('top',65),('bottom',65),('left',65),('right',65)]:
                e=OxmlElement('w:'+side);e.set(qn('w:w'),str(val));e.set(qn('w:type'),'dxa');m.append(e)
            pr.append(m)
            if j==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8EDF2');pr.append(sh)
            for p in cell.paragraphs:
                fmt(p,9);p.paragraph_format.line_spacing=1;p.paragraph_format.space_after=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next=j<len(rows)-1
                for r in p.runs:r.bold=(j==0)
        props=tbl.rows[j]._tr.get_or_add_trPr();ns=OxmlElement('w:cantSplit');props.append(ns)
        if j==0:props.append(OxmlElement('w:tblHeader'))
    if old is not None:old.addprevious(tbl._tbl);old.getparent().remove(old)
    return tbl

table_at(els[57],[['ID','Stem','Stage aggregation / gates','Activation','Batch','Params/M'],
['N0','Standard','SE-path at all stages','ReLU','16','3.7376'],
['N1','MPS','ESE at all stages','ReLU','4','5.3404'],
['N2','Standard','ESE / SimAM / Lite / Dilated','ReLU','4','7.5356'],
['N3','Standard','ESE at all stages','SiLU','4','5.3350'],
['N4','MPS','ESE / SimAM / Lite / Dilated','SiLU*','4','7.5409']],[.4,.72,2.65,.8,.5,.63])
before_el(els[58],'*SiLU applies to configurable backbone blocks; MSCA-Dilated retains internal ReLU. Parameter counts are from training logs, not deployment counts.')
before_el(els[59],'Table 2. Best-AP log rows for the n configurations. Metrics are fractions on [0,1]; epoch is zero-based. This table is not a single-factor ablation.','Caption')
rows=[['ID','Epoch','AP','AP50','AP75','APs','AR@100']]
for key in ['N0','N1','N2','N3','N4']:
    r=data[key];rows.append([key,r['best_epoch']]+[f'{r["metrics"][k]:.4f}' for k in ['AP','AP50','AP75','APs','AR100']])
t=table_at(None,rows,[.5,.65,1.05,1.05,1.05,1.05,1.05]);els[59].addprevious(t._tbl)
rows=[['Scale','Epoch','AP','AP50','AP75','APs','Train params/M']]
for k,s in [('N4','n'),('S1','s')]:
    r=data[k];rows.append([s,r['best_epoch']]+[f'{r["metrics"][x]:.4f}' for x in ['AP','AP50','AP75','APs']]+[f'{r["n_parameters"]/1e6:.4f}'])
table_at(els[63],rows,[.5,.65,.95,.95,.95,.95,1.5])
table_at(els[68],[['Model','GFLOPs','Deployed params/M'],['DEIM-D-FINE-S','24.86','10.18'],['MPSMA-DEIM (s)','29.63','13.99']],[3.2,1.5,1.9])
table_at(els[72],[['Model','Network/ms','Combined/ms','Combined FPS'],['DEIM-D-FINE-S','18.154','19.011','52.60'],['MPSMA-DEIM (s)','22.196','22.572','44.30']],[2.8,1.2,1.3,1.3])
ops.append({'operation':'replace_tables_and_insert_protocol_table','source_blocks':[57,63,68,72],'note':'Numbers from run_audit.json and archived benchmark; protocol separation explicit'})

for i,name in [(17,'fig1_architecture'),(21,'fig2_mps'),(25,'fig3_stages'),(32,'fig4_lite'),(43,'fig5_dilated')]:
    p=Paragraph(els[i],d._body);p.clear();p.add_run().add_picture(str(OUT/'figures'/f'{name}.png'),width=Inches(6.6));p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(3)
    ops.append({'source_block':i,'operation':'replace_figure','source_svg':str(OUT/'figures'/f'{name}.svg')})

refs=[
'G. Tang, J. Ni, Y. Zhao, Y. Gu, and W. Cao, “A survey of object detection for UAVs based on deep learning,” Remote Sensing, vol. 16, no. 1, Art. 149, 2024. doi: 10.3390/rs16010149.',
'W. Liang, J. Tan, H. He, H. Xu, and J. Li, “Detection of small objects from UAV imagery via an improved Swin Transformer,” in Proc. IEEE IGARSS, 2024, pp. 9134–9138. doi: 10.1109/IGARSS53475.2024.10640832.',
'S. Huang, Z. Lu, X. Cun, Y. Yu, X. Zhou, and X. Shen, “DEIM: DETR with improved matching for fast convergence,” in Proc. IEEE/CVF CVPR, 2025, pp. 15162–15171. https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html',
'Y. Peng, H. Li, P. Wu, Y. Zhang, X. Sun, and F. Wu, “D-FINE: Redefine regression task in DETRs as fine-grained distribution refinement,” in Proc. ICLR, 2025. https://openreview.net/pdf?id=MFZjrTFE7h',
'A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, “YOLOv10: Real-time end-to-end object detection,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 107984–108011. doi: 10.52202/079017-3429.',
'Y. Lee and J. Park, “CenterMask: Real-time anchor-free instance segmentation,” in Proc. IEEE/CVF CVPR, 2020, pp. 13906–13915. https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_CenterMask_Real-Time_Anchor-Free_Instance_Segmentation_CVPR_2020_paper.html',
'L. Yang, R.-Y. Zhang, L. Li, and X. Xie, “SimAM: A simple, parameter-free attention module for convolutional neural networks,” in Proc. ICML, PMLR, vol. 139, 2021, pp. 11863–11874. https://proceedings.mlr.press/v139/yang21o.html',
'S. Elfwing, E. Uchibe, and K. Doya, “Sigmoid-weighted linear units for neural network function approximation in reinforcement learning,” arXiv:1702.03118v3, 2017. doi: 10.48550/arXiv.1702.03118.',
'D. Du et al., “VisDrone-DET2019: The vision meets drone object detection in image challenge results,” in Proc. IEEE/CVF ICCV Workshops, 2019, pp. 213–226. https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ICCVW_2019_paper.pdf',
'T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in Computer Vision – ECCV 2014, LNCS, vol. 8693, 2014, pp. 740–755. doi: 10.1007/978-3-319-10602-1_48.'
]
for k,ref in enumerate(refs,1):
    p=d.add_paragraph(f'[{k}] {ref}');fmt(p,9);p.paragraph_format.left_indent=Inches(.26);p.paragraph_format.first_line_indent=Inches(-.26);p.alignment=WD_ALIGN_PARAGRAPH.LEFT

for p in d.paragraphs:
    if re.match(r'^[1-5]\. (?!\d)',p.text) or p.text in ['Abstract','References']:
        p.style='Heading 1';fmt(p,12);p.paragraph_format.keep_with_next=True
        for r in p.runs:r.bold=True
    elif re.match(r'^[1-5]\.\d ',p.text):
        p.style='Heading 2';fmt(p,11);p.paragraph_format.keep_with_next=True
        for r in p.runs:r.bold=True
    elif re.match(r'^(Fig\.\s+\d+\.|Table\s+\d+\.)',p.text):
        p.style='Caption';fmt(p,9);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=p.text.startswith('Table ')
        p.paragraph_format.keep_together=True
    elif re.match(r'^Table\s+\d+\s',p.text):
        p.style='Normal';fmt(p);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.keep_with_next=False
        for r in p.runs:r.bold=False
for i in [28,30,35,37,39,41]:
    p=Paragraph(els[i],d._body);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_together=True
    # Native Office Math container; symbols are editable in Word.
    txt=p.text;p.clear();om=OxmlElement('m:oMath');mr=OxmlElement('m:r');mt=OxmlElement('m:t');mt.text=txt;mr.append(mt);om.append(mr);p._p.append(om)
for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
    d.styles[name].font.color.rgb=RGBColor(0,0,0);d.styles[name].font.name='Times New Roman'
for style in d.styles:
    for border in list(style.element.xpath('.//w:pBdr')):
        border.getparent().remove(border)
for p in d.paragraphs:
    for border in list(p._p.xpath('./w:pPr/w:pBdr')):
        border.getparent().remove(border)
p=Paragraph(els[0],d._body);p.style='Title';fmt(p,15);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
for r in p.runs:r.bold=True
comment(16,'图1–5和方法公式依据本轮复制的当前源码、原始运行args及checkpoint张量结构统一。张量形状能核对结构，但不能证明原训练的dilation、激活、插值、lambda等无参数操作；仍需原训练源码版本或完整归档关联。不得把当前源码哈希称为训练时哈希。')
comment(52,'已从原始args确认N0 batch16，其余batch4。原稿声称全部RTX4080/Windows11的环境未由运行环境清单独立证实，正文已撤回该统一环境断言。N3保存resume到本组last，不能仅凭pretrained:false认证完整随机初始化过程。')
comment(78,'待解决：N2/N4/S1 best_stg2.pth的last_epoch分别为296/305/294，最高日志AP的epoch分别为300/306/302。当前solver存在恢复best_stg1并改last_epoch的路径，可解释元数据不可靠，但尚未证明历史运行使用完全相同版本。需要原始控制台/保存回执，或授权后重评对应权重。')
comment(69,'24.86/29.63 GFLOPs是旧分析汇总的旧树数值；另有25.37/30.14的记录。本轮未运行新profiling，也未将其认证为与精度权重同一部署构建。')
d.core_properties.comments='Content and evidence revision 2026-11-03; unresolved evidence items are recorded in comments and the revision log.'
d.save(DEST)
(OUT/'revision_patch.json').write_text(json.dumps({'source_sha256':EXPECTED,'authority':'User 2026-11-03 authorized autonomous content and evidence revision of all listed manuscript sections, including abstract; no new experiments. Venue not selected.','operations':ops,'output':str(DEST),'output_sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'contract_note':'Task-local deterministic OOXML edits; not an ARS machine-validated integrity PASS.'},ensure_ascii=False,indent=2),encoding='utf-8')
lines=[]
for el in d.element.body:
    if el.tag==qn('w:p'):lines.append(text_of(el))
    elif el.tag==qn('w:tbl'):
        for tr in el.xpath('./w:tr'):lines.append(' | '.join(text_of(tc) for tc in tr.xpath('./w:tc')))
(OUT/'revised_manuscript.txt').write_text('\n\n'.join(lines),encoding='utf-8')
(OUT/'abstract.txt').write_text(changes[5],encoding='utf-8')
# Scope headings and equation delimiters let the advisory acronym checker read body
# and abstract independently; the DOCX remains the authoritative artifact.
scope_lines=['# Abstract',changes[5],'# Body']
started=False
for p in d.paragraphs:
    if p.text.startswith('1. Introduction'):started=True
    if p.text=='References':break
    if not started:continue
    if p.style.name.startswith('Heading'):scope_lines.append('## '+p.text)
    elif p._p.xpath('.//m:oMath'):scope_lines.append('$$'+text_of(p._p)+'$$')
    else:scope_lines.append(p.text)
(OUT/'acronym_input.txt').write_text('\n\n'.join(scope_lines),encoding='utf-8')
print(json.dumps({'output':str(DEST),'backup':str(backup),'paragraphs':len(d.paragraphs),'tables':len(d.tables),'figures':len(d.inline_shapes),'words':sum(len(t.split()) for t in lines),'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()},ensure_ascii=True))
