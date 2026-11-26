"""Freeze this architecture's one-shot engineering scope before execution."""
from pathlib import Path
import ast,datetime,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/small_object_encoder_decoder_20261010_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    files=['small_object_model.py','geometry_memory.py','preflight.py','controller.py']
    for f in files:ast.parse((HERE/f).read_text(encoding='utf8'),filename=f)
    OUT.mkdir(exist_ok=False)
    source=ROOT/'outputs/encoder_pool_diagnostic_20261010_v1/collection_seed1/capture_v1/source_identity.json'
    # Only use authoritative source/config, never a stopped arm recipe.
    config=json.loads(source.read_text(encoding='utf-8-sig'))['config']
    prior=ROOT/'英文小论文/文献_相关原始论文_20261004/112_BHCL_Decoupled_Queries_CVPR2026.pdf'
    spec={
      'status':'IMPLEMENTED_PROTOTYPE_UNVALIDATED_BEFORE_PREFLIGHT',
      'hypothesis':'Keep task-specific semantic and geometric information paths; test whether a fine spatial field with predicted-box coordinate-aware local reading improves small-box observability beyond extra decoder capacity.',
      'baseline':'DEIM-D-FINE-S / HGNetv2-B0,640x640,300queries,3decoder layers,C256,original native FDR and loss contract',
      'model_recipe':{k:config[k] for k in ['HGNetv2','HybridEncoder','DFINETransformer']},
      'criterion_recipe':config['DEIMCriterion'],
      'tensor_contract':{
        'P2':[64,160,160],'P3':[256,80,80],'P4':[512,40,40],'P5':[1024,20,20],
        'semantic_S3':[256,80,80],'semantic_S4':[256,40,40],'semantic_S5':[256,20,20],
        'semantic_flat_memory':[8400,256],'geometry_G':[64,160,160],
        'semantic_Q':[300,256],'geometry_Q':[300,256],'geometry_samples_per_query':9,
        'geometry_heads':4,'geometry_head_channels':16,'coordinate_chart_channels':8,
        'class_output':[300,10],'bbox_output':[300,4],
        'batch_dimension_omitted':True,'DN_queries_added_during_training_and_native_split_retained':True},
      'encoder':'Native semantic HybridEncoder(P3,P4,P5); G=depthwise3x3+pointwise(fine1x1+GN+SiLU(P2))+bilinear(context1x1+GN+SiLU(S3)). Return semantic pyramid and independent geometry field.',
      'proposal':'Native TopK300 ranking unchanged. Selected initial box residual additionally reads G with its selected anchor box. Output delta4 zero initialized. No labels or raw image ROI in forward.',
      'decoder':'Three semantic native layers and three independent localization layer copies. Both read native multiscale semantic memory; localization additionally reads nine local G samples with coordinate chart. Class heads read semantic state; pre-box and FDR corner heads read localization state. Native LQE intentionally still couples class scores to geometric quality.',
      'coordinate_chart':'Relative dx/dy in G-cell units(2),log1p box dimensions in G-cell units(2),sin/cos center phase for x/y(4). Ordinary Euclidean tensors; no manifold distance is learned.',
      'sampling':'3x3fixed offsets, radius max(half reference width/height,one Gcell),bilinear border,align_cornersFalse,FP32 grid/sample. Predicted/reference boxes only; no GT use.',
      'controls':{'fine':'Unpooled native stage0 spatial field.','degraded':'AvgPool2stride2 on P2 then bilinear restore160x160 before identical trainable geometry projection. All learned modules/parameter count/semantic memory/query budget/coordinate chart/losses equal.','baseline':'Existing same-seed fixed12 native prefix results, require source/data/common initialization/evaluation/recipe verification before formal comparison.'},
      'not_claimed':['The root cause is established.','A guaranteed improvement or first/novel dual decoder.','A learned hyperbolic manifold or measured curvature.','All information lost in the stem can be recovered.','Initial predictions equal means training gradients/trajectory equal.','Classification/localization fully independent: shared backbone, memory, references and LQE still couple them.'],
      'nearest_precedents':[
        {'title':'Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images','venue':'CVPR2026','source':'https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Balanced_Hierarchical_Contrastive_Learning_with_Decoupled_Queries_for_Fine-grained_Object_CVPR_2026_paper.html','local_pdf':str(prior),'local_pdf_sha256':sha(prior),'read_scope':'Abstract and section3.2/figure2 from existing extracted full text.','overlap':'Separate classification/localization queries and task-specific cross-attention/FFN already present.','difference_to_test':'P2-derived independent G field, cell coordinate chart, joint encoder initial-box and geometry-state FDR refinement; no hierarchical label contrastive objective. Difference is a hypothesis, not established novelty.'},
        {'title':'Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection','venue':'ICCV2023','source':'https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html','scope':'Historical prior outside recent2year corpus. Task-specific queries/attention are not new.'}],
      'closed_route_comparison':[
        'Plain P2 token addition was not successful; this candidate keeps semantic token budget8400 and uses a separate64-channel fine field with sparse geometry reading.',
        'Old selected P3 3x3 box residual shares the initial-box correction idea. The proposal correction alone is not new; here G is learned jointly and a separate geometry state uses it at all decoder layers.',
        'Closed raw/degraded final-output ROI uses fixed final outputs and image patches. This candidate changes jointly trained encoder/decoder states, not a post-hoc ROI refiner.',
        'Original FDR range, fixed reference chart and loss-target coding stay native; no revival of range/context experiment.',
        'No wavelet/high-frequency-loss premise or revival of interrupted wavelet arm.'],
      'remaining':['Engineering import/forward/backward/source identity preflight.','Capacity+AMP scaler/EMA/data alignment preflight before actual training.','Freeze bounded fixed12 same-seed comparison, no endpoint switching or extensions.','Only promote if APs>BASE and overall AP>=BASE; fine-vs-degraded also needed to attribute fine field usefulness.']}
    save(OUT/'design_specification.json',spec)
    protocol={'status':'FROZEN_BEFORE_FIRST_REMOTE_PREFLIGHT','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','scientific_scope':'ANALYZED_DESIGN_ONLY'},'implementation_sha256':{f:sha(HERE/f) for f in files},'design_specification_sha256':sha(OUT/'design_specification.json'),'GPU_uuid':'GPU-d29c55c5-02ab-1e69-c978-0b28f2102003','timeout_seconds':600,'budget':{'max_full_forwards':17,'max_backwards':6,'optimizer_constructions':0,'optimizer_steps':0,'epochs_added':0},'fixture':'seed1 ordinary EVAL image1,81GT,640x640, also reused under train mode/DN. Not formal augmented training input.','matrix':'native/fine/degraded x FP32/AMP x EVAL/TRAIN(12); three mature FP32 EVAL identity forwards(3); two mature AMP TRAIN manually activated gradient probes(2).','backwards':'Four common-init candidate train backwards plus two logged disposable manual activation probes. No scaler/optimizer/EMA update.','checks':['Common native state keys/dtypes/shapes/equality; geometry layer copies match loaded native layers.','Constructor CPU/CUDA/Python/NumPy RNG unchanged.','Exact initial all output tensors/losses/native BN/RNG in12case matrix; ordinary outputs300boxes and10classes and native DN/loss contract.','Mature native checkpoint initial EVAL boxes/logits exact.','At zero heads added output weights receive gradients; after explicitly logged deterministic .001*sin activation each new parameter group receives finite nonzero gradients.','Model parameters restored or discarded; all468 native source bytes unchanged.','Source/config/data/checkpoint SHA bindings and local artifact verification.'],'manual_activation':'Only disposable mature model output projection weights become .001*sin(arange+1), no optimizer. Explicitly recorded, restored then discarded. Never used as formal initialization.','success_scope':'Engineering contract only, no AP/curvature/manifold/novelty/root cause/benchmark claim.','failure_policy':'Preserve logs/failure and one-shot lock; no automatic retry or training promotion. A separately documented engineering correction can be prepared without altering failed evidence.','formal_training_launched':False}
    save(HERE/'preflight_protocol.json',protocol);save(OUT/'preflight_protocol.json',protocol)
    print(json.dumps({'status':'FROZEN','AST_files':len(files),'protocol_sha256':sha(HERE/'preflight_protocol.json')}))
if __name__=='__main__':main()
