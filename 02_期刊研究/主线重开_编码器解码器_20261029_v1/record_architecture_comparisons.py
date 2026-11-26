from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent
now = datetime.now(timezone.utc).isoformat()
matrix = [
 {'id':'DeepSeek_CED','title':'DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression','url':'https://arxiv.org/html/2609.19969v1','tier':'preprint',
  'read_scope':'abstract; sections2.1,2.1.1,2.2,opening2.3; official DeepInfer README and system overview',
  'author_method':'Causal encoder/decoder and cross-layer KV/index reuse in language inference',
  'our_consequence':'Architecture inspiration only; static encoder value is already shared in inspected DETR decoders',
  'unresolved':'No demonstrated detection operator or tiny-object benefit transfers from this language cache design'},
 {'id':'TinyFormer','title':'TinyFormer: Preserving Tiny Objects in YOLO-DETR Hybrid Real-time Detectors','url':'https://arxiv.org/html/2605.25046v1','tier':'preprint',
  'read_scope':'sections3.1-3.2,4.3-4.4,conclusion; complete model/config source closure audited; strict author EMA inference executed',
  'author_method':'SSA shallow detail path and PBM multi-scale detail propagation',
  'our_consequence':'Plain shallow injection/P2 fusion is prior art; frozen100 diagnostics localize remaining readout gaps',
  'unresolved':'Missing candidate readout does not distinguish absent information from inadequately trained readout'},
 {'id':'FSDC_DETR','title':'FSDC-DETR: A Frequency-Spatial Domain Collaborative DETR for Small Object Detection','url':'https://arxiv.org/html/2607.05176v1','tier':'preprint; conference acceptance not independently checked here',
  'read_scope':'abstract,section2.2,sections3.1-3.2(partial); not full method or experiment',
  'author_method':'Frequency/spatial adaptation of CNN-ViT fusion, feature propagation and downsampling',
  'our_consequence':'Frequency-aware heterogeneous branch fusion is not an empty gap',
  'unresolved':'Full source and method must be read if a related encoder increment becomes the lead'},
 {'id':'UHR_DETR','title':'UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery','url':'https://arxiv.org/html/2604.21435v1','tier':'preprint',
  'read_scope':'abstract,introduction,sectionsIII-A through III-D; not full experiments',
  'author_method':'Coverage-driven sparse high-resolution patches plus global/local query decoding',
  'our_consequence':'Global semantics with sparse local detail decoding is already proposed',
  'unresolved':'Its UHR setting is not equivalent to640-square VisDrone; no cross-paper speed/AP ranking'},
 {'id':'BetterSampling','title':'Better Sampling, towards Better End-to-end Small Object Detection','url':'https://arxiv.org/pdf/2407.06127','tier':'read preprint; later chapter DOI separately discovered',
  'read_scope':'abstract,introduction,sections4.1-4.3 in primary PDF; not full experimental/appendix reading',
  'author_method':'SPR supervises outside-box offsets/attention; scale-aware targets and positive sample reweighting',
  'our_consequence':'Constrain attention sampling around small targets is existing work',
  'unresolved':'Our grid-basis hypothesis was not supported after size/initial-quality stratification'},
 {'id':'LookConfirm','title':'Look-and-Confirm: A Hierarchical Active Sampling for Reliable Aerial Search and Rescue','url':'https://doi.org/10.1109/ICIGP68997.2026.11619773','tier':'IEEE conference record',
  'read_scope':'abstract and introductory excerpt only; complete method unavailable',
  'author_method':'Hierarchical tree sampling and residual gated aggregation for clustered attention',
  'our_consequence':'Generic sampling-spread or hierarchical sampling cannot be claimed new',
  'unresolved':'Do not claim full-method novelty clearance; not the Hybrid Matching H-DETR repository'},
 {'id':'AnchorDETR','title':'Anchor DETR: Query Design for Transformer-Based Object Detection','url':'https://arxiv.org/pdf/2109.07107','tier':'AAAI2022; read primary preprint',
  'read_scope':'abstract,introduction; not complete method/experiments',
  'author_method':'Multiple query patterns per anchor location allow one region to predict multiple objects',
  'our_consequence':'Multiple slots per spatial position is prior art',
  'unresolved':'Any content-dependent subcell design needs stronger differences and equal-budget P2/pattern controls'},
 {'id':'RelationDETR','title':'Relation DETR: Exploring Explicit Position Relation Prior for Object Detection','url':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06646.pdf','tier':'ECCV2024',
  'read_scope':'abstract; complete sections4.1-4.3; selected results5.3-5.6; official position-relation and decoder source audited',
  'author_method':'Point-box normalized relative geometry feeds sinusoidal encoding and learned attention bias; contrast relation training',
  'our_consequence':'A geometric relation decoder itself is not new; source uses detached deterministic point boxes',
  'unresolved':'Effect of localization-distribution uncertainty in pairwise relations is a candidate question, not established harm'},
 {'id':'DETRNN','title':'Order Matters: Competition-Guided Query Ordering for RNN-Based Object Detection','url':'https://arxiv.org/pdf/2610.05191','tier':'NeurIPS2026 stated in primary paper and author README',
  'read_scope':'abstract,introduction,sections3.1-3.4,main-results passages; not whole24-page appendix',
  'author_method':'Confidence/similarity ordering, density-adaptive grouping and bidirectional RNN/Mamba query refinement',
  'our_consequence':'Simple query sorting, replacing self-attention with Mamba, or a density threshold is prior art',
  'unresolved':'Verified default Git branch currently contains only README; author code and weights promised after conference'},
 {'id':'DS_Det','title':'DS-Det: Single-Query Paradigm and Attention Disentangled Learning for Flexible Object Detection','url':'https://arxiv.org/pdf/2507.19807','tier':'ACMMM2025; DOI10.1145/3746027.3755045 on primary paper',
  'read_scope':'abstract,introduction,section3.4 opening/architecture figure text; full decoder method still pending',
  'author_method':'Separates box-locating cross-attention from de-duplication self-attention with flexible queries',
  'our_consequence':'Simply decouple localization and duplicate suppression is prior art',
  'unresolved':'Need complete execution/supervision details before treating it as a closest controlled baseline'},
 {'id':'UncertaintyDETR','title':'An Uncertainty-aware DETR Enhancement Framework for Object Detection','url':'https://arxiv.org/pdf/2507.14855','tier':'preprint; no venue claim made',
  'read_scope':'abstract; section3.3 and4.1-4.3; selected ablations; not whole appendix/proof audit',
  'author_method':'Gaussian bbox modeling and Bayes-risk-based output embedding, loss and matching adjustments',
  'our_consequence':'Adding a variance head or using uncertainty to reweight a query is not new',
  'unresolved':'Pairwise distribution-marginalized relation encoding differs in location, but its novelty/usefulness/calibration remain unverified'},
 {'id':'EASE_DETR','title':'EASE-DETR: Easing the Competition among Object Queries','url':'https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_EASE-DETR_Easing_the_Competition_among_Object_Queries_CVPR_2024_paper.pdf','tier':'CVPR2024',
  'read_scope':'primary search-visible abstract/results passages; direct PDF403; complete method not yet read',
  'author_method':'Leading/trailing query competition biases applied to decoder interactions',
  'our_consequence':'Protect a high-priority query via attention bias is prior art',
  'unresolved':'Full method acquisition is still required if query-competition reliability is pursued'}
]
(ROOT/'14_架构核心文献矩阵.json').write_text(json.dumps({'recorded_utc':now,'review_type':'Targeted non-systematic architecture research',
    'human_read_attestation':'none inferred','rows':matrix},ensure_ascii=False,indent=2),encoding='utf-8')
queries = [
 ['readout and nearest architectures',['"tiny object detection" "token" "decoder" "2026"','"small object detection" "query" "high-resolution" decoder 2025 2026','"DETR" "semantic" "spatial" "decoupled" "memory" small objects']],
 ['sampling mechanism and counterevidence',['"deformable attention" "small objects" "sampling points" collapse','"tiny object detection" "sampling" "radius"','"deformable attention" "bilinear interpolation" "diversity"','"Look-and-Confirm" "Hierarchical Active Sampling"']],
 ['sequence and multiplicity alternatives',['"MambaSOD" "Query-Feature" github','"Mamba" "DETR" "permutation" queries','"DETR" "multiple proposals per" feature','"DETR" "subcell" OR "sub-cell" detection transformer']],
 ['relation uncertainty closest work',['"DETR" "uncertainty" "query interaction"','"DETR" "distribution" "de-duplication"','"Relation-DETR" "uncertainty"','"DETR" "uncertainty-aware relation"','"DETR" "probabilistic positional"','"Relation DETR+" "uncertainty"']]
]
log={'recorded_utc':now,'strategy':'Search broad architecture mechanisms, verify closest primary methods, then inspect actual source before choosing experiments',
     'query_groups':[{'name':n,'queries':q} for n,q in queries],
     'inclusion':'Directly related encoder/detail propagation or decoder/query interaction mechanisms with verifiable primary source',
     'exclusion':'Unverifiable titles, unsupported third-party method descriptions, irrelevant modalities as direct detection evidence; no exhaustive/systematic-review claim',
     'raw_primary_excerpt_files':[str(p.relative_to(ROOT)) for p in (ROOT/'source_audit_v1').glob('arch_*.json')],
     'failed_or_limited_access':['EASE-DETR direct CVF PDF403; only indexed primary passages read','Look-and-Confirm full method not obtained','OpenReview multi-pattern discovery hit challenge page; unverified title excluded','A guessed arXiv2403.16226 was unrelated astrophysics, excluded and not cited'],
     'source_read_scope_matrix':'14_架构核心文献矩阵.json'}
(ROOT/'13_架构检索增补记录.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')

repos=json.loads((ROOT/'comparison_sources_v1/acquisition_receipts.json').read_text())
relation=next(r for r in repos if 'xiuqhou' in r['repo'])
repo=Path(relation['local'])
symbols=[]
for rel in ['models/bricks/relation_transformer.py','models/bricks/position_encoding.py']:
    path=repo/rel
    data=path.read_bytes()
    tree=ast.parse(data)
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ['box_rel_encoding','PositionRelationEmbedding','get_sine_pos_embed','get_dim_t','RelationTransformerDecoder']:
            symbols.append({'symbol':node.name,'file':rel,'line':node.lineno,'end_line':node.end_lineno,'file_sha256':hashlib.sha256(data).hexdigest()})
audit={'recorded_utc':now,'Relation_DETR':{'commit':relation['commit'],'verified_files':len(relation['verified_files']),'symbols':symbols,
    'observed_in_code':['Normalized center differences and width/height ratios from point boxes; eps1e-5','Sinusoidal relation encoding uses embed_dim16,temperature10000,scale100','Boxes and sinusoidal geometry read under torch.no_grad; learned1x1 projection outside','Previous/current decoder point boxes provide next-layer self-attention bias','Hybrid query training path can skip the relation prior'],
    'not_observed_in_this_operator':['Predictive covariance input','FDR probability input','Marginalization over alternative geometries'],
    'scope_limit':'This says what this specific operator consumes, not that all uncertain geometric relation methods are absent from literature'},
    'DETRNN':{'commit':repos[1]['commit'],'verified_files':len(repos[1]['verified_files']),
              'public_default_branch_scope':'README only; code/weights release promised after NeurIPS2026; not a runnable author implementation'},
    'candidate_D_verdict':'Concrete source-level difference is possible; actual harmful effect, calibrated signal and nontrivial algorithm value remain unverified',
    'new_model_forwards':0,'training_updates':0}
(ROOT/'16_最近工作源码核查.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'literature_matrix_rows':len(matrix),'query_groups':len(queries),'Relation_source_files_verified':len(relation['verified_files']),
                  'DETRNN_source_files':len(repos[1]['verified_files']),'goal':'ACTIVE','new_model_forwards':0}))
