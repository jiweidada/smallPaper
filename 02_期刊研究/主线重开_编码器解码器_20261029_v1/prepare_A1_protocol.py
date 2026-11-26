"""Freeze the exact source contract and bounded A1 protocol before observation."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json
import py_compile
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE/'A1_冻结特征读出诊断_v1'
SOURCE = BASE/'official_assets_v1/TinyFormer_git_9075d9f_v1'
NOW = datetime.now(timezone.utc).isoformat()
sys.path.insert(0,str(ROOT))
from readout_common import selfcheck,fit_readouts,predict

assert not (ROOT/'freeze_manifest.json').exists(), 'Do not silently re-freeze an executed protocol'
input_contract = json.loads((BASE/'A0_冻结模型诊断结果_v1/receipts/frozen_input_contract.json').read_text(encoding='utf-8'))
names = input_contract['file_names_in_existing_order'][:16]
groups = {name:name.split('_')[0] for name in names}
ordered_groups = sorted(set(groups.values()),key=lambda g:hashlib.sha256(('A1-group-20261030:'+g).encode()).hexdigest())
fold_for_group = {g:i%4 for i,g in enumerate(ordered_groups)}
images = [dict(file_name=n,group=groups[n],fold=fold_for_group[groups[n]]) for n in names]
assert len({r['group'] for r in images}) >= 8 and {r['fold'] for r in images} == set(range(4))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def function_dump(path,name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.dump(node,include_attributes=False)

assert function_dump(ROOT/'extract_features.py','image_tensor') == function_dump(BASE/'tinyformer_stage_probe.py','image_tensor')
(ROOT/'tinyformer_probe_loader.py').write_bytes((BASE/'tinyformer_probe_loader.py').read_bytes())
symbols = []
for rel,names_to_find in [
    ('engine/backbone/dinov3_adapter.py',{'DINOv3SSAs_4Scale'}),
    ('engine/deim/hybrid_encoder.py',{'HybridEncoder_4Scale'}),
    ('engine/deim/deim_decoder.py',{'TransformerDecoder','DEIMTransformer'}),
    ('engine/deim/deim_criterion.py',{'DEIMCriterion'})]:
    tree = ast.parse((SOURCE/rel).read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in names_to_find:
            symbols.append(dict(file=rel,symbol=node.name,line=node.lineno,end_line=node.end_lineno,sha256=sha(SOURCE/rel)))
source_contract = dict(created_utc=NOW,official_commit='9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1',
    actual_model='DEIM(backbone=DINOv3SSAs_4Scale,encoder=HybridEncoder_4Scale,decoder=DEIMTransformer)',
    actual_decoder_module='engine/deim/deim_decoder.py; this module imports shared MSDeformableAttention/LQE/Integral from dfine_decoder.py',
    actual_positional_path='deim_decoder.py computes the query-position MLP once before the decoder loop; updated reference boxes still drive each layer sampling. Do not substitute dfine_decoder.py loop semantics.',
    supervision='enc_score_head ranks all8400 tokens; selectedTop300 logits/boxes enter encoder auxiliary loss. Selected content/boxes detach before decoder; cross-attention memory still trains through decoder. Unselected all-token head readout is not a supervised representation upper bound.',
    tensors=[dict(view='P2',tap='backbone output[0] == proj_c1(sda[1](sda[0](x)))',shape=[1,192,160,160],stride=4,
                  note='Pure CNN shallow path;16 channels before pointwise projection/GELU,not a high-resolution ViT token grid'),
             dict(view='P2_down',tap='encoder.fpn_downsample_convs[1] output in idx=1 FPN iteration',shape=[1,192,80,80],stride=8,
                  note='Actual learned3x3 stride2 convolution+BN of P2,not fpn_downsample_convs[0]'),
             dict(view='P3_before',tap='backbone output[1] == proj_c2(concat(sda stride8,upsampled early ViT map))',shape=[1,192,80,80],stride=8),
             dict(view='P3_after',tap='encoder output[0] == final stride8 PBM feature before decoder enc_output',shape=[1,192,80,80],stride=8)],
    coordinate_convention='Grid-bin centers4*j+2; bilinear sampling x/640*2-1,align_corners=False. This matches normalized grid-bin convention,not a claim that every convolution receptive-field center is identical.',
    exact_symbols=symbols,original_source_numerical_edits=0,preprocessing_ast_equal_to_A0=True)
protocol = dict(material_passport=dict(origin_skill='academic-research-suite',origin_mode='run,bounded frozen-feature diagnostic',origin_date=NOW,
                                      verification_status='FROZEN_NOT_YET_EXECUTED',version='A1_linear_readout_v1'),
    status='FROZEN_BEFORE_EXTRACTION_AND_PROBE_FITS',images=images,
    reason='Distinguish finite-capacity local-center readout differences across the exact detail/downsample/fusion path; this is mechanism screening,not a novel algorithm evaluation.',
    authorization='Within continuing user authorization to investigate the mainline and run bounded remote experiments; no separate confirmation or new training queue implied.',
    budget=dict(initial_full_model_forwards=169,new_full_model_forward_cap=16,maximum_cumulative=185,existing_global_cap=200,
                main_model_optimizer_updates=0,diagnostic_readout_fits=32,real_fits=16,shuffled_label_fits=16,
                extraction_hard_timeout_seconds=180,cpu_analysis_hard_timeout_seconds=180,automatic_retry=False),
    sampling=dict(fine_grid=[160,160],bin_stride=4,coordinates='(4*column+2,4*row+2)',
                  border='only row/column2..157; excludes coarse bilinear padding effects',
                  ignore='exclude points within4 input pixels of nonpositive/ignored annotation boxes',
                  points='union of GT center bins and8 rings at row/column offsets -2,0,+2 plus512 uniformly sampled other valid bins per image; unique sorted flat IDs',
                  seed=20261030,target='max over all positive GT centers of exp(-squared_distance/8),sigma2 input pixels',
                  evaluation='GT center vs at least4 eligible ring samples; ring target <=0.1 center target; ties <=1e-7 count0.5',
                  limitation='GT-derived neighborhoods are oracle local probe regions,not a deployable detector or exhaustive center search'),
    fit=dict(type='linear ridge with intercept',input_dim=192,parameters_per_readout=193,penalty=.01,
             normalization='weighted training-fold channel mean/std; equal total weight per source image; intercept unpenalized',
             crossvalidation='4 fixed folds,filename-prefix groups assigned by deterministic hash then round-robin; group identity heuristic only',
             null='within-training-image label permutation,identical feature view,capacity,penalty and split',
             hyperparameter_search=False,epochs=None,solver='float64 NumPy closed form',feature_views=['P2','P2_down','P3_before','P3_after']),
    primary=dict(cohort='eligible GT sqrt(input area)<16',endpoint='image-averaged per-GT fraction of eligible rings scored below center',
                 contrast='P2 minus P3_after',interval='2000 filename-group cluster bootstraps,seed20261030,percentile95%; small group count remains a limitation',
                 gate=['at least50 primary GT and8 groups','primary mean difference>=0.05 and95% interval lower>0',
                       'both <8 and8-16 point contrasts nonnegative','at least30 <16 GT missing in all-encoder IoU50 readout and positive contrast in that cohort',
                       'P2 and P3_after shuffled-label primary means each<0.60'],
                 stop='If any criterion fails,no new encoder module or full training is justified by this probe. Do not enlarge the cohort or change probe capacity to seek a positive.'),
    secondary=['P2 vs P2_down','P3_before vs P3_after','scale bins<8,8-16,16-32,>=32','native encoder max-class score contrast,descriptive only'],
    equivalence='Every new hooked forward must match saved A0 final boxes/sigmoid scores<=1e-6 and exact flatTop300 query/class IDs; same model digest before/after',
    limitations=['In-sample author checkpoint despite probe-fit held-out folds','Equal heads do not make receptive fields/intrinsic capacity equal',
                 'Negative linear probes do not exclude nonlinear information','Same fine-cell GT share a probe center','Not object AP,generalization or causal representation-loss proof'])
for file,obj in [('source_contract.json',source_contract),('protocol.json',protocol)]:
    (ROOT/file).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(selfcheck())
# Independent known linear relation verifies weighted/intercept solver and held-out prediction.
rng = np.random.default_rng(7)
x = rng.normal(size=(512,192)); xt = rng.normal(size=(64,192))
beta = np.zeros(192); beta[:3] = [.3,-.5,.2]
row = dict(features_P2=x,target=x@beta+.7,file_name='known-linear')
fitted = fit_readouts([row],'P2',ridge=1e-8,fold=0)
prediction = predict(dict(features_P2=xt),'P2',fitted)
assert np.max(np.abs(prediction[:,0]-(xt@beta+.7))) < 1e-6
assert np.mean((prediction[:,1]-(xt@beta+.7))**2) > .01
for file in ['readout_common.py','extract_features.py','analyze_readouts.py','setup_remote.py','tinyformer_probe_loader.py']:
    py_compile.compile(str(ROOT/file),doraise=True)
files = [dict(file=file,sha256=sha(ROOT/file),bytes=(ROOT/file).stat().st_size) for file in
         ['protocol.json','source_contract.json','readout_common.py','extract_features.py','analyze_readouts.py','setup_remote.py','tinyformer_probe_loader.py']]
manifest = dict(created_utc=NOW,status='FROZEN_BEFORE_NEW_FORWARDS_OR_PROBE_FITS',files=files,
                local_selfchecks='PASS known field,coordinates,rings,ties,IoU and held-out exact linear relation/shuffled control',
                preprocessing_ast_identical_to_A0=True)
(ROOT/'freeze_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(BASE/'19_A1实际路径与冻结协议索引.json').write_text(json.dumps(dict(created_utc=NOW,
    protocol=str(ROOT/'protocol.json'),source_contract=str(ROOT/'source_contract.json'),freeze_manifest=str(ROOT/'freeze_manifest.json'),
    freeze_sha256=sha(ROOT/'freeze_manifest.json'),status='PREPARED_NOT_EXECUTED',new_full_model_forwards_cap=16,
    maximum_cumulative_forwards=185,diagnostic_readouts_to_fit=32),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='A1_FROZEN_LOCAL_SELFCHECKS_PASS',images=16,groups=len(ordered_groups),
                     folds={i:sum(r['fold']==i for r in images) for i in range(4)},
                     freeze_sha256=sha(ROOT/'freeze_manifest.json'),new_forward_cap=16,cpu_readout_fits=32),ensure_ascii=False))
