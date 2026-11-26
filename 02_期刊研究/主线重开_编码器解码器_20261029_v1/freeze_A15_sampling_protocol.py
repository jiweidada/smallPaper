"""Freeze independent bounded mechanism protocol before intervention outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'A15_固定query的采样中心干预_v1'
BASE=ROOT/'A12_原生边界分布诊断_v1'
SOURCE=ROOT/'official_assets_v1'/'Dome_DETR_2dde3bc_v1'/'source'
NOW=datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

DEST.mkdir(exist_ok=True)
assert not (DEST/'protocol.json').exists()
code=['a15_sampling_intervention.py','a15_sampling_analysis.py','a15_process_wrapper.py']
for name in code:
    ast.parse((ROOT/name).read_text(encoding='utf-8'))
    shutil.copy2(ROOT/name,DEST/name)
baseline=[BASE/'asset_selection.json']+sorted((BASE/'predictions').glob('case*.npz'))
assert len(baseline)==13
source_files=['src/zoo/dome/dome_decoder.py','src/zoo/dome/dome_utils.py','src/zoo/dome/utils.py']
source_audit={'utc':NOW,'source_commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
    'question':'Does translating thefinalsamplingreference towardtheassignedtinyGT centreimprovefrozenlocalization throughactual imagevaluecontent?',
    'source_facts':['MSDAoffsets/weights arecomputedfromquery;referencecentre enterssamplinglocations additively.',
        'Last-layerincomingtarget,querypositionembedding,andpreviousFDRcornerlogits canbeheldfixed.',
        'ReplaysuseoriginalfinalTransformerDecoderLayerandoriginalbboxhead/integral/LQE,withfixedfirst-layerregressionreference.',
        'Positionembedding interventionalsoaffectsself-attentionandquery-predictedoffsets/weights;it isnotapurecoordinateprioralone.'],
    'nearest_work_reading':{'IoU-Net':'Complete3.1-3.3 algorithm;PrRoIPoolformula excerpt;notwholepaper/code. Existinggradient-basedboxrefinementwithoutanexternaltrackingtemplate.',
        'ATOM':'Complete3.1;selected3.2. Usesreferenceappearanceandfirst-frameboxforinstance-specificIoU;notauthorrun.',
        'DiMP':'Complete3.1-3.3and3.5;selected3.4/3.6/3.7. GaussNewtonupdatesclassificationfilter;boxrefinementusesATOMstyleIoU. Noauthorrun.',
        'TAPTRv2':'PreviouslypinnedKDA/APUsource;genericattention-basedpositionupdateisprior.',
        'PixLoc_DEQDet':'A14readingscopepreserved;notnewtousefeaturegradientsorrefinementawareness.'},
    'novelty_scope':'Mechanismdiagnosticnotanewalgorithm;no learnedgradientmetricornewarchitectureapproved.'}
write(DEST/'source_operator_audit.json',source_audit)
p={'utc':NOW,'status':'FROZEN_BEFORE_A15_NATIVE_CAPTURE_AND_INTERVENTION_OUTCOMES',
    'design':'Exploratorycausal interventionon12alreadyobservedA12official-valcases. KnownA13localizationgap;notnewindependentgeneralization.',
    'dataset':'Same12prediction-blind-selectedofficialVisDronevalimages;263GTsqrtarea<8at800-square. KnownA12nativeboxesandmatchingarefrozen.',
    'source_commit':source_audit['source_commit'],
    'source_hashes':{name:sha(SOURCE/name) for name in source_files},
    'loader_sha256':sha(ROOT/'dome_probe_loader.py'),
    'state_digest':'82c3ba49d69370a3d0105e47fc0ab5d715a9e7b1ad15feb275b1b497b5b0ccce',
    'checkpoint_sha256':'42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7',
    'baseline_files':[{'file':f.relative_to(BASE).as_posix(),'sha256':sha(f)} for f in baseline],
    'preprocessing':'Exactauthor800-squarebilinearantialias,float32/255,noImageNetnormalization,FP32TF32off.',
    'seed':2026103012,'bootstrap_seed':20261030151,
    'phase_full_forward_cap':13,'phase_suffix_forward_cap':72,
    'prior_cumulative_full_model_forwards':213,
    'budget':'1plainfirst-imageidentity+12nativecapturefullpasses;6fixedfinal-layersuffixarmsperimage=72;no warmup,optimizer,trainingormodulesearch. Previouscapsremainclosed.',
    'arms':['REPLAY_NATIVE','SAMPLE_GT_CENTER','POSITION_GT_CENTER','BOTH_GT_CENTER',
        'NATIVE_PERMUTED_VALUE','SAMPLE_GT_CENTER_PERMUTED_VALUE'],
    'intervention':'OnlyA12matchedtinyquerycentresarechangedtoGTcentres. Width/height,first-layerFDRreference,previouscornerlogits,incomingqueryandweightsremainnative. SAMPLE keepsquerypositionembeddingnative;POSITION onlyusesGTcentrewhencomputingquerypositionembedding;BOTHcombinespaths.',
    'negative_control':'Roll eachlast-layervaluespatialmap by(H//2+3,W//2+5),moduloitsownshape,identicallyinnativeandSAMPLEarms. Incomingqueryalreadycontainsrealimageinformation;controlisolateslast-readcontentonly.',
    'preinterpretation_gates':['Strictstate/sourcehashes;atleast2GiBfreeGPUandleaveotherjobsunchanged.',
        'Nativehookedrawboxes/logits exactlymatchplainfirstimageandall12frozenA12predictionfiles.',
        'REPLAY_NATIVEfinallayeroutput/rawboxes/logitsmatchfreshnativeexactly.',
        'SAMPLE-onlycrossattentionqueryequalsnativeexactly;henceoffset/weightbranchunchanged.',
        'Nonintervenedqueryboxes/logitsunchangedexactlyforSAMPLE-only.',
        'Modelstateunchangedandallfrozeninputfilesunchanged.'],
    'primary_metric':'FixedA12queryassignmentgeometricIoU>=.75fractionamongall263tinyGT,including0formissingmatch;equal-imageaverageand2000prefixbootstrap. NoAPorclassificationclaim.',
    'primary_effect':'SAMPLE_GT_CENTER minusREPLAY_NATIVE IoU75coverage.',
    'content_interaction':'(SAMPLEreal-NATIVEreal)-(SAMPLEpermuted-NATIVEpermuted),sameIoU75coverage.',
    'resource_screen_gate':'Atleast8images/prefixesand120tinyGT;primarymean>=.05andbootstraplower>0;contentinteractionmean>=.03andlower>0. Bothrequired. Thisisresourceprioritization,notconfirmatorysignificanceornovelty.',
    'secondary_metrics':'POSITION/BOTHcoverageeffects,meanIoU,invalidboxcounts,rawlogitchanges. Theycannotreplaceafailedprimarygate.',
    'stop':'Atmost13fullpassesand72suffixpasses;failurewritesterminalreceipt,noautomaticretry;noextracases,variants,thresholdsearchortraining.',
    'not_claimed':['GTcentreadjustmentisdeployableinference','NewalgorithmorAPgain','Unknownquerytemplatehasbeensolved',
        'Encoderinformationlossifnegative','Alllayerfailuresareexplained','Noveltyorindependentgeneralization','Trainingmemory/timefeasibility']}
write(DEST/'protocol.json',p)
frozen=[DEST/name for name in code]+[DEST/'protocol.json',DEST/'source_operator_audit.json']
write(DEST/'freeze_manifest.json',{'utc':NOW,'status':'FROZEN','files':[{'file':f.name,'sha256':sha(f),'bytes':f.stat().st_size} for f in frozen]})
print(json.dumps({'status':p['status'],'full_cap':13,'suffix_cap':72,'baseline_npz':12,'new_training':0}))
