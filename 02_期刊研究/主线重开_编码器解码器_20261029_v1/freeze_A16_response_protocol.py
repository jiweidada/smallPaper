"""Freeze source-specific, bounded response audit before any response outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A16_实际特征空间变化审计_v1'
ASSETS=ROOT/'official_assets_v1'/'Dome_DETR_2dde3bc_v1'
assert not (OUT/'protocol.json').exists() and not (OUT/'analysis_summary.json').exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
asset=json.loads((OUT/'asset_selection.json').read_text(encoding='utf-8'))
assert len(asset['cases'])==25 and asset['selected_tiny_GT']==590 and asset['new_full_forward_cap']==26
source_files=['src/zoo/dome/dome_decoder.py','src/zoo/dome/utils.py','src/zoo/dome/matcher.py','src/zoo/dome/box_ops.py']
scripts=['a16_spatial_operator.py','a16_source_validation.py','a16_capture_native_patches.py',
         'a16_process_wrapper.py','a16_response_analysis.py']
for name in scripts:
    path=ROOT/name;ast.parse(path.read_text(encoding='utf-8'));shutil.copyfile(path,OUT/name)
loader=ROOT/'dome_probe_loader.py'
p={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_ACTUAL_FEATURE_RESPONSE_METRICS_AND_NEW_PREFIX_PREDICTIONS',
    'question':'Does native final-layer weighted multi-scale aggregation cancel materially more local spatial response in poorly localized tiny queries than geometry-matched successful queries?',
    'scope':'Exploratory architecture mechanism audit,not a proposed detector or a claim of encoder loss.',
    'source_commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
    'source_hashes':{rel:sha(ASSETS/'source'/rel) for rel in source_files},'loader_sha256':sha(loader),
    'state_digest':'82c3ba49d69370a3d0105e47fc0ab5d715a9e7b1ad15feb275b1b497b5b0ccce',
    'checkpoint_sha256':'42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7',
    'sampler_parameters_sha256':sha(OUT/'final_sampler_state.npz'),
    'native_sampler':{'strides':[4,8,16,32],'spatial_shapes':[[200,200],[100,100],[50,50],[25,25]],
        'heads':8,'channels_per_head':32,'points_per_level':[4,4,4,4],'offset_scale':.5,'point_scale':.25},
    'data':'Reuse12A12/A15official-valprefixeswith263tinyGT;add25annotation-selectednewprefixeswith590tinyGT. All37exploratory,notblindtest.',
    'selection_sha256':sha(OUT/'asset_selection.json'),'old_patch_receipt_sha256':sha(OUT/'old_patch_packing_receipt.json'),
    'old_geometry_pairs_sha256':sha(OUT/'geometry_pairs.json'),'new_predictions_seen_before_freeze':False,
    'feature_response_metrics_seen_before_freeze':False,
    'seed':2026103016,'bootstrap_seed':20261030161,'epsilon_input_pixels':.125,
    'preprocessing':'Existingauthorloader800squaretorchvisionbilinearantialiasTrue,float32/255,noImageNetnorm,TF32off.',
    'phase_full_forward_cap':26,'budget':'Exactlyoneplainidentity+25nativehookedpasses;nowarmups,nointerventions,nosuffix,nofitsortraining. Prior226fullpassesand72suffixphaseclosed.',
    'operator_caps':{'AUTHOR_CPU_DOUBLE_SAMPLING_CORE':25,'CPU_DOUBLE_OFFSET_LINEAR':5,'CPU_DOUBLE_ATTENTION_LINEAR':5},
    'operator_validation':'3syntheticrectangular/boundary/constant/rampcasesplusoldcase00andnewcase00;5positionspercase;exactASTauthorcoreonCPUdouble.10separatelinearprojectionchecks. NofullmodelorCUDAinthesechecks.',
    'matcher':{'weight_dict':{'cost_class':2,'cost_bbox':5,'cost_giou':2},'use_focal_loss':True,'alpha':.25,'gamma':2.},
    'matching':'ExactauthorHungarian2/5/2focal.25/2,validpositivewhrawqueries;matchallGTthenretainmatchedsqrtarea<8GT. Unmatchedtinycountedseparately.',
    'finite_response':'Freezeactualnativepostselfattentionca_query;reconstructoffset/softmaxinFP64fromnativecheckpoint. Moveonlysamplingreferencecentreby±1/8inputpixelonfixedvaluefields. Symmetricdifference,notclaimedinfinitesimalorfullimageJacobian.',
    'packing':'3x3FP32sourcepatcharoundnearestfeaturepixelcentreatnativepoint;align_corners=Falsecontinuouspixel=loc*WH-.5;zeropadoutside.±epsmoves<=.03125featurecell,within3x3support. CompareagainstexactCPUdoubleauthorcore.',
    'grams':'Gstack=sum_h,p a_hp D_hp^T D_hp;Glevel=sum_h,l mass_hl J_hl^T J_hl;Gpool=sum_h J_h^T J_h. BasicJensen0<=Gpool<=Glevel<=Gstack,notnoveltheorem.',
    'primary':'C=1-trGpool/trGstack;exactzerostack=>C0+zeroresponseflag. BadminusgoodCwithinmatchedpairs. Secondarywithin/acrosslevelandabsolutetracesdescriptiveonly.',
    'pairs':'Sameimage/class/input-referenceIoUquarterbin;GTsqrtsizeratio<=1.5,inputreferenceboxsqrtsizeratio<=1.5,positiveinputwh. GoodfinalnativeIoU>=.75,bad<.75. Allsuchpairs,keepold51pairsexact.',
    'aggregation':'Mean paired contrast per image,then equal image mean;2000 prefix bootstrap seed20261030161 for combined and+1 for new. Correlated GT/pairs are not independent replicates.',
    'minimum_data':{'all':{'images_with_pairs':8,'distinct_good':20,'distinct_bad':40,'pairs':100},
                    'new':{'images_with_pairs':6,'distinct_good':10,'distinct_bad':20,'pairs':50}},
    'resource_gate':'Bothcombinedandnew-onlymatchedcohortsadequate;bothmeanbad-minus-goodC>=.10andbootstraplower>0. Positivesupportsonlynextcausaloperatorcomparison,nottraining/mainlineclaim.',
    'stop':'At26fullpasses;fixed25newprefixes,pairing,epsilon,metrics,gates. Noadaptivecase/threshold/epsilon/moduleloop. Inadequatecohort=>inconclusive;nogradient-preservingmoduletrainingfromgenericcancellation.',
    'source_identity_gates':['Strict1033keystate82c3;samplerweightsbyteequalexport;sourcehashesunchanged',
        'Firstplainandhookrawboxes/logitsexact','Allimagehashesmatch;P2present;statebeforeafterexact',
        'NumPypatchandCPUdoubleexactsourcecoreallcloseatol1e-8rtol1e-10','FreeGPU>=2GiB;leaveotherjobsalone'],
    'hard_timeout':'Eachof3boundedstages600s;singlewrapper,noautomaticretryorotherqueue.',
    'not_claimed':['Causalencoderfailure','Genericfrequencyenhancementisnovel','DeploymentorAPgain','Newmethod','Trainingmemoryfeasibility'],
    'nearest_sources':'HowDoVisionTransformersWork,ViT-Anti-Oversmoothing/AttnScale/FeatScale,FDAM. Genericlow-pass/high-frequencyclaimsareprior;non-squarefixedqueryMSDApartialresponsediffersfromdeepNxNselfattentiontheorem.'}
write(OUT/'protocol.json',p)
files=scripts+['protocol.json','asset_selection.json','final_sampler_state.npz']
write(OUT/'freeze_manifest.json',{'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_METRICS',
    'files':[{'file':name,'sha256':sha(OUT/name),'bytes':(OUT/name).stat().st_size} for name in files]})
print(json.dumps({'status':p['status'],'new_images':25,'new_tiny_GT':590,'full_forward_cap':26,
    'source_core_cap':25,'separate_linear_calls_cap':10,'protocol_sha256':sha(OUT/'protocol.json')}))
