"""Freeze cached-score diagnostics and source-aligned region policy before effects."""
from pathlib import Path
import datetime,json,platform,sys
import numpy as np
from cache_components import R,N,A30,D,read,sha
sys.stdout.reconfigure(encoding='utf-8')
def main():
    assert not (R/'cache_protocol_FROZEN.json').exists()
    checks=read(R/'component_preflight.json');assert checks['status'].startswith('PASS_')
    old=read(N/'71_A30双头任务与原始支持核验回执.json')
    for p,h in old['files_sha256'].items():assert sha(p)==h,p
    cfg=read(A30/'native_dual_head_protocol_FROZEN.json');raw=read(R/'raw_annotations_verified.json');assert raw['original_valid_GT']==4491
    p={'phase':'A31','status':'FROZEN_BEFORE_SHIFTED_SCORE_EFFECTS','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_type':'Targetednearest-method review plus exploratory consumed-cache control;not a confirmatory study',
       'question':'Does predicted-box-center reading of the existingone2one class probability field reveal a geometry-specific,fixed-FP improvement while retaining duplicate suppression?',
       'not_algorithm_claim':'Plain interpolation,locality,featurealignment andlocalcross-scale competition are prior. Even passing these gates promotes a mechanism question only,not a novel architecture.',
       'source_data':'A30fixedD0YOLO26s100sourceimages,unmodifiedrawoutputs;no newimage forward ortraining',
       'input_hashes':{'raw_annotations_verified.json':sha(R/'raw_annotations_verified.json'),'cache_components.py':sha(R/'cache_components.py'),'A30_capture_metadata':sha(D/'capture_metadata.json'),'A30_result':sha(D/'result.json'),'A27_region_helper':sha(N/'A27_查询扩容同预算任务诊断_v1'/'o27_metrics.py'),'A30_independent_metric_helper':sha(A30/'independent_coco_reaggregation.py')},
       'prior_A30_receipt':str(N/'71_A30双头任务与原始支持核验回执.json'),'prior_A30_files':len(old['files_sha256']),
       'versions':{'python':platform.python_version(),'numpy':np.__version__},
       'coordinate_spec':'Float32 scalar-reciprocal andclipping reproduces allcachednativeO originalcorners exactly;inputrecordinverse snaps originalcorners tofloat32 and is exacton76726native records. Probabilities are analyticfloat64sigmoid ofcachedlogits. Gridindex=x/stride-0.5,visibleclippedpredictedcenter,borderreplication,within samelevel.',
       'zero_fidelity':{'max_native_AP_abs_diff':1e-4,'max_small_TP_fixed_FP_diff':2,'score_vs_native_abs_prechecked':1e-7,'raw_box_original_abs_prechecked':2e-4,'zero_sample_bitwise_identity':True,'native_pairs_score_ge005_prechecked':True},
       'region_policy':{'repository':'VisDrone/VisDrone2018-DET-toolkit','commit':'005445782213e20cb91bc50a597db3dd949e749a','functions':['matlab_round','ignore_integral','region_keep'],'behavior':'Class0rectanglesunion;dropGT/detections ifunionfraction>=0.5,source1basedbitmap/integral,halfawayrounding,min1positions/dimensions. Capbeforefilter,norefill. Class11unscored,noregioncreation. ScorevalidGT>0,classes1..10mapto0..9.','scope':'Source-alignedignored-region preprocessing plusCOCO101;not officialVisDrone continuouspresence-weightedbenchmark reproduction.'},
       'groups':[
         {'id':'M_NATIVE_NMS_REFERENCE','kind':'existingreference','source':'Mbox_Mscore_NMS'},
         {'id':'O_CACHED_NATIVE','kind':'cachednative','source':'Obox_Oscore_TOPK'},
         {'id':'ZERO_GRID','kind':'analyticidentity','alpha':0.,'pool':'all8400'},
         {'id':'CENTER_FLOW','kind':'diagnostic','alpha':1.,'pool':'all8400'},
         {'id':'REVERSE_FLOW','kind':'matchedmagnitudecontrol','alpha':-1.,'pool':'all8400'},
         {'id':'HALF_CENTER_FLOW','kind':'diagnostic','alpha':.5,'pool':'all8400'},
         {'id':'HALF_REVERSE_FLOW','kind':'matchedmagnitudecontrol','alpha':-.5,'pool':'all8400'},
         {'id':'CENTER_KEEP_POOL','kind':'fixedfinalidentitycontrol','alpha':1.,'pool':'nativeeligibleanchor-classpairs'},
         {'id':'REVERSE_KEEP_POOL','kind':'fixedfinalidentitycontrol','alpha':-1.,'pool':'nativeeligibleanchor-classpairs'}],
       'selection':'All8400boxidentities unchanged. Flowgroupsnative2stagetopk500overall10classes,scorefloor0.001. Keep-poolgroupsretainthesamecachednativeeligibleanchor-classpairs,countandboxes,onlyre-score/sortwithoutnewfloororrefill. Sourceignorefilteraftercap;classmapping0..9 throughout.',
       'metrics':{'primary':'COCO101input-area smallAP50:95,strictarea<1024;originalcoordinateIoU,officialsourceignorepreprocessing,maxDet500.','also':['originalallAP','AP50','originalsmallAP','fixedFPsmallrecall','duplicateFPgeometry','pergroupoutputcounts','boundaryclamps'],
         'primary_FP_budget':'NumberofnonignoredFP ofO_CACHED_NATIVE atconfidence0.25/IoU0.5underthisfrozenregionpolicy;function-definedbeforeseeingnewpolicyresults','secondary_FP_budgets':[300,1200,3000],
         'fixed_FP':'Wholeequalscoregroups,maximaloneglobalthresholdprefixwithFP<=budget;uniqueGTTP;reportactualFP.',
         'duplicateFP':'UnmatchednonignoreddetectionwhoseboxIoU>=0.5toavalidsameclassGT;reportedatfixedFPthreshold,not anindependentcausalcategory.',
         'bootstrap':{'draws':1000,'seed':431103,'unit':'pairedimage,singlecheckpoint,consumed100devimages','contrasts':[['CENTER_FLOW','ZERO_GRID'],['CENTER_FLOW','REVERSE_FLOW'],['CENTER_KEEP_POOL','O_CACHED_NATIVE']]}}
       ,'gates':{'purpose':'Decideonlywhetherdirectpredicted-centerclassfieldtransportdeservesfurthermechanismwork;not automaticalgorithmapproval.','CENTER_FLOW_inputsmall_AP_gain_min':.005,'CENTER_FLOW_all_AP_gain_min':-.002,'CENTER_minus_REVERSE_inputsmall_AP_min':.002,'CENTER_FLOW_smallTP_gain_at_primary_FP_min':20,'CENTER_KEEP_POOL_inputsmall_AP_gain_min':0.,'CENTER_minus_ZERO_bootstrap95_lower_strictly_positive':True,'CENTER_minus_REVERSE_bootstrap95_lower_strictly_positive':True,'fail_action':'Do not train orpromotedirectcentertransport onthisevidence;reportallgroupsandlimitations,do notselectbestalphaafterseeingresults.'},
       'limits':{'cache_images':100,'evaluation_groups':9,'scorefields_per_image':5,'bilinear_level_evaluations':1500,'whole_model_calls':0,'head_or_encoder_calls':0,'optimizer_updates':0,'training_jobs':0,'wall_seconds':1800},
       'stop_conditions':['Hash/class/shape/boxfidelityfailure','ZERO_GRIDdriftbeyondfrozenfidelitybounds','nonfinitevalues','exceedcachegrouporwallcap','Noautomaticretry,parametersearchornewGPUjob'],
       'semantic_limits':['NoGTpositionsorGTclassusedtoformscoretransport.','Rawcoverandlowconfidencenotcausaldefectproof.','Consumeddevset/onecheckpointdo notconfirmgeneralization.','ThiscontrolisnotTOOD,DeFCNorConsistentTeacher reproduction.']}
    with (R/'cache_protocol_FROZEN.json').open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':p['status'],'groups':len(p['groups']),'whole_model_calls':0,'old_A30_files_reverified':len(old['files_sha256'])}))
if __name__=='__main__':main()
