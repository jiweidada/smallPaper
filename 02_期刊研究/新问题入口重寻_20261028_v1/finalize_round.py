"""Read-only source/arithmetical audit, then version current state and seal outputs."""
import json,hashlib,copy,time,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
LIB=ROOT/'文献_相关原始论文_20261004'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    started=time.perf_counter();inputs={};checks=[]
    def bind(p,h=None):
        p=Path(p);s=str(p);v=sha(p)
        if h is not None:assert v==h,('input_hash',s,v,h)
        assert s not in inputs or inputs[s]==v
        inputs[s]=v;return v
    outcomes={}
    for key,rel in [('LQE','LQE_effect_attempt02/results.json'),('boundary','boundary_condition_attempt01/results.json'),('duplicates','image_duplicates_attempt02/results.json'),('labels','identical_pixels_labels_attempt01/results.json')]:
        p=HERE/rel;r=read(p);bind(p)
        for s,h in r['sources'].items():bind(s,h)
        assert r['status']=='COMPLETED';outcomes[key]=r
    for n in [1,4,5,7,10]:
        p=next(HERE.glob(f'{n:02d}_*.json'));side=p.with_suffix('.sha256')
        if side.exists():assert sha(p)==side.read_text(encoding='utf-8').strip();bind(side)
    lq=outcomes['LQE'];br=outcomes['boundary'];du=outcomes['duplicates'];la=outcomes['labels']
    assert lq['evaluator_calls']==8 and la['CPU_evaluator_calls']==4
    for s in ['1','2']:
        x=lq['seeds'][s]
        assert x['counts']['allGT']==38759 and x['counts']['smallGT']==26586
        for arm,z in x['comparisons'].items():
            for group,ts in z['GT_transition'].items():
                for t,v in ts.items():assert v['both']+v['recovered']+v['reverse_loss']+v['neither']==v['denominator']
            assert z['delta_pp']['AP']<0 and z['delta_pp']['APs']<0
        for g in ['allGT','visible_proxy']:
            total=sum(z['GT'] for z in br['groups'][g].values())
            if g=='allGT':assert total==38759
        q=la['val_prediction_comparison'][s]
        assert q['max_final_box_difference']==q['max_final_logits_difference']==q['max_export_score_difference']==0
        for t,z in q['by_IoU'].items():assert sum(z['counts'].values())==z['denominator_common_cap_pairs']
        for name,z in la['annotation_version_sensitivity'][s]['scenarios'].items():
            assert z['fullGT']==(38760 if name=='A_for_both' else 38758)
            assert sum(v['GT'] for v in z['per_alias'].values())==(82 if name=='A_for_both' else 80)
            for k,v in z['delta_pp'].items():assert abs(v-(z['metrics_pp'][k]-lq['seeds'][s]['arms']['native']['metrics_pp'][k]))<1e-12
    assert du['counts']['train']==6471 and du['counts']['val']==548
    assert du['exact']['decoded_RGB_pixels']['all_duplicate_groups']==3
    assert du['exact']['decoded_RGB_pixels']['crosssplit_groups']==0
    assert du['strong_similarity_proxy']['valimages']==0
    assert sum(z['GT'] for z in du['model_reference'].values())==38759
    assert len(la['groups'])==3 and all(not z['rawTXT_identical'] for z in la['groups'])
    assert br['standardized']['support_near']['GT']==719 and br['standardized']['near_total']==1770
    # Independent partner artifacts: verify every sealed artifact and bind the seal.
    partner_verified=[]
    for dirname in ['讨论伙伴_新问题入口_20261028_v1','重复图标注独立核查_20261028_v1']:
        p=ROOT/dirname/'delivery_manifest.json';m=read(p);bind(p)
        for x in m['artifacts']:bind(x['path'],x['sha256'])
        partner_verified.append({'directory':str(p.parent),'artifacts':len(m['artifacts']),'hashes_match':True})
    # Attach exact primary-source identities and actual read scope. Metadata != full read.
    papers=[]
    specs=[
      ('77_IoU_Net_ECCV2018.pdf','https://openaccess.thecvf.com/content_ECCV_2018/papers/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.pdf','Local16page PDF; published ECCV2018 metadata','Root PDF1 abstract/introduction only; no fullpaper claim'),
      ('06_GFLv2_CVPR2021_distribution-statistics_LQE__arXiv-2011.12885.pdf','https://arxiv.org/pdf/2011.12885v1','ActualPDF arxiv2011.12885v1,2020-11-25; filenameCVPR2021 is not actualreadversion','Root PDF1/2 problem and PDF4 §3.2 Eq3–5; PDF6 text also seen, no newperformanceclaim; figures not independentlyimageinspected'),
      ('11_D-FINE_distribution-refinement__arXiv-2410.13842v1.pdf','https://arxiv.org/pdf/2410.13842v1','arxiv2410.13842v1,2024-10-17,18pages','Root PDF1 and referencepage12; currentLQEimplementation from SHA-bound source, not inferredsolelyfrompaper'),
      ('20261028_新问题_Zone_Evaluation_arXiv2310.13215v2.pdf','https://arxiv.org/pdf/2310.13215v2','arxiv2310.13215v2,2024-06-01,16pages; notclaimactualpublisherPDF','Root PDF1/2, PDF5 §4.1–4.2 relevanttext, PDF12 §6.5 start; object/canvas versuswholeimageshift limits'),
      ('20261028_新问题_Motion_Blur_CVPR2021_Publisher.pdf','https://openaccess.thecvf.com/content/CVPR2021/papers/Sayed_Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR_2021_paper.pdf','CVFofficialCVPR2021 PDF11pages','Root webextraction§3.4/4.2–4.4; partnerreadPDF4–6 relevantsections plus furtherPDF6/9, actualscopes inpartnerledger. RootnotclaimfullPDFread'),
      ('20261028_新问题_Geospatial_Dedup_CVPR2026_Publisher.pdf','https://openaccess.thecvf.com/content/CVPR2026/papers/Adimoolam_Data_Leakage_Detection_and_De-duplication_in_Large_Scale_Geospatial_Image_CVPR_2026_paper.pdf','CVFofficialCVPR2026 PDF10pages','Root PDF1–5 text, includesFigure1 textdescription andlowtexturefalsepositive limits. Notfullpaperorallfiguresread')]
    for fn,url,version,scope in specs:
        p=LIB/fn;h=bind(p);papers.append({'path':str(p),'url':url,'bytes':p.stat().st_size,'sha256':h,'version':version,'actual_read_scope':scope})
    access=[]
    for p in [LIB/'_sources_新问题条件下载_20261028_v1.json',LIB/'_sources_新条件_LossMix下载_20261028_v1.json',LIB/'_sources_方法边界_ObjectLab下载_20261028_v1.json',HERE/'11_官方验证源取回回执.json',HERE/'11b_官方验证源正常下载回执.json']:
        if p.exists():bind(p);access.append({'receipt':str(p),'content':read(p)})
    ledger={'date':'2026-10-28 America/Los_Angeles','workflow':'academic-research-suite inlinevalidate+targetedprimarysourceconditions;notformalreview','papers':papers,
            'download_and_access_receipts':access,
            'abstract_only_priorwork':[{'title':'ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data','url':'https://arxiv.org/abs/2309.00832','scope':'Officialsearchabstract only;v1PDFSSLfailure;no fullmethodorcomprehensivegapclaim'},
                                       {'title':'Supervision Interpolation via LossMix','url':'https://ojs.aaai.org/index.php/AAAI/article/download/28335/28658','scope':'Officialwebfirstpage/problemtext only;PDFdownloadfailed;notproposed asprojectmethod'}],
            'unused_search_hits':'WACV2024LossInspection and2026real-errorbenchmarks were searchleads only,no fullread/performance/gapclaims. Initial guessedarxiv2409.16089 was unrelatedFaceRecognition and discarded. WrongCVF authorURL403 entries are not sourceevidence.',
            'peer_scope_ledgers':[str(ROOT/'讨论伙伴_新问题入口_20261028_v1/source_ledger.json'),str(ROOT/'重复图标注独立核查_20261028_v1/source_chain_results.json')],
            'official_dataset_source_status':'OfficialREADMEowner GoogleDriveval link checked; normaldownloadreturnedHTML, no ZIPretrieved; no auth/captcha bypass, officialoriginremainsunverified',
            'no_comprehensive_novelty_claim':True}
    save(HERE/'13_原文与来源账本.json',ledger);save(LIB/'_sources_新问题入口_20261028_v1.json',ledger)
    fallacies=[
      ('Simpson','AllGTboundarydirectionreversal retained; no pooledseedssubstitution'),
      ('Ecological','No grouprate orweightedexpectedhits asindividualcause/recoveredGT'),
      ('Selection_Berkson','Duplicatepoolselectedbyimageidentity, notmodelerrors; finiteclosestcontrolsnotpopulationaudit'),
      ('Collider','Sameimage/GTstrata are descriptivematching, notcausaladjustment; rawflagsnotgold'),
      ('Base_rate','4/6471train,2/548val,92/343205and81/38759areinvolvementnoterrorrates;0/0rateNA'),
      ('Regression_to_mean','No failedGTchosenforimprovement;alternativeA/Bbothreported,no bestversionchoice'),
      ('Survivorship','Boundary719/1770supportonly; all3exactgroupsandallcrossretrievaloutcomesretained'),
      ('Look_elsewhere','Allentered/failed/deferredrouteslisted;no AP50 rescue ornewshiftgroups'),
      ('Forking_paths','Allscientificprotocolhashesheld; engineeringparity/BLASrepairsrecorded'),
      ('Causation','Annotationversionaffectsfixedoutputjudgment locally; nottrainingharm,methodgain,orofficialdatasetrootcause'),
      ('Reverse_causality','Sourcehashandrawlabels precedemodeljudgment; spatial/blurassociationnotcause')]
    verification={'status':'PASS','verification_status':'ANALYZED;inputhashandarithmeticchecksPASS;noindependentfullnumericrerunorhumangoldlabels',
                  'input_bindings_checked':len(inputs),'CPU_evaluator_calls':12,'new_model_forwards':0,'training':0,'fallacy_scan_coverage':'11/11',
                  'fallacies':[{'type':k,'handling':v} for k,v in fallacies],'partner_artifacts_verified':partner_verified,
                  'current_data_problem_cards':1,'supported_method_mainlines':0,'elapsed_verification_s':time.perf_counter()-started,
                  'constraints':'No models/checkpoints/heads loaded,noforward/backward/training/BNupdate/GPU/remotetraining/testdata;originalANN/checkpoints/cachefilesunchanged',
                  'engineering_exception':'Native matrixmultiplication0xc06d007f;explicitnonBLAScontractionscompleted;failedprocessalreadyexited;preciseDLLnotdiagnosed'}
    save(HERE/'14_来源与算术核验.json',verification)
    statepath=ROOT/'raw_EMA逐GT对照_20261026_v1/work_state_v2.json';old=read(statepath)
    assert old['version']==8 and statepath.read_bytes()==(HERE/'previous_work_state_v8.json').read_bytes()
    new=copy.deepcopy(old);new.update(version=9,client_date='2026-10-28',latest_user_instruction='继续换新问题入口，不绕回老问题；不训练、不提前模块或损失；用户错误弹窗已核失败进程退出',
      current_objective='Evidence-supported concrete projectproblems;boundedroundcomplete1localdata-integritycard,no supportedmethodmainline',
      current_authority=str(HERE/'12_本轮问题地图与取舍.txt'),current_problem_card=str(HERE/'问题卡01_同像素异标注_数据问题成立但未立为研究主线.txt'),
      current_problem_cards=[str(HERE/'问题卡01_同像素异标注_数据问题成立但未立为研究主线.txt')],
      hypothesis_ledger=str(HERE/'research_state.json'),last_wait_cursor='a5fd4e68-4e8d-42ec-8a44-96f8ba393ec1:4',
      discussion_status='Publicconsensus:exactlocal samepixels/differentrawlabels+fixedpredictionevaluationdifference supported,limitedimpact;no methodmainline;sourceadjudicationunclosed',
      current_round_status='BOUNDED_ROUND_COMPLETE_ONE_DATA_PROBLEM_NO_METHOD_MAINLINE',current_supported_innovation_line=None,
      completed_current_stage='All7019source exact/perceptualaudit;3exactgroupsrawsourcechain;4fullannotationversionsensitivityevaluations;LQEfullflownegative;boundaryforwardnotexecuted',
      completed_current_counts={'train_images':6471,'val_images':548,'valGT':38759,'duplicate_groups':3,'train_involved_images':4,'val_involved_images':2,'train_involved_annotation_records':92,'val_involved_annotation_records':81,'crosssplit_exact_duplicate_groups':0,'crosssplit_strong_hash_proxy_valimages':0},
      root_execution_this_stage={'CPU_COCO_evaluator_calls':12,'unique_source_images_in_completed_audit':7019,'model_head_checkpoint_loads':0,'model_forwards':0,'training':0,'backwards_VJP_optimizer_BN_updates':0,'GPU_remote_training_test_reads':0,'failed_round_RGB_attempts':'Incompletefatalattemptstartedreadsnotrecoverable;notreported0'},
      latest_stage_cost={'CPU_COCO_evaluator_calls':12,'models_checkpoints_heads_forwards_backwards_training_BN_GPU_remote_training_test_reads':0,'successful_full_sourceimage_audit':7019,'explicit_originalpixel_displays':'8boundarypairs,3exactduplicatepairs,8crossretrievalcontrols','sourcefailure_cleanup_receipt':str(HERE/'09_失败进程清理回执.json')},
      next_hypothesis=None,next_independent_work=[],confirmed_innovation_mainline_count_this_round=0,
      bounded_stage_status='Bounded condition/effectauditcompleted;broaderinnovationsearchnotdeclaredexhausted',
      previous_state_snapshot=str(HERE/'previous_work_state_v8.json'),lowfrequency_is_default_mainline=False,
      important_limits=['Samecurrentpixelidentity doesnotestablish officialdistributionerror;officialZIPnotretrieved','Currenttrainingcopy factdoesnotestablishhistoricaltrainingRGBidentity','A/Bannotationversion sensitivityisnotcorrectedGTor detectorimprovement','Smallknownissueimpact isnotboundonallannotationerrors','TwoAIchecksarenothumangoldadjudication','OldCPUvsoriginalhost reproductiongap remains','Boundaryonlyproxyassociation;plannedshiftunexecuted;notdefaultline'])
    save(HERE/'research_state.json',{'version':1,'client_date':'2026-10-28','routes':[
      {'id':'LQE_output_correction_harm','status':'PRECHECK_NOT_SUPPORTED','source':str(HERE/'LQE_effect_attempt02/results.json')},
      {'id':'boundary_condition','status':'OBSERVATIONAL_ONLY_NOT_DEFAULT;forwardnotexecutedafterusersteer','source':str(HERE/'06_用户转向后的边界候选状态.txt')},
      {'id':'exact_samepixels_differentlabels','status':'LOCAL_DATA_PROBLEM_CONFIRMED;notmethodmainline','source':str(HERE/'问题卡01_同像素异标注_数据问题成立但未立为研究主线.txt')},
      {'id':'crosssplit_identity_overlap','status':'Noexactgroupsorstrongproxywithinfrozenaudit;notallsceneindependenceproof','source':str(HERE/'image_duplicates_attempt02/results.json')},
      {'id':'natural_motion_blur','status':'Visibleheterogeneitywitnesses,conditionandindependenterrorrolenotestablished','peer_source':str(ROOT/'讨论伙伴_新问题入口_20261028_v1/02_自然模糊条件_本轮结论与最小下一证据.txt')},
      {'id':'MixUp_targetweights','status':'Staticcodefactonly;notbugorharmestablished;no numericalexperiment'}],
      'problem_cards':new['current_problem_cards'],'supported_method_mainline_count':0,'training':0,'broader_search_exhausted':False})
    statepath.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
    output=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json']
    save(HERE/'delivery_manifest.json',{'status':'PASS','date':'2026-10-28 America/Los_Angeles','inputs':[{'path':p,'sha256':h} for p,h in sorted(inputs.items())],
                                      'outputs':output,'mutable_work_state':{'path':str(statepath),'sha256_at_delivery':sha(statepath),'version':9},
                                      'verification_report':str(HERE/'14_来源与算术核验.json'),'scope':'Currentboundedroundonly;no wholeprojectcompletionclaim'})
    print(json.dumps({'status':'PASS','input_bindings':len(inputs),'output_files':len(output),'state_version':9,'data_problem_cards':1,'method_mainlines':0,'CPU_COCO_evaluations':12,'model_forwards':0,'verification_s':time.perf_counter()-started}),flush=True)
if __name__=='__main__':main()
