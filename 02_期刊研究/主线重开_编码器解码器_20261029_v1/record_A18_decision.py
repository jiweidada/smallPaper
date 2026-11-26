"""Commit A18 source/annotation evidence after source and selected-geometry checks."""
import ast, hashlib, json, math, shutil, sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parent
SA=ROOT/"source_audit_v1"
OUT=ROOT/"A18_邻近实例证据与编码机制审计_v1"
CANON=["01_候选判别表.json","13_架构检索增补记录.json","14_架构核心文献矩阵.json","15_当前状态与结果校验.json","16_最近工作源码核查.json"]
REPORT="46_A18最近方法对照与新编码器问题边界.txt"
RECEIPT=ROOT/"47_A18方法证据与标注预检回执.json"
def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def intersects(a,b): return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
def box_records(path,w,h):
    out={}
    for uid,s in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if not s.strip(): continue
        vals=[int(v.strip()) for v in s.split(",") if v.strip()]
        x,y,bw,bh,score,cat,trunc,occ=vals[:8]
        l,r=max(0,min(w,x)),max(0,min(w,x+bw))
        t,b=max(0,min(h,y)),max(0,min(h,y+bh))
        if r<=l or b<=t: continue
        box=[F(l*800,w),F(t*800,h),F(r*800,w),F(b*800,h)]
        area=(box[2]-box[0])*(box[3]-box[1])
        out[uid]={"box":box,"area":area,"valid":score>0 and 1<=cat<=10,"cat":cat}
    return out
def main():
    assert not RECEIPT.exists(), "A18 already recorded; do not rerun"
    assert (ROOT/REPORT).is_file()
    utc=datetime.now(timezone.utc).isoformat()
    previous=read(ROOT/"45_A17完成核验与编码器研究边界回执.json")
    for row in previous["files"]:
        assert sha(ROOT/row["path"])==row["sha256"],row["path"]
    cfg=read(OUT/"source_targets.json")
    source_files=[]
    for key,spec in cfg.items():
        tree=read(SA/("A18_"+key+"_tree.json"))
        assert tree["sha"]==spec["tree"] and not tree.get("truncated")
        commit=read(SA/("A18_"+key+"_HEAD_commit.json"))
        assert commit["sha"]==spec["commit"] and commit["commit"]["tree"]["sha"]==spec["tree"]
        blobs={x["path"]:x for x in tree["tree"] if x["type"]=="blob"}
        for item in spec["targets"]:
            p=SA/item["local"];b=p.read_bytes()
            blob=hashlib.sha1(b"blob "+str(len(b)).encode("ascii")+b"\0"+b).hexdigest()
            assert blob==blobs[item["path"]]["sha"],item["path"]
            assert len(b)==blobs[item["path"]]["size"]
            if p.suffix==".py": ast.parse(b.decode("utf-8"))
            source_files.append({"repo":spec["repo"],"commit":spec["commit"],"path":item["path"],
                                 "local":str(p.relative_to(ROOT)),"git_blob":blob,
                                 "sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b),
                                 "read_scope":spec["read_scope"][item["path"]]})
    assert len(source_files)==17
    pdfs=[]
    for file,scope,visual in [
        ("A18_VSSD_ICCV2025.pdf","Complete3.1-3.3;selected4.2setup;abstract/partialintroduction;notwholeexperiments/supplement",[3,4,5]),
        ("A18_Samba_CVPR2025.pdf","Complete3.1-3.5.3includingAlgorithm1;selected4.2setup;notwholeexperiments/supplement",[4,5,6])]:
        p=SA/file
        assert p.read_bytes().startswith(b"%PDF")
        prefix="A18_VSSD" if "VSSD" in file else "A18_Samba"
        for page in visual: assert (SA/(prefix+"_method-"+str(page).zfill(2)+".png")).is_file()
        pdfs.append({"local":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size,
                     "read_scope":scope,"visually_reviewed_pdf_pages":visual,
                     "render_note":"Poppler reported missing display fonts; inspected rendered method pages are legible."})
    web=read(SA/"A18_web_evidence_records.json")["records"]
    assert len(web)==13
    nqueries=sum(len(r["request"].get("search_query",[])) for r in web)
    assert nqueries==26
    c=read(SA/"A18_PIDE_Crossref.json")["message"]
    assert c["DOI"].lower()=="10.1016/j.eswa.2026.131194"
    assert "PIDE-Net" in c["title"][0]
    xml=(SA/"A18_PIDE_Elsevier_XML.xml").read_text(encoding="utf-8")
    assert "<openaccess>0</openaccess>" in xml and "<originalText" not in xml
    summary=read(OUT/"cohort_preflight_summary.json")
    assert summary["exit_status"]=="COMPLETE" and summary["new_model_forwards"]==0
    rules=OUT/"cohort_preflight_rules.json"
    assert summary["rules_sha256"]==sha(rules)
    excluded=set(summary["excluded_prefixes"])
    assert len(excluded)==37
    inputs=read(OUT/"annotation_inputs.json")
    per_image=read(OUT/"per_image_feasibility.json")
    selected=read(OUT/"selected_prefix_candidates.json")
    eligible=read(OUT/"eligible_image_pairs.json")
    DATA=Path(r"D:\数据集\visdrone2019\VisDrone2019-DET-val")
    parsed={};tiny_total=0
    for row in inputs:
        image=DATA/"images"/row["image"]
        ann=DATA/"annotations"/(image.stem+".txt")
        assert sha(ann)==row["annotation_sha256"]
        with Image.open(image) as im: assert im.size==(row["width"],row["height"])
        if image.stem.split("_")[0] not in excluded:
            parsed[row["image"]]=box_records(ann,row["width"],row["height"])
            tiny_total+=sum(x["valid"] and x["area"]<64 for x in parsed[row["image"]].values())
    assert len(inputs)==548 and len(parsed)==166 and tiny_total==256
    assert sum(bool(x.get("excluded")) for x in per_image)==382
    assert sum(x.get("eligible_oriented_pairs",0) for x in per_image)==14
    assert sum(x.get("eligible_oriented_pairs",0)>0 for x in per_image)==5
    assert len(selected)==len(eligible)==5 and len({r["prefix"] for r in selected})==5
    for row in selected:
        assert row["prefix"] not in excluded
        raw=parsed[row["image"]];a=raw[row["target_uid"]];b=raw[row["neighbour_uid"]]
        assert a["valid"] and b["valid"] and a["cat"]==b["cat"]==row["category"]
        assert a["area"]<64 and b["area"]<64
        assert max(a["area"],b["area"])<=4*min(a["area"],b["area"])
        assert [F(x) for x in row["target_box800_exact"]]==a["box"]
        assert [F(x) for x in row["neighbour_box800_exact"]]==b["box"]
        ca=[(a["box"][i]+a["box"][i+2])/2 for i in range(2)]
        cb=[(b["box"][i]+b["box"][i+2])/2 for i in range(2)]
        d2=sum((ca[i]-cb[i])**2 for i in range(2))
        assert d2*d2<=16*a["area"]*b["area"]
        nb=row["neighbour_mask"];bg=row["background_mask"];guard=row["target_guard"]
        assert nb==[math.floor(b["box"][0]),math.floor(b["box"][1]),math.ceil(b["box"][2]),math.ceil(b["box"][3])]
        assert guard==[math.floor(a["box"][0]),math.floor(a["box"][1]),math.ceil(a["box"][2]),math.ceil(a["box"][3])]
        assert not intersects(guard,nb) and not intersects(guard,bg)
        assert nb[2]-nb[0]==bg[2]-bg[0] and nb[3]-nb[1]==bg[3]-bg[1]
        assert bg[0]==math.floor(2*ca[0]-nb[2]+F(1,2))
        assert bg[1]==math.floor(2*ca[1]-nb[3]+F(1,2))
        assert 0<=bg[0]<bg[2]<=800 and 0<=bg[1]<bg[3]<=800
        assert not any(intersects(bg,v["box"]) for v in raw.values())
        assert not any(k!=row["neighbour_uid"] and intersects(nb,v["box"]) for k,v in raw.items())
    verification={"utc":utc,"status":"PASS_WITH_STATED_SCOPE","annotation_hashes_and_image_headers_verified":548,
                  "unseen_tiny_GT_exact_recomputed":256,"selected_prefix_cases_geometry_verified":5,
                  "oriented_pair_count14":"Reaggregatedfromper_imagecounts,notindependentlyenumeratedinthischecker",
                  "source_blobs_against_fixed_git_trees_verified":17,
                  "full_model_forwards":0,"new_fits":0,"main_model_updates":0}
    save(OUT/"verification.json",verification)
    audit={"utc":utc,"source_files":source_files,"pdfs":pdfs,
           "repositories":{k:{j:v[j] for j in ["repo","commit","tree","limits"]} for k,v in cfg.items()},
           "static_only":True,"model_imports_executed":False,"weights_downloaded":False,
           "PIDE_scope":"Publisherabstract,introductionpreview,sectionfragmentsandmetadataonly;nofullmethod/code",
           "PIDE_access_evidence":{"openalex_oa":read(SA/"A18_PIDE_OpenAlex.json")["open_access"],
                                  "github_repository_search":"7fuzzymatches,allunrelatedPidevsoftware;noauthorrepoidentified",
                                  "elsevier_xml":"HTTP200metadataonly;notfulltext",
                                  "publisher_web_fetch":"403;indexedpublisherpreviewwasread"},
           "VSSD_version_boundary":"2024v2preprintandlegacyMicroareseparatefromICCV2025Tiny;noAP/memoryreproduction"}
    save(OUT/"source_operator_audit.json",audit)
    snapshot=ROOT/"history_before_A18_update_v1"
    snapshot.mkdir(exist_ok=False)
    for name in CANON+["45_A17完成核验与编码器研究边界回执.json"]:
        shutil.copy2(ROOT/name,snapshot/name)
    files={n:read(ROOT/n) for n in CANON}
    board,search,matrix,state,source=[files[n] for n in CANON]
    rows={r["id"]:r for r in matrix["rows"]}
    assert len(rows)==69 and len(board["candidates"])==13
    rows["CrowdDet"].update(
        read_scope="Completepapersection3and3.1;selectedpinnedhead/EMD/refinement/SetNMSoperatorsinbothMegEngineandPyTorchpaths;notalltrainingdependencies/supplement",
        author_method="PerproposalK-instanceprediction;minimumclassification/regressioncostoversetpermutations;SetNMSprotectssameproposalpredictions;optionalrefinement",
        our_consequence="Genericlocalmulti-instanceoutputandpermutationmatchingarepriorart;movingtheideaearlierisnotitselfnovel",
        unresolved="Notnumericallyreproduced;PyTorchdependencyclosureinselectedpublictreenotestablished;notproofthatourencodermergesinstances")
    rows["Repulsion_Loss"].update(
        read_scope="Complete4and4.1;3.2failureanalysis;selectedexperiments;notwholesupplement/authorcode",
        author_method="RepGTIoGpredictedboxvsnontargetGT;RepBoxIoUbetweendifferent-targetpredictions;Smooth_lnpenaltyplusattraction",
        our_consequence="Neighbourattractionandunion-of-instanceslocalizationerrorsareoldproblems,notnewdiscoveries",
        unresolved="Feature/state-levelcauseand24GBhostadaptationnotestablished;algebraicnonzeropenaltyonoverlappingGTisnotobservedtrainingharm")
    additions=[
        {"id":"PIDE_Net","title":"PIDE-Net: A heterogeneous processing paradigm for UAV object detection",
         "authors":["Shuming Lin","Sang Feng","Jinyi Liang","Junnan Tan"],"doi":"10.1016/j.eswa.2026.131194",
         "url":"https://doi.org/10.1016/j.eswa.2026.131194","tier":"ExpertSystemswithApplications309,131194(2026)",
         "read_scope":"Publisherabstract,introductionpreviewandsectionfragments;Crossref/publisherXMLmetadata;NOTfullmethods/formula/code",
         "author_method":"PRISMposition-semanticdisentanglement;SG-SSMsemanticrouting/dynamicprompts;PEP-Netspatialweavingupsampling,asdescribedinpublisherpreview",
         "our_consequence":"DirectUAVneighbourprecludesgenericposition-semanticdisentanglementplusSSMnoveltyclaim",
         "unresolved":"Fulloperator/loss/implementationcomparisonunavailable;noauthorrepoidentifiedintheboundedsearch;doNOTassumeourfutureinstance-specificoperatorisabsent"},
        {"id":"VSSD","title":"VSSD: Vision Mamba with Non-Causal State Space Duality",
         "authors":["Yuheng Shi","Mingjia Li","Minjing Dong","Chang Xu"],
         "url":"https://openaccess.thecvf.com/content/ICCV2025/papers/Shi_VSSD_Vision_Mamba_with_Non-Causal_State_Space_Duality_ICCV_2025_paper.pdf",
         "tier":"ICCV2025officialpaper;2024v2separatelyread",
         "read_scope":"Completeofficial3.1-3.3,selected4.2setup,abstract/partialintro;7pinnedsource/configfilesselectedoperators;notallresults/supplement/modelrun",
         "author_method":"NoncausalglobalSSDstate;2DRoPE;localCPE/DWConv,gatesandresiduals;laststageattention;4-leveldetectorbackbone",
         "our_consequence":"Noncausality,positionencoding,P2andglobalstateweightingareexistingdesigns;useasoperatorcomparator,notanewmodule",
         "unresolved":"Actualtiny-instanceinterference,matched-budgetincrementand24GBtrainingfeasibilityunmeasured;doNOTapplybarelinearbranchrankboundtofullnormalized/gatednetwork",
         "code_commit":cfg["VSSD"]["commit"],"version_boundary":"ICCVTiny28MvslegacyMicro14M/Tiny24M;authororderdiffersfrom2024v2"},
        {"id":"Samba_salient","title":"Samba: A Unified Mamba-based Framework for General Salient Object Detection",
         "authors":["Jiahao He","Keren Fu","Xiaohong Liu","Qijun Zhao"],
         "url":"https://openaccess.thecvf.com/content/CVPR2025/papers/He_Samba_A_Unified_Mamba-based_Framework_for_General_Salient_Object_Detection_CVPR_2025_paper.pdf",
         "tier":"CVPR2025","read_scope":"Complete3.1-3.5.3includingAlgorithm1;selected4.2setup;notauthorcode/wholesupplement",
         "author_method":"Saliency-guidedspatialneighbouringscanwithforeground/backgroundandreverseroutevariants;context-awarecross-scaleupsampling",
         "our_consequence":"Genericspatial-continuity-preservingSSMserializationisalreadymethodprior",
         "unresolved":"SalientsegmentationisnotUAVtinybboxdetection;noautomatictransferofmetrics,noveltyorresourceclaims"}
    ]
    assert not any(r["id"] in rows for r in additions)
    matrix["rows"].extend(additions)
    assert len(matrix["rows"])==72 and len({r["id"] for r in matrix["rows"]})==72
    priority="查证非因果共享状态编码器中，相邻微小实例的写入/读取是否产生有害交互；先对照VSSD、EfficientViM及相关线性注意力近邻。通用实例区分尚不是论文主线；Dome未见预测的干预预检仅5个前缀，不启动相应统计实验。"
    evidence={"utc":utc,"report":str(ROOT/REPORT),"source_blobs":17,"source_repositories":2,
              "web_requests":13,"search_queries":26,"literature_rows":72,
              "cohort_preflight":summary["counts"],"new_full_forwards":0,"new_suffix_forwards":0,
              "new_fits":0,"main_model_updates":0,"mainline_validated":False,
              "decision":"GENERIC_INSTANCE_ENCODING_NOT_A_CONFIRMED_GAP_DOME_UNSEEN_COUNTERFACTUAL_COHORT_TOO_SMALL",
              "next":"Nearest-operatorauditofEfficientViMandlinear-attentionfeature-diversity/rankmethods;defineanarchitecture-specificfalsifiablequestionandnewhostprotocolbeforeanymodelphase."}
    board["candidates"].append({
        "id":"N_NONCAUSAL_STATE_INSTANCE_INTERFERENCE",
        "question":"Doesnativeposition-aware,locallyaugmentednoncausalshared-stateinteractionharmneighbouringtinyinstancesatfixedstate/computebudget?",
        "status":"UNTESTED_OPERATOR_SPECIFIC_QUESTION_NOVELTY_NOT_CLEARED",
        "increment_approved":False,"failure_cause_identified":False,"causal_improvement_evidence":False,
        "nearest":["VSSD","PIDE_Net(partial)","Samba_salient(operatorprior)","EfficientViM(discoveryonly)"],
        "scope":"Dome5-prefixpreflightisresourcefeasibilityonly,notafailuretestoraglobaldataavailabilitylimit;noVSSDmodelrun.",
        "next":"Readnearestlinearattentionoperatorsbeforeselectingahostorfreezingcausaldiagnostics."})
    for cand in board["candidates"]:
        if cand["id"].startswith("L_GENERIC"):
            cand["nearest"]=list(dict.fromkeys(cand["nearest"]+["VSSD","PIDE-Net(partial)","Samba_salient"]))
    board["A18_evidence"]=evidence
    board["current_priority_question"]=priority
    board["latest_update_utc"]=utc
    board["latest_report"]=str(ROOT/REPORT)
    board["mainline_validated"]=False
    for claim in ["Genericneighbourattractionisnotnew:RepulsionLoss","Genericoneproposal-multipleinstancesisnotnew:CrowdDet",
                  "GenericnoncausalSSDwith2DpositionandP2isnotnew:VSSD","Genericposition-semanticSSMnoveltynotcleared:PIDE-Net"]:
        if claim not in board["rejected_generic_novelty_claims"]: board["rejected_generic_novelty_claims"].append(claim)
    search["A18_source_and_feasibility"]={
        **evidence,"raw_web_records":str(SA/"A18_web_evidence_records.json"),
        "source_targets":str(OUT/"source_targets.json"),"access_limits":audit["PIDE_access_evidence"],
        "queries":[q for record in web for q in record["request"].get("search_query",[])],
        "resolved_PDF_access":"CVFwebfetch403resolvedwithdirectofficialPDFdownload;methodpagesrendered/read",
        "review_type":"targeted non-systematic;singleassistant"}
    search["supplement_utc"]=utc
    matrix["updated_utc"]=utc;matrix["latest_report"]=str(ROOT/REPORT)
    matrix["latest_empirical_boundary"]="A18annotation-onlypreflight:5independentunseenDomeprefixes,noeffectestimated;VSSDnotrun"
    source["A18_pinned_sources_and_prior_boundary"]={
        **evidence,"source_audit":str(OUT/"source_operator_audit.json"),
        "CrowdDetection":audit["repositories"]["CrowdDetection"],"VSSD":audit["repositories"]["VSSD"],
        "VSSD_actual_native_path":"normalizedELUpositiveQ/K;head-specificsoftmaxwriteweights;2DRoPE;normalizedKTVmatmul;D*Vskip;LayerNorm/gate/outprojection;localCPEandblockresidual",
        "no_model_rank_claim":"Barestatefactorizationdoesnotprovethewholenormalizedgatedencoderlosesinstanceseparation",
        "PIDE_read_scope":audit["PIDE_scope"]}
    source["next_instance_ownership_question"]={
        "question":priority,"new_frozen_protocol_required_before_any_new_model_phase":True,
        "new_user_permission_required":False,
        "Dome_unseen_annotation_preflight":"5prefixes,noreliableconfirmatoryexperimentlaunched",
        "new_host_independence":"Ifnewencoderhostischosen,defineitspredictionindependenceandtrainingexposureseparately;doNOTrescuetheDomecohortbyrelaxingitsrules."}
    source["updated_utc"]=utc
    previous_phase=state["latest_authoritative_phase"]
    hist=state.setdefault("authoritative_phase_history",[])
    if not any(x.get("utc")==previous_phase.get("utc") for x in hist): hist.append(previous_phase)
    phase={**evidence,"classification":"progress","goal_status":"active","complete_mainline":False,
           "previous_actual_goal_turn_classification":"progress_PIDE_primary_metadata_discovery",
           "full_model_forwards":252,"cumulative_authorized_protocol_cap":252,
           "new_full_cap":0,"new_full_forwards":0,"new_suffix_forwards":0,
           "historical_A15_suffix_forwards":72,"bounded_phase_complete":True,
           "preflight_tool_exit_code":0,"source_and_geometry_verification":"PASS_WITH_STATED_SCOPE",
           "latest_report":str(ROOT/REPORT),"next":priority}
    state["previous_goal_turn"]=state.get("current_goal_turn")
    state["current_goal_turn"]="progress_A18_classic_and_new_encoder_sources_and_zero_forward_cohort_feasibility"
    state["latest_authoritative_phase"]=phase
    state["recorded_utc"]=utc;state["latest_report"]=str(ROOT/REPORT)
    state["current_priority_question"]=priority
    state["goal_status"]="ACTIVE";state["mainline_validated"]=False
    assert state["model_forwards"]==state["cap"]==252 and state["training_updates"]==0
    for name,data in files.items(): save(ROOT/name,data)
    required=CANON+[REPORT,"a18_neighbour_annotation_preflight.py","record_A18_decision.py",
          "A18_邻近实例证据与编码机制审计_v1/source_targets.json",
          "A18_邻近实例证据与编码机制审计_v1/source_operator_audit.json",
          "A18_邻近实例证据与编码机制审计_v1/verification.json",
          "A18_邻近实例证据与编码机制审计_v1/cohort_preflight_rules.json",
          "A18_邻近实例证据与编码机制审计_v1/annotation_inputs.json",
          "A18_邻近实例证据与编码机制审计_v1/per_image_feasibility.json",
          "A18_邻近实例证据与编码机制审计_v1/eligible_image_pairs.json",
          "A18_邻近实例证据与编码机制审计_v1/selected_prefix_candidates.json",
          "A18_邻近实例证据与编码机制审计_v1/cohort_preflight_summary.json",
          "source_audit_v1/A18_web_evidence_records.json",
          "source_audit_v1/A18_PIDE_Crossref.json",
          "source_audit_v1/A18_PIDE_OpenAlex.json",
          "source_audit_v1/A18_PIDE_GitHub_repository_search.json",
          "source_audit_v1/A18_PIDE_Elsevier_XML.xml",
          "source_audit_v1/A18_VSSD_ICCV2025.pdf",
          "source_audit_v1/A18_Samba_CVPR2025.pdf"]
    receipt={"utc":utc,"status":"A18_COMPLETED_SOURCE_AND_RESOURCE_AUDIT_GOAL_ACTIVE",
             "goal_status":"ACTIVE","mainline_validated":False,"new_complete_model_forwards":0,
             "cumulative_complete_model_forwards":252,"historical_suffix_forwards":72,
             "new_suffix_forwards":0,"new_fits":0,"main_model_updates":0,
             "core_literature_rows":72,"candidate_rows":14,"source_blobs_verified":17,
             "primary_PDFs_method_read":2,"search_queries":26,"web_records":13,
             "annotation_preflight_independent_prefixes":5,"preflight_tool_exit_code":0,
             "prior_A17_completion_hashes_verified":len(previous["files"]),
             "prior_canonical_snapshot":str(snapshot),"decision":evidence["decision"],
             "files":[{"path":f,"sha256":sha(ROOT/f),"bytes":(ROOT/f).stat().st_size} for f in required],
             "next":priority,"limits":"Notmainlinecompletion,nofailureeffect/algorithmnovelty/trainingcapacityproved."}
    save(RECEIPT,receipt)
    print(json.dumps({k:receipt[k] for k in ["status","goal_status","mainline_validated","core_literature_rows","candidate_rows","source_blobs_verified","annotation_preflight_independent_prefixes","cumulative_complete_model_forwards"]},ensure_ascii=False))
if __name__=="__main__": main()

