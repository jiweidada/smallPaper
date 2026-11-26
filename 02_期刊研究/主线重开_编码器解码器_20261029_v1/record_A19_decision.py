"""Record verified A19 source/asset preparation. No author-model execution."""
import ast,hashlib,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
sys.stdout.reconfigure(encoding="utf-8")
R=Path(__file__).resolve().parent
S=R/"source_audit_v1"
O=R/"A19_共享状态近邻与诊断宿主准备_v1"
A=R/"official_assets_v1/VSSD_ICCV2025_Tiny_c10de2f_v1"
C=["01_候选判别表.json","13_架构检索增补记录.json","14_架构核心文献矩阵.json","15_当前状态与结果校验.json","16_最近工作源码核查.json"]
REPORT="48_A19共享状态最近方法与诊断宿主准备.txt"
RECEIPT=R/"49_A19方法与宿主资产核验回执.json"
def rd(p): return json.loads(p.read_text(encoding="utf-8"))
def wr(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def main():
    assert not RECEIPT.exists(),"A19 already recorded; do not rerun"
    utc=datetime.now(timezone.utc).isoformat()
    old=rd(R/"47_A18方法证据与标注预检回执.json")
    for f in old["files"]: assert sha(R/f["path"])==f["sha256"],f["path"]
    assert (R/REPORT).is_file()
    targets=rd(O/"source_targets.json")["targets"]
    assert len(targets)==7
    scopes={
     "A19_EfficientViM_detection_backbone.py":"complete181lines",
     "A19_EfficientViM_detection_utils.py":"complete159lines",
     "A19_EfficientViM_README.md":"completeREADME",
     "A19_EfficientViM_detection_README.md":"completeREADME",
     "A19_RALA_detection_backbone.py":"imports1-22;attention108-152;Block228-257;PatchEmbed/GLTA335-486;otherhelpersnotfullyread",
     "A19_RALA_README.md":"completeREADME",
     "A19_VSSD_pinned_readme.md":"selectedofficialversion/pretraininglinksanddetectiontablelines10-80"
    }
    sourcefiles=[];trees={}
    for row in targets:
        prefix="A18_" if row["id"]=="VSSD" else "A19_"
        tree=rd(S/(prefix+row["id"]+"_tree.json"))
        commit=rd(S/(prefix+row["id"]+"_HEAD_commit.json"))
        assert not tree.get("truncated") and commit["sha"]==row["commit"]
        assert commit["commit"]["tree"]["sha"]==tree["sha"]
        item=next(v for v in tree["tree"] if v["path"]==row["path"])
        f=S/row["local"];b=f.read_bytes()
        gitsha=hashlib.sha1(b"blob "+str(len(b)).encode("ascii")+b"\0"+b).hexdigest()
        assert item["sha"]==gitsha and item["size"]==len(b)
        if f.suffix==".py":ast.parse(b.decode("utf-8"))
        sourcefiles.append({**row,"git_blob":gitsha,"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b),"read_scope":scopes[row["local"]]})
        trees[row["id"]]={"repo":row["repo"],"commit":row["commit"],"tree":tree["sha"],"entries":len(tree["tree"]),"truncated":False}
    pdfs=[]
    for stem,pages,scope in [
       ("EfficientViM_CVPR2025",[3,4,5],"Complete3.1-3.3;Algorithm1/Proposition1statement;selected4.2/ablation;notwholesupplement/proof"),
       ("RALA_CVPR2025",[3,4,5],"Complete3.1-3.3;notallappendix/experiments"),
       ("PFE_Det_Sensors2026",[12,13,14,15],"Abstract,partialintro/framework,complete3.4.1-3.4.3/3.4,Figures4-5,dataavailability;notothermodules/authorcode")
    ]:
        f=S/("A19_"+stem+".pdf")
        assert f.read_bytes().startswith(b"%PDF")
        imageprefix="A19_PFE_method" if stem.startswith("PFE") else "A19_"+stem.split("_CVPR")[0]+"_method"
        for n in pages:assert (S/(imageprefix+"-"+str(n).zfill(2)+".png")).is_file()
        pdfs.append({"file":str(f.relative_to(R)),"sha256":sha(f),"bytes":f.stat().st_size,"read_scope":scope,"visually_reviewed_pages":pages})
    nonpdf=S/"A19_EVMDet_publisher_access.html"
    assert nonpdf.is_file() and not nonpdf.read_bytes().startswith(b"%PDF")
    web=rd(S/"A19_web_evidence_records.json")["records"]
    assert len(web)==7 and sum(len(x["request"].get("search_query",[])) for x in web)==20
    expected=rd(A/"expected_asset.json")
    hf=rd(S/"A19_VSSD_HF_model_metadata.json")
    item=next(x for x in hf["siblings"] if x["rfilename"]==expected["filename"])
    weight=A/expected["filename"]
    assert hf["sha"]==expected["weights_revision"] and hf["private"] is False
    assert weight.stat().st_size==expected["expected_bytes"]==item["lfs"]["size"]
    assert sha(weight)==expected["expected_sha256"]==item["lfs"]["sha256"]
    ops_path=R.parent/"前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl"
    ops=[json.loads(s) for s in ops_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    ops=[x for x in ops if str(x.get("id","")).startswith("A19_")]
    op={x["id"]:x for x in ops}
    envop=op["A19_environment_read_only_20261030"];assert envop["exit_code"]==0
    env=json.loads(envop["stdout"])
    upload=op["A19_upload_verified_VSSD_Tiny_ICCV2025_weight"]
    assert upload["status"]=="UPLOADED" and upload["bytes"]==expected["expected_bytes"]
    remote=rd(O/"initial_remote_weight_inspection.json")
    assert remote["asset_sha256"]==expected["expected_sha256"] and remote["asset_bytes"]==expected["expected_bytes"]
    assert remote["safe_CPU_checkpoint_load_ok"] is False
    tensor=rd(O/"remote_weight_tensor_data.json")
    assert tensor["status"]=="CPU_TENSOR_DATA_LOAD_OK_NOT_MODEL_STRICT_LOAD" and tensor["weights_only"] is True
    assert tensor["model_instantiated"] is False and tensor["model_forward_calls"]==tensor["optimizer_updates"]==0
    assert all(tensor["tensor_groups"][k]["tensor_keys"]==347 for k in ["model","model_ema"])
    draft=rd(O/"next_capacity_protocol_DRAFT.json")
    assert draft["status"]=="DRAFT_NOT_FROZEN_NOT_EXECUTED"
    assert draft["proposed_cap"]["new_complete_detector_forwards"]==3
    assert draft["proposed_cap"]["capacity_optimizer_updates"]==2
    wr(O/"remote_operation_receipts.json",{"operations":ops,"scope":"Publicasset/read-onlyruntimeandCPUcheckpointdataonly;noauthormodelinstantiated"})
    audit={"utc":utc,"trees":trees,"source_files":sourcefiles,"primary_PDFs":pdfs,
           "source_operators_executed":False,"source_complete_model_environment_reproduced":False,
           "EVMDet":{"read_scope":"Publisherabstract/metadata/referencesonly","doi":"10.1007/978-981-95-5758-5_30","conference":"PRCV2025","first_online":"2026-02-07","fulltext_attempt":"HTTP200HTMLnotPDF","preserved_response_sha256":sha(nonpdf)},
           "operator_findings":["EfficientViMstemoutputsnominalstrides16/32/64;VSSD/RALAretainstride4",
             "HSM-SSDhasstate-domainSiLUgatingandblocklocal/FFN/bypasspaths;notapurelinearwholeencoder",
             "HSMB/Caresignedgeneralfeatures;ABcannotbeassumedaninstanceprobabilitydistribution",
             "RALAalreadyhascontextconditionedKVweights,inputconditionedoutputgating,RoPEand5x5LePEininspectedcode",
             "PFE-DetdirectUAVnearbyHSM-SSD+CGLU;EVMDetdirectEfficientViMsmallobjectprior"],
           "mathematical_inference_not_experiment":"IninspectedHSMcode(dt+self.A[:,None]).softmax(-1),self.Aisconstantacrosssoftmaxaxisandcancelsinexactarithmetic.Thatisnottiny-objectfailureevidenceoranalgorithmincrement;noautograd/modeltestperformed.",
           "no_low_rank_capacity_claim":"Globalbranchrankorstatecountdoesnotboundthenumberofdetectableinstancesinthefullresidual/gatednetwork."}
    wr(O/"source_operator_audit.json",audit)
    asset={"utc":utc,"local_file":str(weight),"remote_file":upload["remote"],"bytes":expected["expected_bytes"],
           "sha256":expected["expected_sha256"],"HF_revision":hf["sha"],
           "local_and_remote_hash_verified":True,"tensor_data_read_success":True,
           "author_model_strict_load_verified":False,"model_instantiated":False,
           "forward_backward_capacity_verified":False,"remote_runtime_snapshot":env,
           "remaining":"Isolateddependencies,strictclassificationmodelEMAload,documentedbackboneadapterandfixedcapacityscript/inputmanifest"}
    wr(O/"asset_preparation_receipt.json",asset)
    docs={n:rd(R/n) for n in C}
    board,search,matrix,state,source=[docs[n] for n in C]
    assert len(matrix["rows"])==72 and len(board["candidates"])==14
    additions=[
     {"id":"EfficientViM","title":"EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality",
      "authors":["Sanghyeok Lee","Joonmyung Choi","Hyunwoo J. Kim"],"url":"https://arxiv.org/abs/2411.15241","tier":"CVPR2025officialPDFread",
      "read_scope":pdfs[0]["read_scope"]+";completeauthordetectionbackbone/utilsandREADMEspinned",
      "author_method":"Channelmixing/gatinginreducedhiddenstates;state-wisewritingweights;localconvs/FFN/bypasses;classificationmulti-statelogitfusion",
      "our_consequence":"State-spacegatingandlightweightdetectortransferareprior;nativefirststride16confoundsdirectcomparisonagainstP2hosts",
      "unresolved":"Noauthorcodeexecution,detectionfailurecauseor24GBtrainingclaim;Proposition1specialconditionsnotageneralcompressed-stateequivalence",
      "code_commit":trees["EfficientViM"]["commit"]},
     {"id":"RALA","title":"Breaking the Low-Rank Dilemma of Linear Attention",
      "authors":["Qihang Fan","Huaibo Huang","Ran He"],"url":"https://arxiv.org/abs/2411.07635","tier":"CVPR2025officialPDFread",
      "read_scope":pdfs[1]["read_scope"]+";selectedpinneddetectionattention/block/stem/backboneandREADME",
      "author_method":"GlobalqueryconditionedKVreweightingandinputconditionedtokenoutputmodulation;inspectedcodealsoRoPE/LePE/normalization",
      "our_consequence":"Genericstateweighting/outputdiversitygateisnotnew;empiricalrankgainisnotourinstance-preservationguarantee",
      "unresolved":"Noactualtinymissmechanismevidenceorourreproduction;paper/sourceoperatorcontextsnotidenticalbarematrixform",
      "code_commit":trees["RALA"]["commit"]},
     {"id":"EVMDet","title":"EVMDet: EfficientViM for Small Object Detection",
      "authors":["Tianxiang Zhang","Jichao Jiao","Ning Li","Yuqing Peng","Yingchao Zeng","Jiajie Huang","Ziyi Bao","Zimo Guo","Zilong Liu"],
      "url":"https://doi.org/10.1007/978-981-95-5758-5_30","doi":"10.1007/978-981-95-5758-5_30",
      "tier":"PRCV2025conference;publisherfirstonline2026-02-07","read_scope":"Publisherabstract,metadata,referencesonly;notfullmethod/code",
      "author_method":"Abstractdescribestraining-onlyHADFdynamicTopK/historyfusionandSAFEspatialenhancement;VisDrone/SODA-Dvalidation",
      "our_consequence":"FirstEfficientViM-to-VisDronetransferclaimexcluded",
      "unresolved":"Exactoperator/lossanddetailednoveltycomparisonnotcleared;publicPDFattemptreturnedHTML"},
     {"id":"PFE_Det","title":"PFE-Det: Progressive Feature Evolution for Small Object Detection in UAV Aerial Images",
      "authors":["Aolin Fang","Yongzi Zhang","Xiaotong Dong","Liuyang Gu","Shengshi Li","Daoheng Zhu","Xiuchun Xiao"],
      "url":"https://doi.org/10.3390/s26155003","doi":"10.3390/s26155003","tier":"Sensors2026,26(15),5003",
      "read_scope":pdfs[2]["read_scope"],
      "author_method":"Progressivepreprocessing/backbone/encoderpath;readMG-SSMcomponentcouplesHSM-SSDandCGLUonP3/P4fusionwhileP5retainsTransformer",
      "our_consequence":"GenericHSM-SSDplusspatialgateforUAVsmallobjectsisdirectprior",
      "unresolved":"Othermodules/entiretrainingprotocol/authorcodenotfullyaudited;noindependentreproduction"}
    ]
    assert not any(r["id"] in {x["id"] for x in matrix["rows"]} for r in additions)
    matrix["rows"].extend(additions)
    assert len(matrix["rows"])==76 and len({x["id"] for x in matrix["rows"]})==76
    priority="查证原生非因果状态交互是否对邻近微小实例产生有害内容干扰；先完成VSSD-Tiny严格装载与有界容量预检准备，再冻结训练后机制诊断。通用重加权/输出门控/HSM-SSD+CGLU已有近邻，尚无新算法或失败因果证据。"
    evidence={"utc":utc,"report":str(R/REPORT),"web_records":7,"search_queries":20,
              "source_trees":3,"source_blobs":7,"literature_rows":76,
              "new_full_forwards":0,"new_suffix_forwards":0,"new_fits":0,"optimizer_updates":0,
              "weight_bytes":asset["bytes"],"weight_sha256":asset["sha256"],
              "weight_local_remote_verified":True,"checkpoint_tensor_data_read_ok":True,
              "author_model_strict_load_ok":False,"capacity_protocol":"DRAFT_NOT_FROZEN_NOT_EXECUTED",
              "mainline_validated":False,"decision":"GENERIC_STATE_WEIGHTING_GATING_AND_TRANSFER_ALREADY_PRIOR_PREPARE_NATIVE_HOST_NO_NEW_MODULE"}
    for cand in board["candidates"]:
        if cand["id"]=="N_NONCAUSAL_STATE_INSTANCE_INTERFERENCE":
            cand.update(status="UNTESTED_MECHANISM_GENERIC_FIXES_ALREADY_PRIOR_NATIVE_HOST_PREPARATION",
                        nearest=["VSSD","EfficientViM(fullmethods/selectedcode)","RALA(fullmethods/selectedcode)","EVMDet(abstractonly)","PFE_Det(componentmethod)","PIDE_Net(partial)"],
                        next="StrictCPUload/adapterandboundedcapacityprotocol;thentrained-hostcausalquestion,notgenericgatemoduletraining.")
        if cand["id"].startswith("L_GENERIC"):
            cand["nearest"]=list(dict.fromkeys(cand["nearest"]+["EfficientViM","RALA","EVMDet(abstractonly)","PFE_Det"]))
    board.update(A19_evidence=evidence,current_priority_question=priority,latest_update_utc=utc,latest_report=str(R/REPORT),mainline_validated=False)
    search["A19_shared_state_and_native_host"]={**evidence,"queries":[q for x in web for q in x["request"].get("search_query",[])],
       "raw_records":str(S/"A19_web_evidence_records.json"),"scope":"Targetednon-systematic,singleassistant",
       "limits":["EVMDetfulltextnotobtained:HTTP200HTMLpreserved","PFEonlyMG-SSMcomponentcomplete;notallmodules/code",
                 "Fog/changedetectionpapersretainedasdiscoveryonly,notnewresearchdirection","Regression/Deltaquerydoesnotestablishnovelvisualalgorithm"]}
    search["supplement_utc"]=utc
    matrix.update(updated_utc=utc,latest_report=str(R/REPORT),latest_empirical_boundary="A19sourceandcheckpointdataonly;noauthormodelrun,noeffectestimate,nocapacityresult")
    nextq={"question":priority,"host_asset_receipt":str(O/"asset_preparation_receipt.json"),
           "next_capacity_draft":str(O/"next_capacity_protocol_DRAFT.json"),
           "freeze_actual_script_input_and_cap_before_forward":True,"new_user_permission_required":False,
           "no_automatic_formal_training":True}
    source["A19_shared_state_sources_and_assets"]={**evidence,"source_audit":str(O/"source_operator_audit.json"),"next":nextq}
    source["next_instance_ownership_question"]=nextq;source["updated_utc"]=utc
    previous=state["latest_authoritative_phase"]
    hist=state.setdefault("authoritative_phase_history",[])
    if not any(x.get("utc")==previous.get("utc") for x in hist):hist.append(previous)
    state["previous_goal_turn"]=state.get("current_goal_turn")
    state["current_goal_turn"]="progress_A19_shared_state_nearest_methods_and_verified_new_encoder_assets"
    state["latest_authoritative_phase"]={**evidence,"classification":"progress","goal_status":"active",
      "complete_mainline":False,"full_model_forwards":252,"cumulative_authorized_protocol_cap":252,
      "new_full_cap":0,"historical_A15_suffix_forwards":72,"bounded_phase_complete":True,
      "CPU_checkpoint_read_is_not_model_forward":True,"latest_report":str(R/REPORT),"next":priority}
    state.update(recorded_utc=utc,latest_report=str(R/REPORT),current_priority_question=priority,goal_status="ACTIVE",mainline_validated=False)
    assert state["model_forwards"]==state["cap"]==252 and state["training_updates"]==0
    snap=R/"history_before_A19_update_v1";snap.mkdir(exist_ok=False)
    for n in C+["47_A18方法证据与标注预检回执.json"]:shutil.copy2(R/n,snap/n)
    for n,x in docs.items():wr(R/n,x)
    required=C+[REPORT,"record_A19_decision.py",
      "A19_共享状态近邻与诊断宿主准备_v1/source_targets.json",
      "A19_共享状态近邻与诊断宿主准备_v1/source_operator_audit.json",
      "A19_共享状态近邻与诊断宿主准备_v1/asset_preparation_receipt.json",
      "A19_共享状态近邻与诊断宿主准备_v1/next_capacity_protocol_DRAFT.json",
      "A19_共享状态近邻与诊断宿主准备_v1/remote_operation_receipts.json",
      "A19_共享状态近邻与诊断宿主准备_v1/remote_weight_tensor_data.json",
      "A19_共享状态近邻与诊断宿主准备_v1/initial_remote_weight_inspection.json",
      "A19_共享状态近邻与诊断宿主准备_v1/a19_remote_asset_inspect.py",
      "A19_共享状态近邻与诊断宿主准备_v1/a19_remote_tensor_data_read.py",
      "source_audit_v1/A19_web_evidence_records.json",
      "source_audit_v1/A19_VSSD_HF_model_metadata.json",
      "official_assets_v1/VSSD_ICCV2025_Tiny_c10de2f_v1/expected_asset.json",
      "official_assets_v1/VSSD_ICCV2025_Tiny_c10de2f_v1/local_download_receipt.json"]
    receipt={"utc":utc,"status":"A19_SOURCE_AND_ASSET_PREPARATION_VERIFIED_GOAL_ACTIVE",
      "goal_status":"ACTIVE","mainline_validated":False,"new_complete_model_forwards":0,
      "cumulative_complete_model_forwards":252,"historical_suffix_forwards":72,
      "new_suffix_forwards":0,"new_fits":0,"optimizer_updates":0,
      "core_literature_rows":76,"candidate_rows":14,"source_blobs_verified":7,
      "method_PDFs_read":3,"weight_local_remote_sha256_verified":asset["sha256"],
      "CPU_tensor_data_load_ok":True,"strict_model_load_verified":False,
      "capacity_forward_backward_verified":False,"capacity_protocol_status":"DRAFT_NOT_FROZEN_NOT_EXECUTED",
      "prior_A18_completion_hashes_verified":len(old["files"]),"prior_canonical_snapshot":str(snap),
      "files":[{"path":n,"sha256":sha(R/n),"bytes":(R/n).stat().st_size} for n in required],
      "next":nextq,"limits":"Noidentifiedfailurecause,distinctalgorithm,capacityresultorfinalmainline."}
    wr(RECEIPT,receipt)
    print(json.dumps({k:receipt[k] for k in ["status","goal_status","mainline_validated","core_literature_rows","source_blobs_verified","CPU_tensor_data_load_ok","strict_model_load_verified","capacity_forward_backward_verified","cumulative_complete_model_forwards"]},ensure_ascii=False))
if __name__=="__main__":main()

