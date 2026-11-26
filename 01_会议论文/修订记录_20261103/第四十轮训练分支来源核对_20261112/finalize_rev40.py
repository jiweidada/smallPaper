from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import zipfile
from lxml import etree as ET

ROOT=Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/"修订记录_20261103"/"第三十九轮内容一致性与贡献收束_20261112"
MASTER=ROOT/"稿件"/"MPSMA-DEIM_conference_EN_revised.docx"
ORIGINAL=ROOT/"稿件"/"MPSMA-DEIM_conference_EN.docx"
CAND=HERE/"MPSMA-DEIM_conference_EN_revised_rev40.docx"
BACKUP=HERE/"MPSMA-DEIM_conference_EN_revised_rev39_backup.docx"
SOURCE_SHA="8f5ba9fd3bcdd9fa8b2f8f297937b6d7de0afc3c77993024fadd7b855fbfa189"
CAND_SHA="0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c"
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main","m":"http://schemas.openxmlformats.org/officeDocument/2006/math"}
NOW=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rec(p):return {"file":p.relative_to(ROOT).as_posix(),"sha256":sha(p)}
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return rec(p)
def text(e):return "".join(n.text or "" for n in e.iter() if n.tag.endswith("}t"))

def main():
    assert sha(MASTER)==sha(BACKUP)==SOURCE_SHA and sha(CAND)==CAND_SHA
    assert sha(ORIGINAL)=="ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323"
    package=[]
    for path in [BACKUP,CAND]:
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            package.append({n:z.read(n) for n in z.namelist()})
    old,new=package
    assert sorted(k for k in new if new[k]!=old[k])==["word/comments.xml","word/document.xml"]
    a,b=ET.fromstring(old["word/document.xml"]),ET.fromstring(new["word/document.xml"])
    ba,bb=a.find("w:body",NS),b.find("w:body",NS)
    assert [i for i,(x,y) in enumerate(zip(ba,bb)) if text(x)!=text(y)]==[16,61,76]
    assert text(ba[5])==text(bb[5])
    for selector in [".//w:tbl",".//w:drawing",".//m:oMath"]:
        assert [ET.tostring(x) for x in a.findall(selector,NS)]==[ET.tostring(x) for x in b.findall(selector,NS)]
    assert [text(p) for p in ba.findall("w:p",NS) if re.match(r"^\[\d+\]",text(p))]==[text(p) for p in bb.findall("w:p",NS) if re.match(r"^\[\d+\]",text(p))]
    branch_path=HERE/"branch_provenance_audit_rev40.json";branch=read(branch_path)
    assert branch["diagnostic_rows_in_manuscript_BASE_trajectory"]==0
    assert branch["BASE_all_320_rounds_checked_against_native_rows"]["maximum_metric_difference"]==0
    pages=[HERE/"rendered_word"/f"page-{i}.png" for i in range(1,10)]
    assert all(p.exists() and p.stat().st_size>80000 for p in pages)
    assert len(list((HERE/"rendered_word").glob("page-*.png")))==read(HERE/"render_metadata_rev40.json")["page_count"]==9
    visual=write(HERE/"visual_qa_rev40.json",{
        "status":"ALL_NINE_RENDERED_PAGES_INSPECTED","observed_utc":NOW,"candidate_sha256":CAND_SHA,
        "renderer":"Bundled renderer diagnosed missing soffice.exe; hidden read-only Word export with macros disabled, then bundled Poppler.",
        "render_log":rec(HERE/"render_attempts_rev40.json"),"pdf":rec(HERE/"rendered_word"/"manuscript.pdf"),
        "pages":[dict(page=i,**rec(p)) for i,p in enumerate(pages,1)],
        "inspection":"All pages legible; updated text on pages 2, 6 and 8 fits. No clipping, overlap, missing equation symbols or broken tables. All six figures and captions remain readable. Poppler font warnings were checked visually.",
        "target_venue_template_applied":False,
    })
    old_acr,new_acr=read(PREVIOUS/"acronym_check_rev39.json"),read(HERE/"acronym_check_rev40.json")
    sig=lambda d:([(x["scope"],x["rule"],x["acronym"],x["occurrences"]) for x in d["findings"]],[(x["scope"],x["acronym"],x["reason"]) for x in d["coverage_limits"]])
    assert sig(old_acr)==sig(new_acr) and not new_acr["unread_sections"]
    acronym=write(HERE/"acronym_dispositions_rev40.json",{
        "automated_status":new_acr["status"],"new_report":rec(HERE/"acronym_check_rev40.json"),
        "identical_advisory_signatures_to_rev39":True,"previous_manual_dispositions":rec(PREVIOUS/"acronym_manual_disposition_rev39.json"),
        "no_new_acronyms_in_changed_sentences":True,"automated_clean_certificate":False,
    })
    citation=write(HERE/"citation_carryforward_rev40.json",{
        "revision":40,"candidate_sha256":CAND_SHA,"prior_audit":rec(PREVIOUS/"citation_audit_rev39.json"),
        "reference_entries":13,"reference_text_and_cited_claims_unchanged":True,"new_bibliography_entries":0,
        "full_text_unavailable_references":[3,10],"reference3_pagination":"Already verified against publisher-registered Crossref metadata in rev39.",
    })
    finalcheck=write(HERE/"final_candidate_readback_rev40.json",{
        "status":"PASSED","observed_utc":NOW,"candidate":rec(CAND),"backup":rec(BACKUP),"original":rec(ORIGINAL),
        "only_changed_paragraph_indices":[16,61,76],"changed_comment_id":2,"abstract_regenerated":False,
        "tables_figures_equations_references_identical_to_rev39":True,"all_nine_pages_checked":True,
        "branch_audit":rec(branch_path),"visual_qa":visual,"acronym_dispositions":acronym,"citation_carryforward":citation,
    })
    assert sha(MASTER)==SOURCE_SHA
    tmp=MASTER.with_name("MPSMA-DEIM_conference_EN_revised.rev40.tmp.docx")
    shutil.copyfile(CAND,tmp);assert sha(tmp)==CAND_SHA;os.replace(tmp,MASTER)
    patch_path=HERE/"targeted_patch_rev40.json";patch=read(patch_path)
    patch["status"]="REV40_PROMOTED_AFTER_BRANCH_AND_VISUAL_REVIEW";patch["final_readback"]=finalcheck
    write(patch_path,patch)
    index_path=ROOT/"修订记录_20261103"/"主要结论证据索引"/"主要结论证据索引_第四十轮.json"
    index=write(index_path,{
        "revision":40,"observed_utc":NOW,"manuscript":rec(MASTER),"previous_index":rec(index_path.with_name("主要结论证据索引_第三十九轮.json")),
        "branch_provenance_audit":rec(branch_path),"branch_comparison_table":rec(HERE/"BASE_three_branches_round214_225.csv"),
        "targeted_patch":rec(patch_path),"final_readback":finalcheck,"visual_qa":visual,"citation_carryforward":citation,"acronym_dispositions":acronym,
        "new_evidence":"All 320 BASE rows checked against three native sources; both Figure 6 series' AP/APs checked across 320 rows; the embedded PNG matches its provenance hash. The separate diagnostic's eleven rounds and the discarded original eleven rounds do not enter the retained BASE trajectory.",
        "interpretation":"A common frozen recipe does not erase different continuation histories. The manuscript now explicitly prevents assigning the complete-configuration difference solely to architecture.",
        "scientific_numbers_and_endpoints_changed":False,"abstract_regenerated":False,
        "new_model_forwards":0,"new_evaluator_runs":0,"new_training_runs":0,"submission_ready":False,
    })
    audit_path=HERE/"current_goal_acceptance_audit_rev40.json"
    audit=read(PREVIOUS/"current_goal_acceptance_audit_rev39.json")
    audit.update({"revision":40,"observed_utc":NOW,"manuscript":rec(MASTER),"previous_revision_backup":rec(BACKUP),"evidence_index":index,"branch_provenance_audit":rec(branch_path),"abstract_regenerated":False,"new_experiment_authorization_requested_or_used":False})
    audit["original_requirements"][0]["evidence"].append(rec(branch_path))
    audit["original_requirements"][0]["remaining_limit"]="The retained branches are correctly identified, but restarting from a validation-selected anchor with missing random states and differing continuation histories remains a comparison-design limitation. Source identity does not remove that limitation."
    audit["original_requirements"][2]["state"]="Rev39 content review carried forward; rev40 corrects three provenance/comparison statements and one comment, then checks all nine pages."
    audit["original_requirements"][2]["evidence"].extend([rec(patch_path),visual])
    audit["original_requirements"][3]["state"]="Original untouched; rev39 exact backup and promoted rev40 retained; current pointers and logs updated."
    audit["original_requirements"][3]["evidence"]=[rec(ORIGINAL),rec(BACKUP),rec(MASTER)]
    audit_ref=write(audit_path,audit)
    progress_path=HERE/"active_goal_progress_rev40_20261112.json"
    progress=write(progress_path,{
        "goal_turn_classification":"progress","previous_goal_turn_classification":"progress","observed_utc":NOW,"goal_status":"active",
        "revision":40,"manuscript":rec(MASTER),"evidence_index":index,"acceptance_audit":audit_ref,
        "concrete_progress":["Reconciled completed diagnostic authorization and consumption.","Verified every BASE row's native source, all twelve metrics, counters and EMA/prediction/array hashes.","Verified Figure 6 series and embedded PNG provenance.","Established that discarded and diagnostic branches were not mixed into retained results.","Corrected the unqualified controlled-comparison wording and brought continuation-history limits next to the main result."],
        "remaining":"Venue/template unspecified; stronger causal, variability and deployment claims still lack evidence.",
        "new_training_runs":0,"model_forwards":0,"evaluator_runs":0,"continuous_monitoring":False,
    })
    state_path=ROOT/"远程公平对照_20261103"/"模块消融"/"current_execution_state.json"
    shutil.copyfile(state_path,HERE/"current_execution_state_before_rev40.json");state=read(state_path)
    state.update({"observed_local_utc":NOW,"paper_revision":40,"paper_sha256":sha(MASTER),"current_evidence_index":str(index_path),"current_evidence_index_sha256":index["sha256"],"current_goal_acceptance_audit":str(audit_path),"current_goal_acceptance_audit_sha256":audit_ref["sha256"],"latest_active_goal_progress":str(progress_path),"latest_active_goal_progress_sha256":progress["sha256"],"latest_BASE_branch_provenance_audit":str(branch_path),"latest_BASE_branch_provenance_audit_sha256":sha(branch_path),"goal_status":"active","last_goal_turn_classification":"progress","paper_submission_ready":False,"operational_work_status":"ALL_AUTHORIZED_RUNS_COMPLETE; REV40_BASE_BRANCH_PROVENANCE_AND_COMPARISON_SCOPE_RECONCILED; VENUE_FORMAT_DEFERRED; NO_NEW_REMOTE_OBSERVATION"})
    write(state_path,state)
    current=ROOT/"当前修订入口.txt";shutil.copyfile(current,HERE/"当前修订入口_before_rev40.txt")
    note=(
        "当前第40轮维护稿（2026-11-12 UTC）\n\n"
        f"稿件：{rec(MASTER)['file']}\nSHA256：{sha(MASTER)}\n"
        f"上一版精确备份：{rec(BACKUP)['file']}\n原始稿保留，SHA256：{sha(ORIGINAL)}\n\n"
        "本轮证据核对：BASE的320行逐项对回原生三段日志（214+94+12行），每行12项指标、更新计数、EMA/预测/评价数组哈希一致。原分支215–225轮的11行与单独诊断的11轮均未混入保留轨迹。Figure 6两臂全部320轮AP/APs与固定数据表一致，嵌入PNG与绘图源SHA一致。\n\n"
        "稿件修改：移除一处无条件的controlled comparison表述；在主结果旁明确恢复历史阻止纯架构归因；写清BASE原分支已记225轮，但保留续训从214轮锚点重新开始，215–225原行不计入。仅改3处正文和批注2；摘要未重新生成，实验数值和端点未改。9页逐页检查通过，2表、6图、6公式、13条文献保留。\n\n"
        "诊断状态：已完成11轮/17787次更新/14次完整验证，未复现原骤降且不能确定根因；它是独立诊断，不是待运行任务。原BASE/FULL及NO_MPS/NO_MSCA授权运行均已完成保留320轮，没有扩大授权启动新训练。\n\n"
        "现有稿件可支持记录中的配置比较，仍缺排除恢复历史和容量影响的证据。单seed、验证集选点、未保留物理工作量未知和未测当前n模型时延/机载性能等限制继续保留。会议未选择，模板、页数和投稿清单未适配；尚不能宣布整体投稿验收完成。\n\n"
        f"当前索引：{rec(index_path)['file']}\n验收：{rec(audit_path)['file']}\n"
        f"来源核查：{rec(branch_path)['file']}\n"
        "本轮只读既有实验记录，没有模型前向、评价器重跑或新训练；未重新查询远程，旧点时观测不代表实时状态。\n"
    )
    current.write_text(note,encoding="utf-8");(HERE/"revision_notes_rev40.txt").write_text(note,encoding="utf-8")
    for p in [ROOT/"目标自主执行入口_20261109.txt",ROOT/"修订记录_20261103"/"修订说明与未解决问题.txt",ROOT/"远程公平对照_20261103"/"结果并入与验收清单.txt"]:
        p.write_text("【第40轮：分支来源与比较边界核对】\n"+note+"\n以下为历史记录。\n\n"+p.read_text(encoding="utf-8-sig"),encoding="utf-8")
    final={"status":"REV40_PROMOTED_AND_POINTERS_READ_BACK","observed_utc":NOW,"manuscript":rec(MASTER),"backup":rec(BACKUP),"original":rec(ORIGINAL),"evidence_index":rec(index_path),"acceptance_audit":rec(audit_path),"progress":rec(progress_path),"current_state":rec(state_path),"goal_status":"active","submission_ready":False}
    assert sha(MASTER)==CAND_SHA and read(state_path)["current_evidence_index_sha256"]==sha(index_path)
    write(HERE/"postpromotion_chain_readback_rev40.json",final)
    print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
