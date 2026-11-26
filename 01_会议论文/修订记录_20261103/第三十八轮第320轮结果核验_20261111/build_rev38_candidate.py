from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json, re
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
DATA=PAPER/"实验数据汇总_20261108"
RECOVERY=PAPER/"远程公平对照_20261103/模块消融/NO_MSCA/recovery30715_20261109"
MASTER=PAPER/"稿件/MPSMA-DEIM_conference_EN_revised.docx"
BACKUP=HERE/"MPSMA-DEIM_conference_EN_revised_rev37_backup.docx"
CANDIDATE=HERE/"MPSMA-DEIM_conference_EN_revised_rev38.docx"
SOURCE_SHA="53060de71535e78591fd5856ff537cf65c12a9463bd532e7525de490eb3c2910"
ORIGINAL_SHA="ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323"
W="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS={"w":W,"m":"http://schemas.openxmlformats.org/officeDocument/2006/math","wp":"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}
Q=lambda x:"{"+W+"}"+x
text=lambda x:"".join(x.xpath(".//w:t/text()",namespaces=NS))
digest=lambda p:sha256(p.read_bytes()).hexdigest()
def readj(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def writej(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

assert digest(MASTER)==SOURCE_SHA and not BACKUP.exists() and not CANDIDATE.exists()
assert digest(PAPER/"稿件/MPSMA-DEIM_conference_EN.docx")==ORIGINAL_SHA
assert digest(PAPER/"稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx")==ORIGINAL_SHA
readback=DATA/"_build/round320_parent_readback.json"
snapshot_path=DATA/"verified_snapshots/NO_MSCA_round320_verified_20261111/snapshot_manifest.json"
audit=RECOVERY/"lineage_second_recovery_through_round320.json"
binder=RECOVERY/"round320_completion_and_endpoints_20261111.json"
assert readj(readback)["status"]=="ROUND320_PARENT_READBACK_PASSED"
assert readj(DATA/"第320轮更新核查.json")["status"]=="ROUND320_DATA_WORKBOOK_SOURCE_INDEX_AND_SNAPSHOT_VERIFIED"
assert readj(snapshot_path)["status"]=="ROUND320_CURRENT_DELIVERY_BYTE_VERIFIED"
assert len(readj(snapshot_path)["files"])==61
assert digest(snapshot_path)=="958b7bd88b206b5ea7693bffedf5bbdde141e37eb52d17c1cb6340244eb557af"
assert digest(audit)=="66ba7a9b2f61c550222230aa0a661994f585abdc8d0d69cee9430ba727cb1734"
assert binder.is_file()

replacements={
"Small objects in unmanned aerial vehicle (UAV) images provide limited visual evidence and occur amid clutter and scale variation. We present MPSMA-DEIM, an adaptation of DEIM-D-FINE that combines a Multi-branch Progressive Stem (MPS) with Multi-scale Context-aware Attention (MSCA). The stem merges detail, downsampling and high-frequency branches into stride-four features. The attention design uses pooled context at an intermediate stage and dilated context at the deepest stage, with SimAM-derived energy gates. We compare the complete n configuration with its original baseline over 320 retained training rounds on 6,471 images using seed 0, FP32 and batch size 4. Evaluation uses all 548 VisDrone2019 validation images under a frozen local evaluation based on Common Objects in Context (COCO). The final-round exponential moving average (EMA) achieves average precision (AP) of 0.2255 and small-object AP (APs) of 0.1539, compared with baseline values of 0.1902 and 0.1215. The absolute differences are 3.53 and 3.23 percentage points. The complete model has 7.54 million trainable parameters, compared with 3.74 million for the baseline. The comparison includes documented checkpoint continuations. The run replacing MPS with a standard stem using the sigmoid-weighted linear unit (SiLU) reaches AP 0.2099 and APs 0.1362. The MSCA-removal run is ongoing.":
"Small objects in unmanned aerial vehicle (UAV) imagery provide limited visual evidence amid clutter and scale variation. We present MPSMA-DEIM, an adaptation of DEIM-D-FINE that combines a Multi-branch Progressive Stem (MPS) and Multi-scale Context-aware Attention (MSCA). MPS fuses detail, downsampling and high-frequency paths into stride-four features. MSCA applies pooled and dilated context with SimAM-derived energy gates at intermediate and deep stages. Using seed 0, batch size 4 and FP32, we compare the complete n configuration with the original baseline over 320 retained training rounds on 6,471 training images. Evaluation uses all 548 VisDrone2019 validation images under a frozen local Common Objects in Context (COCO)-style protocol. At the fixed round-320 EMA endpoint, FULL achieves AP 0.2255 and APs 0.1539, 3.53 and 3.23 percentage points above BASE; their trainable parameter counts are 7.54M and 3.74M. The conditional NO_MPS stem replacement reaches AP 0.2099 and APs 0.1362. Removing the three MSCA gates gives NO_MSCA AP 0.2222 and APs 0.1473 at round 320; its round-302 AP-best checkpoint reaches AP 0.2253. NO_MSCA has 3.84M fewer trainable parameters than FULL, so this capacity-changing ablation does not isolate an attention-only effect. Results are single-seed observations on one local validation split with documented checkpoint continuations.",
"Table 2. Conditional MPS replacement at the fixed round-320 endpoint. Metrics are fractions; trainable parameters are in millions. All other FULL settings are retained.":
"Table 2. Conditional module-removal results at the fixed round-320 endpoint. NO_MPS replaces MPS; NO_MSCA removes the three MSCA gates. Metrics are AP/APs/AR@100/ARl fractions; trainable parameters are in millions.",
"At fixed round 320, FULL exceeds NO_MPS by 1.57 percentage points in AP and 1.77 in APs (Table 2). NO_MPS reaches its highest AP at round 320, so its primary and secondary checkpoints coincide.":
"At fixed round 320, FULL exceeds NO_MPS by 1.57 percentage points in AP and 1.77 in APs (Table 2). Round 320 is also the NO_MPS trajectory's AP-best checkpoint, so its primary and secondary endpoints coincide.",
"This comparison concerns MPS with the remaining FULL settings present. Its interpretation is conditional on the realized training histories and does not isolate mechanism from capacity.":
"The NO_MPS comparison holds the other FULL components present but changes the stem operator and removes 5,325 trainable parameters. Given the documented training histories, this is a conditional comparison of these runs; it does not estimate a general causal effect across seeds.",
"NO_MSCA remains in progress. BASE lacks the ESE/SimAM gates and uses different aggregation paths and activations, so it is not a joint-removal control. The design provides no MPS–MSCA interaction estimate.":
"NO_MSCA replaces the three MSCA gates with identities and retains their aggregation projections, ESE, SimAM, MPS and SiLU (Table 2). At round 320, FULL exceeds NO_MSCA by 0.33 percentage points in AP, 0.66 in APs and 0.28 in AR@100. The NO_MSCA AP-best secondary endpoint occurs at round 302 (AP 0.2253), selected on the same validation split. Removing MSCA gates also reduces the trainable parameter count by 3,841,280; thus, this conditional ablation combines the component change with a substantial capacity reduction. BASE lacks the ESE/SimAM gates and uses different aggregation paths and activations, so it is not a joint-removal control. The design does not estimate an MPS–MSCA interaction.",
"One seed and one validation split provide no estimate of variability or independent test performance. Validation guides both restoration and secondary checkpoint selection. Capacity differs between configurations. The MSCA-removal endpoint, current n-model latency and onboard benchmarks remain unavailable.":
"One seed and one validation split provide no estimate of run-to-run variability or independent test performance. The validation set guides restoration and secondary checkpoint selection. Model capacity differs among configurations, notably between FULL and NO_MSCA. Latency and onboard performance for the current n model remain unmeasured.",
"MPSMA-DEIM combines a multi-branch stem and stage-specific contextual gates in DEIM-D-FINE. At round 320, FULL achieves AP 0.2255 and APs 0.1539: 3.53 and 3.23 percentage points above BASE, with 2.02 times its trainable parameters. FULL exceeds NO_MPS by 1.57 and 1.77 points in AP and APs, respectively, but has lower large-object recall. These single-seed results concern the documented continuations and local COCO-style evaluation. MSCA removal remains under evaluation.":
"MPSMA-DEIM combines a multi-branch stem and stage-specific contextual gates in DEIM-D-FINE. At the fixed round-320 EMA endpoint, FULL reaches AP 0.2255 and APs 0.1539, 3.53 and 3.23 percentage points above BASE, with 2.02 times as many trainable parameters. FULL also exceeds NO_MPS by 1.57 points in AP and 1.77 in APs, although its ARl is 1.69 points lower. NO_MSCA reaches AP 0.2222 and APs 0.1473 at round 320; its AP-best checkpoint is a separate round-302 EMA with AP 0.2253. At round 320, FULL exceeds NO_MSCA by 0.33 points in AP and 0.66 in APs. Because NO_MSCA contains 3,841,280 fewer trainable parameters than FULL, this is a capacity-changing conditional ablation and does not isolate the MSCA gate effect. All results describe single-seed runs on one local COCO-style validation split under documented continuation histories; they do not establish statistical significance or independent-test generalization. Latency and onboard performance remain unmeasured."
}
with ZipFile(MASTER) as z:
    infos=z.infolist()
    originals={name:z.read(name) for name in z.namelist()}
doc=etree.fromstring(originals["word/document.xml"])
comments=etree.fromstring(originals["word/comments.xml"])
before=deepcopy(doc)
old_comments=deepcopy(comments)
body=doc.find("w:body",NS)
paras=body.findall("w:p",NS)
old_para_text=[text(p) for p in paras]
changes=[]
for old,new in replacements.items():
    hits=[p for p in paras if text(p)==old]
    assert len(hits)==1, (old[:60],len(hits))
    nodes=hits[0].findall(".//w:t",NS)
    assert len(nodes)==1,old[:60]
    nodes[0].text=new
    changes.append({"old":old,"new":new})

tables=doc.findall(".//w:tbl",NS)
assert len(tables)==2
table2=tables[1]
rows=table2.findall("w:tr",NS)
assert len(rows)==3
assert [text(c) for c in rows[-1].findall("w:tc",NS)]==["NO_MPS","0.2099","0.1362","0.3621","0.5615","7.5356"]
values=["NO_MSCA","0.2222","0.1473","0.3715","0.5411","3.6997"]
new_row=deepcopy(rows[-1])
for cell,value in zip(new_row.findall("w:tc",NS),values):
    nodes=cell.findall(".//w:t",NS); assert len(nodes)==1
    nodes[0].text=value
table2.append(new_row)
assert len(table2.findall("w:tr",NS))==4
cm=[c for c in comments if c.get(Q("id"))=="6"]
assert len(cm)==1 and len(comments)==7
cts=cm[0].findall(".//w:t",NS); assert len(cts)==1
old_comment=cts[0].text
new_comment=("更新至NO_MSCA第320轮完成证据：worker完成收据、supervisor退出码0、epoch0319清单/ACK与独立审计绑定。"
"320轮主端点AP=0.2221976，APs=0.1472718；AP最佳次端点为第302轮AP=0.2253384。两者各自绑定同一保存行的12项指标及EMA、预测、评价数组和检查点SHA。"
"逐轮数据四组各320行，旧1240行原字节前缀保留，40条新行与累计日志逐项核对；输入来源索引为43条。末轮与AP最佳结果均为单seed同一验证集选点。"
"本次仅读取既有评估记录，model forwards/evaluator runs均为0。恢复限制继续披露：先后从第29和164轮完整状态恢复；旧partial165支路至少观察到1401次更新，其他未保留物理工作量未知。"
"FULL与NO_MSCA相差3,841,280个可训练参数，因此是容量变化的条件消融。数据快照：实验数据汇总_20261108/verified_snapshots/NO_MSCA_round320_verified_20261111；"
"完整审计、完成收据与源文件SHA见数据快照及NO_MSCA completion evidence packet。")
cts[0].text=new_comment
new_para_text=[text(p) for p in body.findall("w:p",NS)]
mp={x["old"]:x["new"] for x in changes}
assert [mp.get(x,x) for x in old_para_text]==new_para_text
assert len([x for x in new_para_text if re.match(r"^\[\d+\]",x)])==13
for xpath,count in [(".//m:oMath",6),(".//wp:inline",6)]:
    a=before.findall(xpath,NS); b=doc.findall(xpath,NS)
    assert len(a)==len(b)==count and [etree.tostring(x) for x in a]==[etree.tostring(x) for x in b]
old_tables=before.findall(".//w:tbl",NS)
new_tables=doc.findall(".//w:tbl",NS)
assert len(old_tables)==len(new_tables)==2 and etree.tostring(old_tables[0])==etree.tostring(new_tables[0])
for a,b in zip(old_tables[1].findall("w:tr",NS),new_tables[1].findall("w:tr",NS)[:3]):
    assert etree.tostring(a)==etree.tostring(b)
for a,b in zip(old_comments,comments):
    if a.get(Q("id"))!="6": assert etree.tostring(a)==etree.tostring(b)
for tag in ["commentRangeStart","commentRangeEnd","commentReference"]:
    assert sorted(n.get(Q("id")) for n in doc.findall(".//w:"+tag,NS))==[str(i) for i in range(7)]

with BACKUP.open("xb") as f: f.write(MASTER.read_bytes())
with ZipFile(CANDIDATE,"x") as out:
    for info in infos:
        if info.filename=="word/document.xml":
            data=etree.tostring(doc,encoding="UTF-8",xml_declaration=True,standalone=True)
        elif info.filename=="word/comments.xml":
            data=etree.tostring(comments,encoding="UTF-8",xml_declaration=True,standalone=True)
        else: data=originals[info.filename]
        out.writestr(info,data)
with ZipFile(BACKUP) as old, ZipFile(CANDIDATE) as new:
    assert new.testzip() is None and old.namelist()==new.namelist()
    changed=[n for n in old.namelist() if old.read(n)!=new.read(n)]
    assert set(changed)=={"word/document.xml","word/comments.xml"}
candidate_report={
 "status":"ROUND320_RESULTS_CANDIDATE_PENDING_VISUAL_QA","revision":38,
 "observed_utc":datetime.now(timezone.utc).isoformat(),
 "source_manuscript_sha256":SOURCE_SHA,"backup_rev37_sha256":digest(BACKUP),
 "candidate_sha256":digest(CANDIDATE),"changed_package_parts":changed,
 "changed_body_paragraphs":changes,"table_changes":[{"table":"Table 2","added_row":values,"original_rows_byte_identical":True}],
 "updated_comment_id":6,"figures":6,"native_equations":6,"result_tables":2,"reference_entries":13,"comments":7,
 "verified_round320_sources":[
   {"file":str(binder),"sha256":digest(binder)},{"file":str(audit),"sha256":digest(audit)},
   {"file":str(readback),"sha256":digest(readback)},{"file":str(snapshot_path),"sha256":digest(snapshot_path)},
   {"file":str(DATA/"outputs/实验数据汇总_20261108/实验数据汇总.xlsx"),
    "sha256":readj(DATA/"_build/round320_workbook_checks.json")["xlsx_sha256"]}],
 "NO_MSCA_terminal_round_1based":320,"NO_MSCA_best_AP_round_1based":302,
 "NO_MSCA_terminal_AP":0.22219763674879622,"NO_MSCA_terminal_APs":0.1472717517007563,
 "NO_MSCA_best_AP":0.2253384467031726,"NO_MSCA_trainable_parameters":3699662,
 "FULL_minus_NO_MSCA_AP_percentage_points":0.3315790616,
 "FULL_minus_NO_MSCA_APs_percentage_points":0.6585042927,
 "new_external_citations_or_sources":False,"model_forwards":0,"evaluator_runs":0,
 "new_training_runs":0,"venue_template_adaptation":False,"submission_ready":False}
writej(HERE/"candidate_structural_and_evidence_check.json",candidate_report)
tables_dump=[]
for i,table in enumerate(new_tables,1):
    tables_dump.append("TABLE "+str(i))
    tables_dump += [" | ".join(text(c) for c in row.findall("w:tc",NS)) for row in table.findall("w:tr",NS)]
(HERE/"candidate_text.txt").write_text("\n\n".join(new_para_text)+"\n\n"+"\n".join(tables_dump)+"\n",encoding="utf-8")
(HERE/"revision_scope_and_evidence.txt").write_text(
"第三十八轮只更新摘要、固定320轮消融表、NO_MPS/NO_MSCA分析、限制、结论及既有批注6。"
"沿用intro/related work/methods/figures/equations/reference list已核验内容。NO_MSCA数据取自完成收据、"
"supervisor exit0、本机备份清单、独立320轮lineage审计、逐轮读回和不可变数据快照。FULL−NO_MSCA为"
"+0.33 AP与+0.66 APs百分点，因参数少3,841,280而解释为容量变化的条件消融。旧partial165与未知物理计算量继续披露。"
"单seed、同一验证集选点和本地非官方COCO转换口径限制统计推断与泛化；latency/onboard未测。未进行新训练、前向、评价器重跑或添加外部文献。\n",encoding="utf-8")
print(json.dumps({"status":candidate_report["status"],"candidate_sha256":candidate_report["candidate_sha256"],
 "backup_rev37_sha256":candidate_report["backup_rev37_sha256"],"changed_paragraphs":len(changes),
 "table2_rows":len(table2.findall("w:tr",NS)),"changed_parts":changed,"submission_ready":False},ensure_ascii=False,indent=2))
