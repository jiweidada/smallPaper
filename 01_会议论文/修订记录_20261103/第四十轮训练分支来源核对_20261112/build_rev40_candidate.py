from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import shutil
import zipfile
from lxml import etree as ET

ROOT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "稿件" / "MPSMA-DEIM_conference_EN_revised.docx"
SOURCE_SHA = "8f5ba9fd3bcdd9fa8b2f8f297937b6d7de0afc3c77993024fadd7b855fbfa189"
CANDIDATE = HERE / "MPSMA-DEIM_conference_EN_revised_rev40.docx"
BACKUP = HERE / "MPSMA-DEIM_conference_EN_revised_rev39_backup.docx"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(e): return "".join(n.text or "" for n in e.iter() if n.tag.endswith("}t"))

def main():
    assert sha(SOURCE) == SOURCE_SHA
    audit = json.loads((HERE / "branch_provenance_audit_rev40.json").read_text(encoding="utf-8"))
    assert audit["status"] == "BASE_320_ROWS_AND_THREE_BRANCH_PROVENANCE_VERIFIED"
    assert audit["source_manuscript"]["sha256"] == SOURCE_SHA
    if BACKUP.exists(): assert sha(BACKUP) == SOURCE_SHA
    else: shutil.copyfile(SOURCE, BACKUP)
    with zipfile.ZipFile(SOURCE) as z:
        assert z.testzip() is None
        infos=z.infolist(); original={i.filename:z.read(i.filename) for i in infos}
    package=dict(original)
    doc=ET.fromstring(package["word/document.xml"]); body=doc.find("w:body",NS)
    edits=[]
    before=plain(body[16])
    assert "it is outside the present controlled comparison." in before
    replacements={
        16:before.replace("it is outside the present controlled comparison.","it is outside the n-model results reported here."),
        61:(
            "At the primary endpoint, FULL reaches AP 0.2255 and APs 0.1539, compared with BASE values of 0.1902 and 0.1215. "
            "The differences are 3.53 and 3.23 percentage points, respectively. "
            "The secondary best-AP rounds are 303 for BASE and 305 for FULL. "
            "These are comparisons of observed retained trajectories under a common frozen recipe. "
            "Their different continuation histories prevent attribution of the difference solely to architecture."
        ),
        76:(
            "BASE resumed from the round-214 checkpoint after its original branch had logged 225 rounds. "
            "The original rows for rounds 215–225 are excluded from the retained trajectory. "
            "FULL resumed after round 1. Both anchors lacked main-process random states and were reseeded with 0. "
            "BASE resumed again after round 308 with saved random states, rerunning a partial round 309. "
            "NO_MPS began at round zero. The FULL per-batch learning-rate record begins at round 2."
        ),
    }
    for index,new in replacements.items():
        p=body[index];nodes=p.findall("w:r/w:t",NS)
        assert len(nodes)==1
        old=plain(p); nodes[0].text=new
        edits.append({"body_child_index":index,"before":old,"after":new,"before_sha256":hashlib.sha256(old.encode()).hexdigest()})
    comments=ET.fromstring(package["word/comments.xml"])
    comment=comments.find("w:comment[@w:id='2']",NS)
    nodes=comment.findall(".//w:t",NS);assert len(nodes)==1
    old_comment=plain(comment)
    nodes[0].text=(old_comment+" 第40轮将BASE保留的全部320行逐项对回三段原生日志；另核对独立诊断的11轮并未进入主轨迹。Figure 6的两臂AP/APs全部320轮与数据快照一致，嵌入图像SHA也与已存绘图来源一致。共同配方不等于相同的恢复历史，正文不再将此比较称为无条件受控比较。")
    package["word/document.xml"]=ET.tostring(doc,xml_declaration=True,encoding="UTF-8",standalone=True)
    package["word/comments.xml"]=ET.tostring(comments,xml_declaration=True,encoding="UTF-8",standalone=True)
    with zipfile.ZipFile(CANDIDATE,"w") as z:
        for info in infos:z.writestr(info,package[info.filename])
    with zipfile.ZipFile(CANDIDATE) as z:assert z.testzip() is None
    changed=sorted(k for k in package if package[k]!=original[k]);assert changed==["word/comments.xml","word/document.xml"]
    assert "controlled comparison" not in "\n".join(plain(p) for p in body.findall("w:p",NS))
    old_doc=ET.fromstring(original["word/document.xml"]);old_body=old_doc.find("w:body",NS)
    assert [ET.tostring(t) for t in body.findall("w:tbl",NS)]==[ET.tostring(t) for t in old_body.findall("w:tbl",NS)]
    assert [ET.tostring(t) for t in doc.findall(".//m:oMath",NS)]==[ET.tostring(t) for t in old_doc.findall(".//m:oMath",NS)]
    assert plain(body[5])==plain(old_body[5])
    text="\n\n".join(plain(p) for p in body.findall("w:p",NS))
    (HERE/"candidate_readback.txt").write_text(text,encoding="utf-8")
    (HERE/"candidate_for_acronyms.md").write_text("# Abstract\n\n"+plain(body[5])+"\n\n# Body\n\n"+"\n\n".join(plain(p) for p in list(body)[7:80] if p.tag==f"{{{NS['w']}}}p")+"\n",encoding="utf-8")
    report={
        "status":"REV40_CANDIDATE_PENDING_VISUAL_QA","revision":40,"observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256":SOURCE_SHA,"backup_sha256":sha(BACKUP),"candidate_sha256":sha(CANDIDATE),
        "branch_audit_sha256":sha(HERE/"branch_provenance_audit_rev40.json"),"changed_package_parts":changed,
        "changed_paragraphs":edits,"changed_comment":{"id":2,"before":old_comment,"after":plain(comment)},
        "counts":{"body_paragraphs":len(body.findall("w:p",NS)),"tables":len(body.findall("w:tbl",NS)),"figures":len(doc.findall(".//w:drawing",NS)),"equations":len(doc.findall(".//m:oMath",NS)),"references":len([p for p in body.findall("w:p",NS) if re.match(r"^\[\d+\]",plain(p))]),"comments":len(comments.findall("w:comment",NS))},
        "abstract_regenerated":False,"tables_figures_equations_references_and_numeric_endpoints_unchanged":True,
        "new_training_runs":0,"model_forwards":0,"evaluator_runs":0,"submission_ready":False,
    }
    (HERE/"targeted_patch_rev40.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["status","candidate_sha256","counts","abstract_regenerated"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
