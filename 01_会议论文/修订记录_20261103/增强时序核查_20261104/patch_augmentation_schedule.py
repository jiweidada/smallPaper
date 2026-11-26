"""Clarify archived augmentation intervals in the current eighth revision."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json,shutil
HERE=Path(__file__).resolve().parent;PAPER=HERE.parent.parent
SOURCE=PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP=HERE/'MPSMA-DEIM_conference_EN_revised_第八轮备份.docx'
CANDIDATE=HERE/'MPSMA-DEIM_conference_EN_revised_第九轮候选.docx'
EXPECTED='37026584852e657c15bce14d522d261762acf2695c7bd681948b56ff0ccc66ac'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
def sha(p):return sha256(p.read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED and not CANDIDATE.exists() and not BACKUP.exists()
audit=json.loads((HERE/'augmentation_schedule_audit.json').read_text(encoding='utf-8'))
assert audit['status']=='AUGMENTATION_SCHEDULE_STATICALLY_TRACED' and len(audit['settings'])==8
for item in audit['source_files']:assert sha(Path(item['path']))==item['sha256']
shutil.copy2(SOURCE,BACKUP)
with ZipFile(SOURCE) as z:original=z.read('word/document.xml')
root=etree.fromstring(original);paragraphs=root.findall('.//w:p',NS)
before=''.join(paragraphs[55].xpath('.//w:t/text()',namespaces=NS))
after=(HERE/'replacement_paragraph.txt').read_text(encoding='utf-8')
assert before.startswith('The saved augmentation configuration includes Mosaic,')
assert len(paragraphs[55].findall('.//w:t',NS))==1
assert original.count(before.encode('utf-8'))==1 and '<' not in after and '&' not in after
updated=original.replace(before.encode('utf-8'),after.encode('utf-8'))
new_root=etree.fromstring(updated)
for i,(a,b) in enumerate(zip(paragraphs,new_root.findall('.//w:p',NS))):
    if i!=55:assert etree.tostring(a)==etree.tostring(b),i
with ZipFile(SOURCE) as src,ZipFile(CANDIDATE,'x') as dst:
    for item in src.infolist():dst.writestr(item,updated if item.filename=='word/document.xml' else src.read(item.filename))
with ZipFile(SOURCE) as a,ZipFile(CANDIDATE) as b:
    assert a.namelist()==b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['word/document.xml']
counts={'tables':len(new_root.findall('.//w:tbl',NS)),'equations':len(new_root.findall('.//m:oMath',NS)),
        'inline_images':len(new_root.findall('.//wp:inline',NS)),
        **{name:len(new_root.findall('.//w:'+name,NS)) for name in ['commentRangeStart','commentRangeEnd','commentReference']}}
report={'status':'CANDIDATE_PENDING_RENDER_AND_REVIEW','revision':9,'source_sha256':EXPECTED,
 'candidate_sha256':sha(CANDIDATE),'backup_sha256':sha(BACKUP),'paragraph':55,'before':before,'after':after,
 'only_document_xml_changed':True,'all_other_paragraphs_unchanged':True,'structural_counts':counts,
 'evidence_file':str(HERE/'augmentation_schedule_audit.json'),'evidence_sha256':sha(HERE/'augmentation_schedule_audit.json'),
 'scope':'Clarify saved settings/current-source semantics. No historical-source certification, new execution or performance conclusion.'}
(HERE/'ninth_revision_pending_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','candidate_sha256','structural_counts']}))
