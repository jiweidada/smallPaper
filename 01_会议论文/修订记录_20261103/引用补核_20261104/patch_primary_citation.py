"""Add a direct VisDrone citation for occlusion and renumber existing citations.

Uses the current seventh revision only. Numerical configuration vectors are
excluded through an explicit paragraph/token allowlist. Other OOXML parts,
equations, media, comments and document formatting remain byte-identical.
"""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from copy import deepcopy
from lxml import etree
import json,re,shutil

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent.parent
SOURCE=PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP=HERE/'MPSMA-DEIM_conference_EN_revised_第七轮备份.docx'
CANDIDATE=HERE/'MPSMA-DEIM_conference_EN_revised_第八轮候选.docx'
EXPECTED='a494f6b7ca2b1c200232ab49fa7f3d3831b161e2552990da3b88109f7b599de2'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
MAP={1:1,9:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,10:10,11:11}
TOKENS={8:['[1]'],9:['[2]','[3]','[4]'],12:['[5]','[3]','[4]'],
        13:['[6]','[7]'],16:['[3,4]'],25:['[7]'],47:['[8]'],
        50:['[9]','[10]'],51:['[11]'],53:['[9]']}
PATTERN=re.compile(r'\[\d+(?:,\s*\d+)*\]')
def digest(p):return sha256(p.read_bytes()).hexdigest()
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
def mapped(match):
    return re.sub(r'\d+',lambda n:str(MAP[int(n.group())]),match.group())
def counts(root):
    return {'tables':len(root.findall('.//w:tbl',NS)),
            'equations':len(root.findall('.//m:oMath',NS)),
            'inline_images':len(root.findall('.//wp:inline',NS)),
            **{name:len(root.findall('.//w:'+name,NS)) for name in ['commentRangeStart','commentRangeEnd','commentReference']}}

assert digest(SOURCE)==EXPECTED and not CANDIDATE.exists()
if BACKUP.exists():assert digest(BACKUP)==EXPECTED
else:shutil.copy2(SOURCE,BACKUP)
with ZipFile(SOURCE) as z:raw=z.read('word/document.xml')
root=etree.fromstring(raw)
assert etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)==raw
original=deepcopy(root)
paras=root.findall('.//w:p',NS)
changes=[]
for index,tokens in TOKENS.items():
    para=paras[index];before=text(para)
    assert PATTERN.findall(before)==tokens,(index,before)
    nodes=para.findall('.//w:t',NS)
    assert [token for node in nodes for token in PATTERN.findall(node.text or '')]==tokens,'Split citation token'
    for node in nodes:node.text=PATTERN.sub(mapped,node.text or '')
    if index==8:
        matches=[node for node in nodes if '[1]' in (node.text or '')]
        assert len(matches)==1
        matches[0].text=matches[0].text.replace('[1]','[1], [2]')
    changes.append({'paragraph':index,'before':before,'after':text(para)})

body=root.find('w:body',NS)
refs=[]
for para in body.findall('w:p',NS):
    m=re.match(r'^\[(\d+)\] ',text(para))
    if m:refs.append((int(m.group(1)),body.index(para),para))
assert [number for number,_,_ in refs]==list(range(1,12))
old_bibliography={number:text(para) for number,_,para in refs}
for number,_,para in refs:
    nodes=para.findall('.//w:t',NS)
    assert (nodes[0].text or '').startswith('['+str(number)+'] ')
    nodes[0].text=nodes[0].text.replace('['+str(number)+']','['+str(MAP[number])+']',1)
new_order=sorted(refs,key=lambda v:MAP[v[0]])
for (_,slot,_),(number,_,para) in zip(refs,new_order):body[slot]=deepcopy(para)
new_refs=[text(p) for p in body.findall('w:p',NS) if re.match(r'^\[\d+\] ',text(p))]
assert [int(re.match(r'^\[(\d+)\]',s).group(1)) for s in new_refs]==list(range(1,12))
for old,new in MAP.items():
    assert new_refs[new-1]==old_bibliography[old].replace('['+str(old)+']','['+str(new)+']',1)

# All non-citation paragraphs, numerical vectors and math elements are intact.
new_paras=root.findall('.//w:p',NS)
for i,(before,after) in enumerate(zip(original.findall('.//w:p',NS),new_paras)):
    if i not in TOKENS and not re.match(r'^\[\d+\] ',text(before)):
        assert etree.tostring(before)==etree.tostring(after),('Unintended paragraph change',i)
assert counts(root)==counts(original)
assert [etree.tostring(n) for n in original.findall('.//m:oMath',NS)]==[etree.tostring(n) for n in root.findall('.//m:oMath',NS)]
order=[]
for i,para in enumerate(new_paras):
    if i not in TOKENS:continue
    for token in PATTERN.findall(text(para)):
        for number in map(int,re.findall(r'\d+',token)):
            if number not in order:order.append(number)
assert order==list(range(1,12)),order
new_xml=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(SOURCE) as src,ZipFile(CANDIDATE,'x') as dst:
    for item in src.infolist():dst.writestr(item,new_xml if item.filename=='word/document.xml' else src.read(item.filename))
with ZipFile(SOURCE) as a,ZipFile(CANDIDATE) as b:
    assert a.namelist()==b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['word/document.xml']
report={'status':'CANDIDATE_PENDING_RENDER_AND_REVIEW','revision':8,'source_sha256':EXPECTED,
 'candidate_sha256':digest(CANDIDATE),'backup_sha256':digest(BACKUP),
 'citation_number_mapping_old_to_new':MAP,'first_appearance_order':order,'references':11,
 'changes':changes,'structural_counts':counts(root),'only_document_xml_changed':True,
 'noncitation_paragraphs_and_numerical_vectors_unchanged':True,
 'evidence':'Existing VisDrone-DET2019 original, Occlusion paragraph (old reference 9, now reference 2).',
 'limits':'No training results added. Full fair-pair training, final endpoints and submission acceptance remain pending.'}
(HERE/'eighth_revision_pending_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['status','candidate_sha256','first_appearance_order','references','structural_counts']}))
