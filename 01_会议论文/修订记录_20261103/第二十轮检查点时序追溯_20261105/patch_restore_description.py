"""Clarify the inspected solver's stage-boundary restore and N2's conditional counters."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
CURRENT=PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP=HERE/'MPSMA-DEIM_conference_EN_revised_第十九轮备份.docx'
CANDIDATE=HERE/'MPSMA-DEIM_conference_EN_revised_第二十轮候选.docx'
PENDING=HERE/'twentieth_revision_pending_qa.json'
OLD_SHA='6c603477dd95225525e4505fbf8c126f9b349e1ba5fa6cffaf87df4672238969'
OLD=('The inspected solver selects a stage-one checkpoint by validation AP and may restore it during later training, '
     'which can reset last_epoch; whether this solver was used historically is unverified.')
NEW=('The inspected solver selects a stage-one checkpoint by validation AP, restores it at the stage boundary '
     'and may restore it after a later non-improving epoch, resetting last_epoch. With 1,617 EMA updates per epoch '
     'and no additional restores, N2’s logged AP sequence and saved counters are consistent with this rule; '
     'historical use of this solver remains unverified.')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(CURRENT)==OLD_SHA and not any(p.exists() for p in [BACKUP,CANDIDATE,PENDING])
    timeline_path=HERE/'n2_conditional_timeline.json'
    proof_path=HERE/'n2_source_semantics_verification.json'
    timeline=json.loads(timeline_path.read_text(encoding='utf-8'))
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    assert timeline['status']=='N2_METADATA_CONSISTENT_WITH_CONDITIONAL_SOLVER_BOOKKEEPING'
    assert proof['status']=='INSPECTED_CODE_AND_N2_COUNTER_ARITHMETIC_CONSISTENT'
    assert proof['timeline_sha256']==sha(timeline_path) and all(proof['static_source_checks'].values())
    assert timeline['assumptions']['assumed_steps_per_epoch']==1617
    assert timeline['stage1_saved_epoch']==291 and timeline['stage1_saved_ema_updates']==472164
    assert timeline['saved_best_stage2_epoch_field']==296 and timeline['saved_best_stage2_ema_updates']==480249
    assert all(x['is_new_global_best'] for x in timeline['conditional_trajectory'])
    for source in proof['source_files']:
        assert sha(Path(source['path']))==sha(Path(source['snapshot']))==source['sha256']
    with ZipFile(CURRENT) as z:
        old_xml=z.read('word/document.xml')
    root=etree.fromstring(old_xml)
    paragraphs=root.findall('.//w:p',NS)
    positions=[i for i,p in enumerate(paragraphs) if OLD in ''.join(p.xpath('.//w:t/text()',namespaces=NS))]
    assert len(positions)==1 and old_xml.count(OLD.encode('utf-8'))==1
    assert not any(c in NEW for c in '<&')
    new_xml=old_xml.replace(OLD.encode('utf-8'),NEW.encode('utf-8'))
    revised=etree.fromstring(new_xml)
    other=revised.findall('.//w:p',NS)
    assert len(paragraphs)==len(other)
    assert [i for i,(a,b) in enumerate(zip(paragraphs,other)) if etree.tostring(a)!=etree.tostring(b)]==positions
    shutil.copy2(CURRENT,BACKUP)
    assert sha(BACKUP)==OLD_SHA
    with ZipFile(CURRENT) as source,ZipFile(CANDIDATE,'x') as target:
        for item in source.infolist():
            target.writestr(item,new_xml if item.filename=='word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as a,ZipFile(CANDIDATE) as b:
        assert b.testzip() is None and a.namelist()==b.namelist()
        assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['word/document.xml']
    counts={'tables':len(revised.findall('.//w:tbl',NS)),'equations':len(revised.findall('.//m:oMath',NS)),
            'inline_images':len(revised.findall('.//wp:inline',NS)),
            **{key:len(revised.findall('.//w:'+key,NS)) for key in ['commentRangeStart','commentRangeEnd','commentReference']}}
    assert counts=={'tables':6,'equations':6,'inline_images':7,'commentRangeStart':5,'commentRangeEnd':5,'commentReference':5}
    report={'status':'CANDIDATE_PENDING_RENDER_AND_REVIEW','revision':20,'source_sha256':OLD_SHA,
            'backup_sha256':sha(BACKUP),'candidate_sha256':sha(CANDIDATE),'changed_paragraph_index':positions[0],
            'old_text':OLD,'new_text':NEW,'only_document_xml_changed':True,'structural_counts':counts,
            'timeline_sha256':sha(timeline_path),'source_semantics_sha256':sha(proof_path),
            'scope':'Clarifies inspected stage-boundary/non-best restore and conditional N2 counter consistency. No experimental result, table, figure or formula changes.'}
    with PENDING.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'candidate_sha256':report['candidate_sha256'],'changed_paragraph_index':positions[0]},ensure_ascii=False))

if __name__=='__main__':
    main()
