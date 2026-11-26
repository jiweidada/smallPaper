"""Add the verified saved-array provenance boundary to one manuscript paragraph."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
CURRENT=PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP=HERE/'MPSMA-DEIM_conference_EN_revised_第十八轮备份.docx'
CANDIDATE=HERE/'MPSMA-DEIM_conference_EN_revised_第十九轮候选.docx'
PENDING=HERE/'nineteenth_revision_pending_qa.json'
OLD_SHA='888f5a17e991885b8581b16684e0a1d28ce06ab5ecd3f7176d2acd22e2551420'
OLD='It does not yet isolate shallow-detail preservation, contextual weighting or activation effects.'
NEW=(OLD+' The 48 retained periodic and latest evaluation arrays yield 12-metric summaries matching their '
     'corresponding log rows within 10⁻¹²; only N2’s selected best row has such a periodic array. '
     'The arrays contain no explicit checkpoint hash.')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert digest(CURRENT)==OLD_SHA and not any(p.exists() for p in [BACKUP,CANDIDATE,PENDING])
    proof=HERE/'independent_saved_array_verification.json'
    fields=HERE/'saved_evaluation_binding_fields.json'
    audit=HERE/'saved_array_log_matching.json'
    p=json.loads(proof.read_text(encoding='utf-8'))
    q=json.loads(fields.read_text(encoding='utf-8'))
    assert p['status']=='ALL_48_CORRESPONDING_ROWS_VERIFIED' and p['arrays_checked']==48
    assert p['maximum_abs_difference']<1e-12 and p['n2_independent_fmean_max_error']<1e-12
    assert [(x['run'],x['epoch']) for x in p['selected_best_log_arrays']]==[('N2',300)]
    assert p['source_audit_sha256']==q['source_sha256']==digest(audit)
    assert q['files']==49 and not q['files_with_checkpoint_or_hash_named_fields']
    with ZipFile(CURRENT) as z:
        old_xml=z.read('word/document.xml')
    root=etree.fromstring(old_xml)
    paragraphs=root.findall('.//w:p',NS)
    positions=[i for i,p in enumerate(paragraphs) if OLD in ''.join(p.xpath('.//w:t/text()',namespaces=NS))]
    assert len(positions)==1 and old_xml.count(OLD.encode('utf-8'))==1
    assert not any(c in NEW for c in '<&')
    new_xml=old_xml.replace(OLD.encode('utf-8'),NEW.encode('utf-8'))
    revised=etree.fromstring(new_xml)
    revised_paragraphs=revised.findall('.//w:p',NS)
    assert len(revised_paragraphs)==len(paragraphs)
    assert [i for i,(a,b) in enumerate(zip(paragraphs,revised_paragraphs)) if etree.tostring(a)!=etree.tostring(b)]==positions
    shutil.copy2(CURRENT,BACKUP)
    assert digest(BACKUP)==OLD_SHA
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
    report={'status':'CANDIDATE_PENDING_RENDER_AND_REVIEW','revision':19,'source_sha256':OLD_SHA,
            'backup_sha256':digest(BACKUP),'candidate_sha256':digest(CANDIDATE),'changed_paragraph_index':positions[0],
            'old_text':OLD,'new_text':NEW,'only_document_xml_changed':True,'structural_counts':counts,
            'saved_array_proof_sha256':digest(proof),'binding_fields_sha256':digest(fields),
            'scope':'Add saved-array provenance and checkpoint-hash limitation only; no table, figure, formula or result changed.'}
    with PENDING.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'candidate_sha256':report['candidate_sha256'],'changed_paragraph_index':positions[0],'counts':counts},ensure_ascii=False))

if __name__=='__main__':
    main()
