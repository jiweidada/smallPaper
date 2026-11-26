"""Keep the reference list together after the added recovery disclosure."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

here = Path(__file__).resolve().parent
source = here / 'MPSMA-DEIM_conference_EN_revised_rev36.docx'
target = here / 'MPSMA-DEIM_conference_EN_revised_rev36_v2.docx'
assert sha256(source.read_bytes()).hexdigest() == '5f87d5819825ee5e7c456ada53d8c1c87ce4163e4aeac30dccf47882be04ea50'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ns = {'w': W}
q = lambda name: '{' + W + '}' + name
with ZipFile(source) as archive:
    document = etree.fromstring(archive.read('word/document.xml'))
    matches = [p for p in document.findall('w:body/w:p', ns)
               if ''.join(p.xpath('.//w:t/text()', namespaces=ns)) == 'References']
    assert len(matches) == 1
    paragraph = matches[0]
    properties = paragraph.find('w:pPr', ns)
    if properties is None:
        properties = etree.Element(q('pPr'))
        paragraph.insert(0, properties)
    assert properties.find('w:pageBreakBefore', ns) is None
    page_break = etree.Element(q('pageBreakBefore'))
    position = 0
    while position < len(properties) and properties[position].tag in [q('pStyle'), q('keepNext'), q('keepLines')]:
        position += 1
    properties.insert(position, page_break)
    with ZipFile(target, 'x') as output:
        for item in archive.infolist():
            value = etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True) if item.filename == 'word/document.xml' else archive.read(item.filename)
            output.writestr(item, value)
renderer = (here / 'render_candidate.ps1').read_text(encoding='utf-8')
renderer = renderer.replace('_rev36.docx', '_rev36_v2.docx').replace("'rendered_word'", "'rendered_word_v2'").replace("'render_metadata.json'", "'render_metadata_v2.json'")
(here / 'render_candidate_v2.ps1').write_text(renderer, encoding='utf-8')
report = {'status': 'LAYOUT_ITERATION_PENDING_VISUAL_QA',
    'reason': 'Initial render put only references12 and13 on page9. Keep the complete reference list on its own page without reducing fonts or removing evidence.',
    'candidate_sha256': sha256(target.read_bytes()).hexdigest(), 'semantic_text_changed_from_first_candidate': False,
    'layout_change': 'References paragraph page_break_before=True', 'conference_page_limit_selected': False}
(here / 'layout_iteration_v2.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
