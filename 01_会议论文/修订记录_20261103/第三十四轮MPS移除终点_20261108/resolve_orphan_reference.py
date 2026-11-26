"""Tighten repeated discussion to keep the final reference with its list."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_v3.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_v4.docx'
assert sha256(SOURCE.read_bytes()).hexdigest() == 'c1baf78681a4e63385889a5cf721c4949a8046aa0935b114d4370a6962c3b459'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(SOURCE) as source:
    document = etree.fromstring(source.read('word/document.xml'))
paragraphs = document.findall('w:body/w:p', NS)
value = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
changes = []

def replace(prefix, new):
    matches = [p for p in paragraphs if value(p).startswith(prefix)]
    assert len(matches) == 1, prefix
    p = matches[0]
    assert not p.xpath('.//w:drawing', namespaces=NS)
    anchors = [etree.tostring(n) for n in p.xpath('.//w:commentRangeStart|.//w:commentRangeEnd|.//w:commentReference', namespaces=NS)]
    old = value(p)
    nodes = p.findall('.//w:t', NS)
    nodes[0].text = new
    for node in nodes[1:]:
        node.text = ''
    assert anchors == [etree.tostring(n) for n in p.xpath('.//w:commentRangeStart|.//w:commentRangeEnd|.//w:commentReference', namespaces=NS)]
    changes.append({'old': old, 'new': new})

replace('We use one seed and one validation split,',
        'One seed and one validation split provide no estimate of variability or independent test performance. '
        'Validation guides both restoration and secondary checkpoint selection. Capacity differs between '
        'configurations. The MSCA-removal endpoint, current n-model latency and onboard benchmarks remain unavailable.')
replace('BASE continued from round 214 after excluding 11 later rows',
        'BASE resumed after round 214, excluding 11 later rows from the interrupted branch; FULL resumed after '
        'round 1. Both anchors lacked main-process random states and were reseeded with 0. BASE resumed again '
        'after round 308 with saved random states, rerunning a partial round 309. NO_MPS began at round zero. '
        'The FULL per-batch learning-rate record begins at round 2.')
replace('MPSMA-DEIM combines a multi-branch stem and stage-specific contextual gating',
        'MPSMA-DEIM combines a multi-branch stem and stage-specific contextual gates in DEIM-D-FINE. At round '
        '320, FULL achieves AP 0.2255 and APs 0.1539: 3.53 and 3.23 percentage points above BASE, with 2.02 '
        'times its trainable parameters. FULL exceeds NO_MPS by 1.57 and 1.77 points in AP and APs, respectively, '
        'but has lower large-object recall. These single-seed results concern the documented continuations '
        'and local COCO-style evaluation. MSCA removal remains under evaluation.')

with ZipFile(SOURCE) as source, ZipFile(TARGET, 'x') as target:
    for item in source.infolist():
        target.writestr(item, etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
                       if item.filename == 'word/document.xml' else source.read(item.filename))
texts = [value(p) for p in paragraphs]
body = texts[texts.index('1. Introduction'):texts.index('References')]
abstract = next(t for t in texts if t.startswith('Small objects in unmanned aerial vehicle'))
for name, content in [('body_for_acronym_check_v4.txt', '\n\n'.join(body)),
                      ('abstract_for_acronym_check_v4.txt', abstract),
                      ('candidate_text_v4.txt', '\n\n'.join(texts))]:
    with (HERE / name).open('x', encoding='utf-8') as stream:
        stream.write(content + '\n')
report = {'status': 'REPEATED_PROSE_TIGHTENED_PENDING_RENDER_QA',
          'source_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
          'candidate_sha256': sha256(TARGET.read_bytes()).hexdigest(),
          'source_layout_issue': 'Final reference 13 alone on page 9 of v3.',
          'changes': changes, 'abstract_word_count': len(abstract.split()),
          'tables_figures_equations_comments_references_or_styles_changed': False,
          'words_removed': sum(len(c['old'].split()) - len(c['new'].split()) for c in changes)}
with (HERE / 'orphan_reference_resolution.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
renderer = (HERE / 'render_final.ps1').read_text(encoding='utf-8-sig')
renderer = renderer.replace('_rev34_v3.docx', '_rev34_v4.docx').replace('rendered_word_final', 'rendered_word_v4').replace('render_metadata_final.json', 'render_metadata_v4.json')
with (HERE / 'render_v4.ps1').open('x', encoding='utf-8') as stream:
    stream.write(renderer)
print(json.dumps({k: report[k] for k in ['status', 'candidate_sha256', 'words_removed']}))
