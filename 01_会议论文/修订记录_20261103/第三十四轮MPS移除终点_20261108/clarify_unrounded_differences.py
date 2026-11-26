"""State the rounding rule and keep the curve caption concise and explicit."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_v2.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_v3.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
assert sha256(SOURCE.read_bytes()).hexdigest() == '6b929c341b41f2d180ff5dbd1f36cf8a256c8cbfe1eda83e426ddff1377f4e36'
with ZipFile(SOURCE) as package:
    document = etree.fromstring(package.read('word/document.xml'))
paragraphs = document.findall('w:body/w:p', NS)
changes = []
for paragraph in paragraphs:
    value = ''.join(paragraph.xpath('.//w:t/text()', namespaces=NS))
    replacement = None
    if value.startswith('The primary endpoint is the EMA evaluated'):
        replacement = value.replace('APs and all other metrics follow that same AP-selected checkpoint.',
                                    'APs and all other metrics follow that same AP-selected checkpoint. '
                                    'Differences are computed from unrounded metrics.')
    elif value.startswith('Fig. 6. All 320 retained validation observations'):
        replacement = ('Fig. 6. BASE and FULL validation curves over all 320 retained rounds, without smoothing. '
                       'Circles mark round 320; stars mark the highest-AP checkpoint and its corresponding APs. '
                       'Shading covers late-restoration rounds 297–320. BASE continued after rounds 214 and '
                       '308; FULL after round 1.')
    if replacement is not None:
        nodes = paragraph.findall('.//w:t', NS)
        assert len(nodes) == 1
        nodes[0].text = replacement
        changes.append({'old': value, 'new': replacement})
assert len(changes) == 2
with ZipFile(SOURCE) as source, ZipFile(TARGET, 'x') as target:
    for item in source.infolist():
        target.writestr(item, etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
                       if item.filename == 'word/document.xml' else source.read(item.filename))
texts = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in paragraphs]
body = texts[texts.index('1. Introduction'):texts.index('References')]
abstract = next(t for t in texts if t.startswith('Small objects in unmanned aerial vehicle'))
for name, content in [('body_for_acronym_check_v3.txt', '\n\n'.join(body)),
                      ('abstract_for_acronym_check_v3.txt', abstract),
                      ('candidate_text_v3.txt', '\n\n'.join(texts))]:
    with (HERE / name).open('x', encoding='utf-8') as stream:
        stream.write(content + '\n')
report = {'status': 'ROUNDING_RULE_CLARIFIED_PENDING_FINAL_QA',
          'source_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
          'candidate_sha256': sha256(TARGET.read_bytes()).hexdigest(),
          'changes': changes, 'abstract_word_count': len(abstract.split()),
          'metrics_tables_figures_equations_or_comments_changed': False}
with (HERE / 'unrounded_differences_clarification.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ['status', 'candidate_sha256', 'abstract_word_count']}))
