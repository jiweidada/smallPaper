"""Clarify retained-round scope, abstract capacity cost, and COCO terminology."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33_v2.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
assert sha256(SOURCE.read_bytes()).hexdigest() == '42b73e1672783f5c606ff1705a0c802f6868edee0c89d9cdcde19d75edfe0404'
with ZipFile(SOURCE) as package:
    document = etree.fromstring(package.read('word/document.xml'))
paragraphs = document.findall('w:body/w:p', NS)
changes = []


def edit(prefix, old, new):
    matches = [p for p in paragraphs if ''.join(p.xpath('.//w:t/text()', namespaces=NS)).startswith(prefix)]
    assert len(matches) == 1
    nodes = [node for node in matches[0].findall('.//w:t', NS) if old in (node.text or '')]
    assert len(nodes) == 1
    nodes[0].text = nodes[0].text.replace(old, new)
    changes.append({'old': old, 'new': new})


edit('Small objects in unmanned aerial vehicle',
     'after 320 epochs of training on 6,471 images',
     'over 320 retained training rounds on 6,471 images')
edit('Small objects in unmanned aerial vehicle',
     'The absolute differences are 3.53 and 3.23 percentage points. The comparison includes documented checkpoint continuations.',
     'The absolute differences are 3.53 and 3.23 percentage points. The complete model has 7.54 million trainable parameters, compared with 3.74 million for the baseline. The comparison includes documented checkpoint continuations.')
edit('Evaluation uses the local converted COCO-style',
     'Evaluation uses the local converted COCO-style bounding-box protocol [12]:',
     'Evaluation follows a local bounding-box protocol based on Common Objects in Context (COCO) [12]:')
replacement = etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
with ZipFile(SOURCE) as source, ZipFile(TARGET, 'x') as target:
    for item in source.infolist():
        target.writestr(item, replacement if item.filename == 'word/document.xml' else source.read(item.filename))
visible = '\n'.join(''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in paragraphs)
with (HERE / 'candidate_text_v2.txt').open('x', encoding='utf-8') as stream:
    stream.write(visible + '\n')
abstract = next(p for p in paragraphs if ''.join(p.xpath('.//w:t/text()', namespaces=NS)).startswith('Small objects in unmanned aerial vehicle'))
abstract_text = ''.join(abstract.xpath('.//w:t/text()', namespaces=NS))
body_lines = []
started = False
for p in paragraphs:
    value = ''.join(p.xpath('.//w:t/text()', namespaces=NS))
    if value == '1. Introduction':
        started = True
    if value == 'References':
        break
    if started and value:
        body_lines.append(value)
with (HERE / 'body_for_acronym_check.txt').open('x', encoding='utf-8') as stream:
    stream.write('\n\n'.join(body_lines) + '\n')
with (HERE / 'abstract_for_acronym_check.txt').open('x', encoding='utf-8') as stream:
    stream.write(abstract_text + '\n')
report = {'status': 'POLISHED_CANDIDATE_PENDING_FINAL_QA',
          'source_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
          'candidate_sha256': sha256(TARGET.read_bytes()).hexdigest(),
          'changes': changes, 'abstract_word_count': len(abstract_text.split()),
          'table_values_and_figures_changed': False, 'submission_ready': False}
with (HERE / 'scope_terminology_polish.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ['status', 'candidate_sha256', 'abstract_word_count']}))
