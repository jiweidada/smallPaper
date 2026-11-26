"""Apply the relevant first-use terminology findings from the advisory check."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33_v2.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33_v3.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
assert sha256(SOURCE.read_bytes()).hexdigest() == 'a54c7ab62eab58fbd5deac2a1a136b9937481ae6637b19e1c11ba4477c8bff25'
with ZipFile(SOURCE) as package:
    document = etree.fromstring(package.read('word/document.xml'))
paragraphs = document.findall('w:body/w:p', NS)
changes = []
for old, new in [
    ('under frozen local COCO-style settings.',
     'under a frozen local evaluation based on Common Objects in Context (COCO).'),
    ('AR@100 denotes average recall at maxDets=100.',
     'Average recall (AR) is reported at maxDets=100 as AR@100.'),
    ('class-aware matching and maxDets=[1, 10, 100].',
     'class-aware matching and maxDets=[1, 10, 100] per image and category.'),
    ('FP32 with mixed precision and TF32 disabled, batch size 4, and one NVIDIA RTX 3090 GPU per run.',
     '32-bit floating point (FP32) with automatic mixed precision and TensorFloat-32 (TF32) disabled, '
     'batch size 4, and one NVIDIA RTX 3090 graphics processing unit (GPU) per run.'),
]:
    matches = [node for paragraph in paragraphs for node in paragraph.findall('.//w:t', NS) if old in (node.text or '')]
    assert len(matches) == 1, old
    matches[0].text = matches[0].text.replace(old, new)
    changes.append({'old': old, 'new': new})
with ZipFile(SOURCE) as source, ZipFile(TARGET, 'x') as target:
    for item in source.infolist():
        target.writestr(item, etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
                       if item.filename == 'word/document.xml' else source.read(item.filename))
texts = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in paragraphs]
body = texts[texts.index('1. Introduction'):texts.index('References')]
abstract = next(t for t in texts if t.startswith('Small objects in unmanned aerial vehicle'))
for name, value in [('candidate_text_final.txt', '\n\n'.join(texts)),
                    ('body_for_acronym_check_final.txt', '\n\n'.join(body)),
                    ('abstract_for_acronym_check_final.txt', abstract)]:
    with (HERE / name).open('x', encoding='utf-8') as stream:
        stream.write(value + '\n')
report = {'status': 'TERMINOLOGY_CLARIFIED_PENDING_FINAL_VISUAL_QA',
          'source_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
          'candidate_sha256': sha256(TARGET.read_bytes()).hexdigest(),
          'changes': changes, 'abstract_word_count': len(abstract.split()),
          'tables_and_figures_changed': False}
with (HERE / 'acronym_terminology_changes.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ['status', 'candidate_sha256', 'abstract_word_count']}))
