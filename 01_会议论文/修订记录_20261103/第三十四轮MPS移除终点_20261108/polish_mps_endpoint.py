"""Remove redundant discussion while preserving all material scope limits."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_v2.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
assert sha256(SOURCE.read_bytes()).hexdigest() == '01b9a3f30a6b4736d0d1cc456d17e2128faaa08830427c4aff284228126998e0'
with ZipFile(SOURCE) as package:
    document = etree.fromstring(package.read('word/document.xml'))
paragraphs = document.findall('w:body/w:p', NS)
changes = []


def value(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))


def edit(prefix, new):
    matches = [p for p in paragraphs if value(p).startswith(prefix)]
    assert len(matches) == 1
    nodes = matches[0].findall('.//w:t', NS)
    assert len(nodes) == 1
    changes.append({'old': nodes[0].text, 'new': new})
    nodes[0].text = new


abstract = next(value(p) for p in paragraphs if value(p).startswith('Small objects in unmanned aerial vehicle'))
edit('Small objects in unmanned aerial vehicle', abstract.replace(
    'Replacing MPS with the standard SiLU stem gives AP 0.2099 and APs 0.1362.',
    'The run replacing MPS with a standard stem using the sigmoid-weighted linear unit (SiLU) reaches AP 0.2099 and APs 0.1362.'))
edit('FULL has 7,540,942 trainable parameters,',
     'FULL has 7,540,942 trainable parameters, versus 3,737,611 for BASE (2.02 times). Table 1 compares the '
     'complete configuration and its additional capacity; it does not isolate the effects of the stem, '
     'contextual gates, aggregation projections or activations.')
edit('NO_MPS completes all 320 rounds with AP',
     'At fixed round 320, FULL exceeds NO_MPS by 1.57 percentage points in AP and 1.77 in APs (Table 2). '
     'NO_MPS reaches its highest AP at round 320, so its primary and secondary checkpoints coincide.')
edit('The stem replacement removes 5,325 trainable parameters.',
     'The stem replacement removes 5,325 trainable parameters. FULL has large-object recall 0.5446, '
     'compared with 0.5615 for NO_MPS, a difference of -1.69 percentage points.')
edit('This comparison concerns the stem while the remaining FULL settings',
     'This comparison concerns MPS with the remaining FULL settings present. Its interpretation is '
     'conditional on the realized training histories and does not isolate mechanism from capacity.')
edit('The NO_MSCA run is ongoing, so its outcome is not reported.',
     'NO_MSCA remains in progress. BASE has different aggregation paths and activations, so it is not '
     'a joint-removal control. The design provides no MPS–MSCA interaction estimate.')
edit('The comparison uses one seed and one validation split.',
     'We use one seed and one validation split, which guides the restoration policy and secondary '
     'checkpoint selection. The fixed primary round provides no independent test estimate or uncertainty '
     'across runs. Capacity differs between configurations; the MSCA-removal endpoint and current '
     'n-model latency and onboard benchmarks remain unavailable.')
edit('The retained trajectories include interrupted training.',
     'BASE continued from round 214 after excluding 11 later rows from the interrupted original branch; '
     'FULL continued from round 1. Those first anchors lacked main-process random states and were '
     'reseeded with 0. BASE later continued from round 308 with saved random states, rerunning a partial '
     'round 309. NO_MPS started from round zero. The FULL per-batch learning-rate record starts at round 2.')
edit('MPSMA-DEIM combines a multi-branch shallow stem',
     'MPSMA-DEIM combines a multi-branch stem and stage-specific contextual gating in DEIM-D-FINE. '
     'At fixed round 320, the complete n configuration achieves AP 0.2255 and APs 0.1539, exceeding '
     'BASE by 3.53 and 3.23 percentage points with 2.02 times its trainable parameters. In the conditional '
     'stem comparison, FULL is higher by 1.57 points in AP and 1.77 in APs, and lower in large-object '
     'recall. These observations from one seed apply to the documented continuations and local '
     'COCO-style evaluation. The MSCA-removal run is ongoing.')
with ZipFile(SOURCE) as source, ZipFile(TARGET, 'x') as target:
    for item in source.infolist():
        target.writestr(item, etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
                       if item.filename == 'word/document.xml' else source.read(item.filename))
texts = [value(p) for p in paragraphs]
body = texts[texts.index('1. Introduction'):texts.index('References')]
abstract = next(t for t in texts if t.startswith('Small objects in unmanned aerial vehicle'))
for name, content in [('body_for_acronym_check_final.txt', '\n\n'.join(body)),
                      ('abstract_for_acronym_check_final.txt', abstract),
                      ('candidate_text_final.txt', '\n\n'.join(texts))]:
    with (HERE / name).open('x', encoding='utf-8') as stream:
        stream.write(content + '\n')
report = {'status': 'POLISHED_MPS_ENDPOINT_CANDIDATE_PENDING_QA',
          'source_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
          'candidate_sha256': sha256(TARGET.read_bytes()).hexdigest(),
          'changes': changes, 'abstract_word_count': len(abstract.split()),
          'tables_figures_equations_or_comments_changed': False,
          'words_removed_from_repeated_discussion': sum(len(c['old'].split()) - len(c['new'].split()) for c in changes)}
with (HERE / 'mps_endpoint_polish.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ['status', 'candidate_sha256', 'abstract_word_count', 'words_removed_from_repeated_discussion']}))
