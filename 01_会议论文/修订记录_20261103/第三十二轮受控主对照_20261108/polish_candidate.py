"""Clarify dataset roles and shorten repeated evidence discussion."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev32.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev32_v2.docx'
SOURCE_SHA = 'dbfaf77ed64486410ead771055f9dc6fb9b3204250c4a3e3a55a000ad5cab0e9'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
spec = importlib.util.spec_from_file_location(
    'prior_controlled_patch', HERE.parent / '第三十轮主对照口径_20261106/patch_controlled_comparison.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def main():
    assert sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA and not TARGET.exists()
    protocol = json.loads((HERE.parent.parent / '远程公平对照_20261103/execution_protocol.json')
                          .read_text(encoding='utf-8'))
    assert protocol['runtime']['torch'] == '2.5.1+cu124'
    assert protocol['runtime']['cuda'] == '12.4'
    assert protocol['runtime']['packages']['faster-coco-eval'] == '1.8.0'
    with ZipFile(SOURCE) as archive:
        document = etree.fromstring(archive.read('word/document.xml'))
    paragraphs = document.findall('w:body/w:p', NS)
    changes = []

    def replace(prefix, replacement):
        matches = [paragraph for paragraph in paragraphs if prior.text(paragraph).startswith(prefix)]
        assert len(matches) == 1
        old = prior.text(matches[0])
        prior.edit(paragraphs, prefix, old, replacement)
        changes.append({'old': old, 'new': replacement})

    replace('Small objects in unmanned aerial vehicle',
        'Small objects in unmanned aerial vehicle (UAV) images provide limited visual evidence and occur amid clutter '
        'and scale variation. We present MPSMA-DEIM, an adaptation of DEIM-D-FINE that combines a Multi-branch '
        'Progressive Stem (MPS) with Multi-scale Context-aware Attention (MSCA). The stem merges detail, downsampling '
        'and high-frequency branches into stride-four features. The attention design uses pooled context at an '
        'intermediate stage and dilated context at the deepest stage, with SimAM-derived energy gates. We compare '
        'the complete n configuration with its original baseline after 320 epochs of training on 6,471 images using '
        'seed 0, FP32 and batch size 4. Evaluation uses all 548 VisDrone2019 validation images under frozen local '
        'COCO-style settings. The final-round exponential moving average (EMA) achieves average precision (AP) '
        'of 0.2255 and small-object AP (APs) of 0.1539, compared with baseline values of 0.1902 and 0.1215. The '
        'absolute differences are 3.53 and 3.23 percentage points. The comparison includes documented checkpoint '
        'continuations. Component-specific effects are being evaluated separately.')
    prefix = 'BASE is the original n baseline'
    old_protocol = next(prior.text(paragraph) for paragraph in paragraphs
                        if prior.text(paragraph).startswith(prefix))
    replace(prefix, old_protocol.replace(
        'A predefined set of 554 shared tensors has identical initial values.',
        'Both runs use one RTX 3090 GPU, PyTorch 2.5.1+cu124, CUDA 12.4 and faster-coco-eval 1.8.0. '
        'A predefined set of 554 shared tensors has identical initial values.'))
    replace('At the prespecified primary endpoint',
        'At the prespecified primary endpoint, FULL exceeds BASE by 3.53 percentage points in AP and 3.23 in APs. '
        'The highest-AP secondary endpoints occur at completed rounds 303 and 305, respectively. FULL has 7,540,942 '
        'trainable parameters, compared with 3,737,611 for BASE. This measures the entire changed backbone '
        'configuration, including its additional capacity and aggregation and activation settings. Component-removal '
        'experiments are in progress.')
    replace('The 48 retained periodic or latest evaluation arrays',
        'All 48 archived periodic or latest evaluation arrays match their corresponding 12-metric log rows. Only '
        'N2 has an archived array for its selected best row, and none of the 48 arrays contains a checkpoint hash. '
        'Checkpoint indices and the inspected restoration logic cannot establish historical weight-to-metric '
        'identity when the training implementation is unverified. Section 4.5 supplies separate current evaluations '
        'of retained N0, N4 and S1 weights. Historical runtime identity and training-condition differences remain limitations.')
    replace('BASE and FULL initially resumed after',
        'BASE and FULL initially resumed after 214 and one completed round, respectively, without saved main-process '
        'random states, which were reinitialized with seed 0. BASE resumed again after round 308 with its recorded '
        'random states, rerunning the interrupted partial round 309. The FULL per-batch learning-rate record starts '
        'at round 2. These observed continuations and their retained endpoints have been verified together.')
    replace('MPSMA-DEIM combines a three-path shallow stem',
        'MPSMA-DEIM combines a multi-branch shallow stem with stage-specific contextual gating in a DEIM-D-FINE '
        'backbone. In the seed-0 n comparison, the prespecified 320-round EMA endpoint achieves AP 0.2255 and '
        'APs 0.1539, exceeding BASE by 3.53 and 3.23 percentage points. These observations apply to the common '
        'local recipe with documented checkpoint continuations. Archived n and s runs supply additional '
        'configuration observations. The two conditional removal runs are ongoing, so component-specific effects remain open.')

    revised = etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(SOURCE) as source, ZipFile(TARGET, 'x') as target:
        for item in source.infolist():
            target.writestr(item, revised if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(SOURCE) as before, ZipFile(TARGET) as after:
        assert before.namelist() == after.namelist() and after.testzip() is None
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ['word/document.xml']
    report = {'status': 'POLISHED_CANDIDATE_PENDING_VISUAL_QA', 'source_sha256': SOURCE_SHA,
              'candidate_sha256': sha256(TARGET.read_bytes()).hexdigest(), 'paragraph_changes': changes}
    with (HERE / 'paired_revision_polish.json').open('x', encoding='utf-8') as out:
        out.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256']}))


if __name__ == '__main__':
    main()
