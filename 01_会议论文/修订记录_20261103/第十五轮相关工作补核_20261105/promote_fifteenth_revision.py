"""Promote the reviewed literature update and preserve the citation-number history."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from zipfile import ZipFile
from lxml import etree
import copy
import json
import os
import re
import shutil

HERE = Path(__file__).resolve().parent
REC, PAPER = HERE.parent, HERE.parent.parent
REMOTE = PAPER / '远程公平对照_20261103'
INDEX = REC / '主要结论证据索引'
S14 = REC / '第十四轮优化器分组核查_20261105'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十五轮候选_v2.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮备份.docx'
OLD = '15c62d7f363775645d73e29710366b32309f2b500ec6a08bd524be826b9eb3f8'
NEW = '617ffd2024d42affb7f6b66f88cdca67204a0de3fef97946ffebf0c90f896984'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())

def replace_text(path, text):
    temp = path.with_name(path.name + '.fifteenth-promotion.tmp')
    assert not temp.exists()
    with temp.open('x', encoding='utf-8', newline='\n') as out:
        out.write(text)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)

def ref(path):
    return {'path': str(path), 'sha256': sha(path)}

def main():
    assert sha(CURRENT) == sha(BACKUP) == OLD and sha(CANDIDATE) == NEW
    original = PAPER / '稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup = PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert sha(original) == sha(original_backup) == ORIGINAL
    patch = read(HERE / 'fifteenth_revision_v2_pending_qa.json')
    qa_path = HERE / 'fifteenth_revision_visual_qa.json'
    qa = read(qa_path)
    evidence_path = HERE / 'literature_evidence.json'
    evidence = read(evidence_path)
    explanation = HERE / '相关工作补核说明.md'
    assert patch['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert qa['status'] == 'VISUAL_CITATION_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages'] == list(range(1, 11))
    assert qa['citation_first_appearance_order'] == list(range(1, 14))
    assert sha(evidence_path) == patch['literature_evidence_sha256'] == qa['literature_evidence_sha256']
    assert evidence['status'] == 'TWO_RELEVANT_PRIMARY_PUBLISHER_RECORDS_CHECKED'
    assert not evidence['cross_paper_metric_comparisons'] and not evidence['new_experiments']
    old_citation_archive = REC / 'citation_archive.json'
    old_citations = read(old_citation_archive)
    assert [item['reference'] for item in old_citations] == list(range(1, 12))
    with ZipFile(CANDIDATE) as z:
        root = etree.fromstring(z.read('word/document.xml'))
    refs = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in root.find('w:body', NS).findall('w:p', NS)]
    refs = [p for p in refs if re.match(r'^\[\d+\] ', p)]
    assert [int(re.match(r'^\[(\d+)\]', p).group(1)) for p in refs] == list(range(1, 14))
    prior_catalog_path = S14 / 'manuscript_version_locations.json'
    prior_catalog = read(prior_catalog_path)
    for item in prior_catalog['versions']:
        assert sha(Path(item['retained_file']['path'])) == item['sha256']
    prior_indices = [INDEX / ('主要结论证据索引_第十四轮' + suffix) for suffix in ['.txt', '.json']]
    assert read(prior_indices[1])['manuscript']['sha256'] == OLD
    diagnostic = REMOTE / 'diagnostic_completion_receipt_20261105.json'
    assert read(diagnostic)['status'] == 'ONE_AUTHORIZED_DIAGNOSTIC_COMPLETED_AND_EVIDENCE_VERIFIED'

    outputs = [HERE / 'fifteenth_revision_patch.json', HERE / 'manuscript_version_locations.json',
               HERE / 'citation_archive_revision15.json', INDEX / '主要结论证据索引_第十五轮.json',
               INDEX / '主要结论证据索引_第十五轮.txt']
    assert not any(path.exists() for path in outputs)
    text_paths = [PAPER / '当前修订入口.txt', REMOTE / '结果并入与验收清单.txt',
                  REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt', REC / '引用核查记录.txt']
    text_backups = [HERE / ('text_before_revision15_%d.txt' % i) for i in range(len(text_paths))]
    assert not any(path.exists() for path in text_backups)
    old_text = [path.read_text(encoding='utf-8-sig') for path in text_paths]
    assert old_text[1].count('当前稿件SHA256：' + OLD) == 1
    assert old_text[3].count('稿件现为第十四轮修订版') == 1
    assert '共11条引用' in old_text[4]
    for source, destination in zip(text_paths, text_backups):
        shutil.copy2(source, destination)

    temp = CURRENT.with_name(CURRENT.stem + '_revision15_promotion.tmp.docx')
    assert not temp.exists()
    with CANDIDATE.open('rb') as src, temp.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    assert sha(temp) == NEW and sha(CURRENT) == OLD
    os.replace(temp, CURRENT)
    assert sha(CURRENT) == NEW and sha(BACKUP) == OLD and sha(original) == sha(original_backup)
    now = datetime.now(timezone.utc).isoformat()
    promoted = dict(patch, status='PROMOTED_AFTER_VISUAL_CITATION_AND_STRUCTURAL_QA', promoted_utc=now,
                    current_manuscript_sha256=NEW, visual_qa_sha256=sha(qa_path))
    write_new(outputs[0], promoted)
    versions = copy.deepcopy(prior_catalog['versions'])
    for item in versions:
        if item['revision'] == 14:
            item['retained_file'] = ref(BACKUP)
        item['current_canonical_has_these_bytes'] = sha(Path(item['canonical_path_recorded_at_the_time'])) == item['sha256']
    versions.append({'revision': 15, 'sha256': NEW, 'canonical_path_recorded_at_the_time': str(CURRENT),
                     'retained_file': ref(CANDIDATE), 'current_canonical_has_these_bytes': True,
                     'reviewed_version_proof': ref(outputs[0])})
    write_new(outputs[1], {'status': 'CURRENT_AND_INHERITED_MANUSCRIPT_VERSIONS_LOCATED', 'checked_utc': now,
                           'current_manuscript_at_this_audit': ref(CURRENT), 'versions': versions,
                           'previous_catalog': ref(prior_catalog_path),
                           'rule': 'Resolve each version-specific hash to its retained_file; prior canonical paths are historical.',
                           'scope': 'Original manuscript and revisions 7–15 in the inherited chain; file identity only.',
                           'historical_indices_modified': False})
    by_number = {}
    for item in old_citations:
        old = item['reference']
        new = old if old < 9 else old + 2
        by_number[new] = {'reference': new, 'manuscript_entry': refs[new - 1],
                          'reference_in_revision14': old, 'inherited_source_record': item}
    for item in evidence['sources']:
        new = item['new_reference']
        by_number[new] = {'reference': new, 'manuscript_entry': refs[new - 1],
                          'reference_in_revision14': None, 'new_source_record': item}
    assert sorted(by_number) == list(range(1, 14))
    citation_catalog = {'status': 'REFERENCE_NUMBERING_AND_READING_SCOPE_RECORDED', 'created_utc': now,
                        'manuscript_sha256': NEW, 'prior_archive': ref(old_citation_archive),
                        'new_evidence': ref(evidence_path), 'references': [by_number[n] for n in range(1, 14)],
                        'limits': 'New sources were checked in publisher search-indexed abstract/sections and metadata, not complete PDFs; no comprehensive novelty claim.'}
    write_new(outputs[2], citation_catalog)
    index = {'revision': 15, 'status': 'RELATED_WORK_UPDATED_WITH_PUBLISHER_PRIMARY_SOURCES',
             'created_utc': now, 'manuscript': ref(CURRENT), 'previous_indices': [ref(x) for x in prior_indices],
             'literature_evidence': ref(evidence_path), 'plain_language_explanation': ref(explanation),
             'citation_archive': ref(outputs[2]), 'visual_qa': ref(qa_path),
             'manuscript_version_locations': ref(outputs[1]), 'diagnostic_completion': ref(diagnostic),
             'previous_citation_numbers': 'Revision-14 references 1–8 remain unchanged; 9–11 move to 11–13.',
             'scope': 'Two related UAV architectures described from publisher excerpts; no external metric comparison, exhaustive novelty claim or full-PDF reading claim.',
             'model_instantiations': 0, 'model_forwards': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
             'submission_ready': False,
             'remaining': 'The interrupted BASE/FULL fair pair is incomplete; the later BASE diagnostic does not replace it. Further experiments require explicit authorization.'}
    write_new(outputs[3], index)
    index_text = ('主要结论证据索引 第十五轮\n当前稿件SHA256：' + NEW + '\n'
                  '相关工作增加DMF-Net及TF-DEIM-DFINE两项原始研究的结构性描述，依据出版方摘要/方法节选和元数据；'
                  '直接打开出版页分别返回429/403，未取得全文PDF，也未将外部指标与本稿混排。\n'
                  '共13条引用按首次出现编号；第十四轮旧[9]–[11]顺延为[11]–[13]。'
                  '新旧引用来源及阅读范围见本轮citation_archive_revision15.json；其余证据继承第十四轮索引。\n'
                  '10页逐页检查；表值、图及公式未改，本轮无新增实验。'
                  '公平BASE/FULL对照仍缺，尚不能投稿。\n机器可读索引：' + str(outputs[3]) + '\n')
    with outputs[4].open('x', encoding='utf-8', newline='\n') as out:
        out.write(index_text)

    header = ('【最新稿件入口：2026-11-05，第十五轮相关工作补核】\n'
              '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
              '相关工作补入DMF-Net和TF-DEIM-DFINE，13条文献按首次出现编号；'
              '只使用出版方可检索摘要/节选和元数据，未将外部结果混排。第十四轮稿件已备份，10页逐页检查。\n'
              '详情见修订记录_20261103/第十五轮相关工作补核_20261105/相关工作补核说明.md；'
              '最新证据索引为主要结论证据索引_第十五轮.txt及JSON。\n'
              '本轮无新增实验。授权的BASE诊断不构成公平主对照；重做方案仍待明确授权，尚不能投稿。'
              '以下较早运行与版本状态均为历史。\n\n')
    note = ('\n第十五轮：相关工作和引用补核（2026-11-05）\n'
            '按出版方摘要/章节节选补入DMF-Net多分支细节与双注意力融合、TF-DEIM-DFINE的Mamba骨干/注意力/特征金字塔；'
            '直接打开出版页分别返回429和403，未取得全文PDF，也未作外部准确率比较或穷尽式新颖性判断。'
            '新增[9]、[10]，原[9]–[11]顺延为[11]–[13]；其他引用内容与原证据继承。\n'
            '第十四轮备份SHA256：' + OLD + '；当前修订稿SHA256：' + NEW + '。'
            '13条文献按首次出现顺序核验，10页逐页检查，6表、7图、6式和5组批注锚点保留。'
            '本轮无模型、评价或训练；公平主对照仍缺，尚不能投稿。\n')
    citation_header = ('引用核查最新说明（2026-11-05，第十五轮）\n'
                       '当前稿共13条引用，按首次出现编号。新增[9] DMF-Net（DOI 10.3390/sym18050763）与'
                       '[10] TF-DEIM-DFINE（DOI 10.1016/j.imavis.2026.105897）；核查出版方可检索摘要/节选和元数据，'
                       '直接打开出版页分别返回429/403，未取得全文PDF。详情见第十五轮相关工作补核_20261105。\n'
                       '旧[1]–[8]编号不变，旧[9] SiLU、[10] COCO、[11] VisDrone工具包依次改为[11]、[12]、[13]。'
                       '逐条新旧编号与来源见citation_archive_revision15.json。以下为第八轮11条引用的历史编号，不能按其旧编号直接读取当前稿。\n\n')
    goal_text = old_text[3].replace('稿件现为第十四轮修订版', '稿件现为第十五轮修订版', 1)
    replace_text(text_paths[0], header + old_text[0])
    replace_text(text_paths[1], old_text[1].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW, 1))
    replace_text(text_paths[2], old_text[2] + note)
    replace_text(text_paths[3], goal_text + note)
    replace_text(text_paths[4], citation_header + old_text[4])
    print(json.dumps({'status': promoted['status'], 'current_sha256': sha(CURRENT),
                      'references': len(refs), 'original_sha256': sha(original), 'index': str(outputs[4])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
