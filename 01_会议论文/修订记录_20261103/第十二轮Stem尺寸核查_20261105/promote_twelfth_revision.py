"""Promote the reviewed stem-shape clarification and update traceable entry records."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import copy
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
REMOTE = PAPER / '远程公平对照_20261103'
INDEX = REC / '主要结论证据索引'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮候选_v3.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十一轮备份.docx'
OLD = '06c51c534c640fe35348bd88ed6c9a2a45c540b35521e378f008fe5a93126809'
NEW = 'c49695664206778e6d0537b050194ddf1ab9a651021e8f64e8d4dacab7e22da9'

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
    temp = path.with_name(path.name + '.twelfth-promotion.tmp')
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
    assert sha(original) == sha(original_backup) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
    patch = read(HERE / 'twelfth_revision_v3_pending_qa.json')
    qa = read(HERE / 'twelfth_revision_visual_qa.json')
    evidence_path = HERE / 'stem_shape_static_audit.json'
    evidence = read(evidence_path)
    assert patch['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages'] == list(range(1, 11))
    assert sha(evidence_path) == patch['evidence_sha256'] == qa['source_and_configuration_audit_sha256']
    assert evidence['status'] == 'STEM_INPUT_SHAPE_STATICALLY_CHECKED_NO_MODEL_EXECUTION'
    assert sha(Path(evidence['source_path'])) == evidence['source_sha256']
    for item in evidence['current_configs']:
        assert sha(REC / 'source_snapshot' / item['run'] / 'args.json') == item['args_sha256']
    prior_catalog_path = REC / '第十一轮评价上限核查/manuscript_version_locations.json'
    prior_catalog = read(prior_catalog_path)
    for item in prior_catalog['versions']:
        assert sha(Path(item['retained_file']['path'])) == item['sha256']
    prior_indices = [INDEX / ('主要结论证据索引_第十一轮' + suffix) for suffix in ['.txt', '.json']]
    assert read(prior_indices[1])['manuscript_sha256'] == OLD
    diagnostic = REMOTE / 'diagnostic_completion_receipt_20261105.json'
    assert read(diagnostic)['status'] == 'ONE_AUTHORIZED_DIAGNOSTIC_COMPLETED_AND_EVIDENCE_VERIFIED'

    outputs = [HERE / 'twelfth_revision_patch.json', HERE / 'manuscript_version_locations.json',
               INDEX / '主要结论证据索引_第十二轮.json', INDEX / '主要结论证据索引_第十二轮.txt']
    assert not any(path.exists() for path in outputs)
    text_paths = [PAPER / '当前修订入口.txt', REMOTE / '结果并入与验收清单.txt',
                  REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt']
    text_backups = [HERE / ('text_before_revision12_%d.txt' % i) for i in range(len(text_paths))]
    assert not any(path.exists() for path in text_backups)
    old_text = [path.read_text(encoding='utf-8-sig') for path in text_paths]
    assert old_text[1].count('当前稿件SHA256：' + OLD) == 1
    assert '稿件仍为第十一轮修订版' in old_text[3]
    for source, destination in zip(text_paths, text_backups):
        shutil.copy2(source, destination)

    temp = CURRENT.with_name(CURRENT.stem + '_revision12_promotion.tmp.docx')
    assert not temp.exists()
    with CANDIDATE.open('rb') as src, temp.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    assert sha(temp) == NEW and sha(CURRENT) == OLD
    os.replace(temp, CURRENT)
    assert sha(CURRENT) == NEW and sha(BACKUP) == OLD and sha(original) == sha(original_backup)
    now = datetime.now(timezone.utc).isoformat()
    patch = dict(patch, status='PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA',
                 promoted_utc=now, current_manuscript_sha256=NEW, visual_qa_sha256=sha(HERE / 'twelfth_revision_visual_qa.json'))
    write_new(outputs[0], patch)
    versions = copy.deepcopy(prior_catalog['versions'])
    for item in versions:
        if item['revision'] == 11:
            item['retained_file'] = ref(BACKUP)
        item['current_canonical_has_these_bytes'] = sha(Path(item['canonical_path_recorded_at_the_time'])) == item['sha256']
    versions.append({'revision': 12, 'sha256': NEW, 'canonical_path_recorded_at_the_time': str(CURRENT),
                     'retained_file': ref(CANDIDATE), 'current_canonical_has_these_bytes': True,
                     'reviewed_version_proof': ref(outputs[0])})
    catalog = {'status': 'CURRENT_AND_INHERITED_MANUSCRIPT_VERSIONS_LOCATED', 'checked_utc': now,
               'current_manuscript_at_this_audit': ref(CURRENT), 'versions': versions,
               'previous_catalog': ref(prior_catalog_path),
               'rule': 'Resolve each version-specific hash to its retained_file; prior canonical paths are historical.',
               'scope': 'Original manuscript and revisions 7–12 in the inherited chain; file identity only.',
               'historical_indices_modified': False}
    write_new(outputs[1], catalog)
    index = {'revision': 12, 'status': 'STEM_INPUT_SHAPE_CLARIFIED_WITH_PRIOR_EVIDENCE_INHERITED',
             'created_utc': now, 'manuscript': ref(CURRENT), 'previous_indices': [ref(x) for x in prior_indices],
             'static_shape_evidence': ref(evidence_path), 'visual_qa': ref(HERE / 'twelfth_revision_visual_qa.json'),
             'manuscript_version_locations': ref(outputs[1]), 'diagnostic_completion': ref(diagnostic),
             'historical_source_version_certified': False,
             'description': 'For the inspected EnhancedStem, odd input dimensions make detail/main concatenation incompatible; the reported 640×640 configuration aligns and Fig. 2 is explicitly scoped to it.',
             'model_executions': 0, 'evaluator_reruns': 0, 'training_updates': 0,
             'submission_ready': False,
             'remaining': 'The interrupted BASE/FULL fair pair is incomplete; the later BASE diagnostic does not replace it. Further experiments require explicit authorization.'}
    write_new(outputs[2], index)
    index_text = (
        '主要结论证据索引 第十二轮\n当前稿件SHA256：' + NEW + '\n'
        '第3.2节明确所查Stem在奇数高或宽时主分支与细节分支尺寸不匹配；图2图注限定H=W=640。'
        '来源为现存源码、三个保存配置及整数尺寸推导，未执行模型；不能据此认证历史运行的不可变源码。\n'
        '图表、公式、实验数值、引用和其他正文段落未改；10页逐页核查通过。'
        '证据、备份及版式见第十二轮Stem尺寸核查_20261105；旧稿按该目录manuscript_version_locations.json定位。\n'
        '其余结论继承第十一轮索引及其哈希链；其中早期“两组训练继续”属历史状态。'
        'BASE诊断已完成但不构成公平主对照；完整对照仍缺，尚不能投稿。\n'
        '机器可读索引：' + str(outputs[2]) + '\n')
    with outputs[3].open('x', encoding='utf-8', newline='\n') as out:
        out.write(index_text)

    header = ('【最新稿件入口：2026-11-05，第十二轮Stem尺寸核查】\n'
              '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
              '第3.2节说明现存Stem实现对奇数高或宽的拼接限制，图2图注明确H=W=640；'
              '第十一轮稿件已备份。6表、7图、6式和5组批注锚点保留，10页逐页核验通过。\n'
              '尺寸静态证据、候选及版式记录见修订记录_20261103/第十二轮Stem尺寸核查_20261105；'
              '结论证据索引为主要结论证据索引_第十二轮.txt及JSON。\n'
              '授权的BASE诊断已完成，未重现原骤降但原因仍未知；公平主对照未完成，尚不能投稿。'
              '重做两组训练须另获明确授权；以下较早运行与版本状态均为历史。\n\n')
    note = ('\n第十二轮：Stem输入尺寸范围核查（2026-11-05）\n'
            '静态核对现存EnhancedStem实现：主分支步长2卷积对奇数尺寸向上取整，细节分支2×2池化向下取整，'
            '因此任一空间维度为奇数时两分支不能拼接；现存N1/N4/S1配置均为640×640，'
            '故本文报告的尺寸可以对齐。图2将H=W限定为640；未改变代码、公式、结果或图像。\n'
            '第十一轮备份SHA256：' + OLD + '；当前修订稿SHA256：' + NEW + '。'
            '打包渲染器缺少soffice，已保留失败日志并用隐藏只读Word导出和Poppler逐页核查，'
            '公式页另以PDFium复核；完整10页、6表、7图、6式和5组批注锚点保留。\n'
            '本轮没有模型执行、训练或重评。现存源码不能证明历史训练时源码完全相同；'
            '公平BASE/FULL对照仍缺，尚不能投稿。\n')
    goal_text = old_text[3].replace('稿件仍为第十一轮修订版', '稿件现为第十二轮修订版', 1)
    replace_text(text_paths[0], header + old_text[0])
    replace_text(text_paths[1], old_text[1].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW, 1))
    replace_text(text_paths[2], old_text[2] + note)
    replace_text(text_paths[3], goal_text + note)
    print(json.dumps({'status': patch['status'], 'current_sha256': sha(CURRENT),
                      'original_sha256': sha(original), 'index': str(outputs[3])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
