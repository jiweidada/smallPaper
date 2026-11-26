"""Index the external data checks and refine the pending fair-comparison data decision."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
INDEX = REC/'主要结论证据索引'
REMOTE = PAPER/'远程公平对照_20261103'
DOC = PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
DOC_SHA = '888f5a17e991885b8581b16684e0a1d28ce06ab5ecd3f7176d2acd22e2551420'
ORIGINAL_SHA = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'

def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def ref(path):
    return {'path':str(path),'sha256':digest(path)}

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    assert digest(DOC)==DOC_SHA
    original = PAPER/'稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup = PAPER/'稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert digest(original)==digest(original_backup)==ORIGINAL_SHA
    val = read(HERE/'public_val_archive_audit.json')
    val_check = read(HERE/'independent_val_archive_verification.json')
    train = read(HERE/'train_annotation_range_audit.json')
    train_check = read(HERE/'independent_train_annotation_verification.json')
    http = read(HERE/'official_val_http_diagnostic.json')
    release = read(HERE/'ultralytics_release_metadata.json')
    assert val['status']=='THIRD_PARTY_ARCHIVE_MEMBERS_COMPARED'
    assert val['images_compared']==val['original_label_files_compared']==548
    assert not val['image_mismatches'] and not val['label_mismatches']
    assert digest(Path(val['archive_path']))==val['archive_sha256']==val['declared_archive_sha256']
    assert val_check['status']=='ALL_CURRENT_VALIDATION_FILES_MATCH_PUBLIC_ARCHIVE_BYTES'
    assert val_check['source_audit_sha256']==digest(HERE/'public_val_archive_audit.json')
    assert train['status']=='FOUR_TRAIN_ANNOTATIONS_COMPARED_BY_HTTP_RANGE' and train['all_four_match_local']
    assert not train['full_archive_downloaded'] and len(train['annotations'])==4
    assert train_check['status']=='TRAIN_LABEL_RANGE_SNAPSHOTS_MATCH_LOCAL'
    assert train_check['range_audit_sha256']==digest(HERE/'train_annotation_range_audit.json')
    assert http['status']=='HTML_AFTER_CONFIRMATION' and http['confirmation']['text_indicators']['quota']
    assert release['status']=='RELEASE_ASSET_METADATA_RETRIEVED' and release['size_matches_mirror']
    assert release['asset_digest'] is None and release['sha256_matches_mirror'] is None
    previous = INDEX/'主要结论证据索引_第十七轮官方源补充.json'
    output = INDEX/'主要结论证据索引_第十八轮.json'
    output_text = INDEX/'主要结论证据索引_第十八轮.txt'
    assert not output.exists() and not output_text.exists()
    text_paths = [PAPER/'当前修订入口.txt', REC/'修订说明与未解决问题.txt',
                  REC/'目标验收与阻塞记录.txt', REMOTE/'公平主对照重做提案_20261105.md',
                  REMOTE/'结果并入与验收清单.txt']
    before = [p.read_text(encoding='utf-8-sig') for p in text_paths]
    anchor = '## 数据来源待核查（2026-11-05）'
    assert before[3].count(anchor)==1
    for i,path in enumerate(text_paths):
        saved = HERE/f'第十八轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path,saved)
    evidence_paths = [HERE/'验证集公开归档比对说明.md', HERE/'public_val_archive_audit.json',
                      HERE/'independent_val_archive_verification.json', HERE/'train_archive_range_probe.json',
                      HERE/'train_annotation_range_audit.json', HERE/'independent_train_annotation_verification.json',
                      HERE/'official_val_http_diagnostic.json', HERE/'ultralytics_release_metadata.json',
                      Path(val['archive_path'])]
    index = {'revision':18, 'status':'PUBLIC_ARCHIVE_DATA_PROVENANCE_AUDITED',
             'created_utc':datetime.now(timezone.utc).isoformat(), 'manuscript':ref(DOC),
             'manuscript_changed':False, 'previous_index':ref(previous), 'evidence':[ref(p) for p in evidence_paths],
             'finding':'A pinned third-party validation ZIP matches all 548 current JPEGs and 548 original annotation TXT files byte for byte; four train TXT labels match bounded range reads of that mirror’s train ZIP.',
             'limits':'The official download returned a quota notice. The third-party source is not authenticated as official, and the full training ZIP hash was not recomputed. AP impact and historical data identity remain unknown.',
             'models_run':0,'forwards':0,'evaluator_runs':0,'optimizer_updates':0,
             'submission_ready':False,'remaining':'Complete the matched BASE/FULL comparison after explicit authorization and address dataset provenance limitations.'}
    with output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(index,f,ensure_ascii=False,indent=2)
        f.write('\n')
    output_text.write_text('主要结论证据索引 第十八轮\n稿件未改，SHA256：'+DOC_SHA+'\n'
        '固定第三方验证归档的548张JPEG和548份原始TXT与现存文件全部逐字节一致；训练归档以范围请求核对4份重复图对应标注，也全部一致。\n'
        '官方Google Drive返回配额提示，未取得官方归档；第三方同名归档与Ultralytics发布文件仅确认大小相同，未证明哈希相同。\n'
        '详情：'+str(HERE/'验证集公开归档比对说明.md')+'\n机器可读索引：'+str(output)+'\n'
        '公平BASE/FULL仍不完整；AP影响及历史数据身份未知，尚不能投稿。\n',encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，第十八轮数据来源核查】\n'
        '当前修订稿未改：'+str(DOC)+'\nSHA256：'+DOC_SHA+'\n'
        '固定版本的第三方验证归档中548张图及548份原始标注与现存文件逐字节一致；训练归档四份相关标注经范围读取也一致。官方链接返回访问配额超限提示，尚未取得官方归档。\n'
        '详情见修订记录_20261103/第十八轮原始数据追溯_20261105/验证集公开归档比对说明.md；最新证据索引为主要结论证据索引_第十八轮.txt及JSON。\n'
        '未运行新训练、推理或评价。公平BASE/FULL主对照仍待授权与完成，尚不能投稿。\n\n')
    text_paths[0].write_text(header+before[0],encoding='utf-8')
    note = ('\n第十八轮：公开数据归档与标注补核（2026-11-05）\n'
        '固定第三方验证ZIP及LFS哈希后，全量548图/548份原始TXT与现存文件及既存清单一致，并以独立逐字节程序复核；第三方训练ZIP的目录与四份相关TXT经范围读取也一致，未下载完整训练归档。官方Drive确认后返回访问配额超限提示，官方归档未取得；Ultralytics同名发布资产仅大小一致，未提供哈希。\n'
        '不推断官方数据或历史运行身份，不估计AP影响；建议获授权时保持当前冻结数据和本地评价协议并披露限制，数据变更需另行固定方案并授权。稿件未修改，SHA256：'+DOC_SHA+'；无模型、评价或训练。公平主对照仍缺，尚不能投稿。\n')
    for i in [1,2]:
        text_paths[i].write_text(before[i]+note,encoding='utf-8')
    proposal = ('\n\n第三方公开验证归档中的548图及548份原始标注与当前文件全部逐字节一致，训练归档中四份相关原始标注也与当前文件一致；官方归档因访问配额未能获取。建议如获授权重做公平对照，在重新核验清单后沿用现存6471/548条记录与本地COCO-style协议，并披露重复与标注差异及官方来源未认证；没有依据悄悄删除或修正记录。若改用其他数据，须重新固定清单、预算和评价端点并获得相应实验授权。详情见修订记录_20261103/第十八轮原始数据追溯_20261105/验证集公开归档比对说明.md。')
    text_paths[3].write_text(before[3].replace(anchor,anchor+proposal,1),encoding='utf-8')
    lead = ('最新数据核查（2026-11-05）：第三方公开验证归档548图/548份原始标注与当前文件全量一致；训练归档四份相关标签也一致。官方归档受访问配额限制未取得，重复记录对AP影响未知；获授权重做时建议冻结现存数据并披露限制。证据见修订记录_20261103/第十八轮原始数据追溯_20261105。\n')
    text_paths[4].write_text(lead+before[4],encoding='utf-8')
    assert digest(DOC)==DOC_SHA and digest(original)==digest(original_backup)==ORIGINAL_SHA
    print(json.dumps({'status':index['status'],'index_sha256':digest(output),'manuscript_sha256':DOC_SHA,
                      'submission_ready':False},ensure_ascii=False))

if __name__=='__main__':
    main()
