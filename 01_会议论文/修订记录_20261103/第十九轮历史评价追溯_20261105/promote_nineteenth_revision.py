"""Promote the visually checked provenance update and update the evidence index."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import os
import shutil

HERE=Path(__file__).resolve().parent
REC=HERE.parent
PAPER=REC.parent
INDEX=REC/'主要结论证据索引'
REMOTE=PAPER/'远程公平对照_20261103'
CURRENT=PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP=HERE/'MPSMA-DEIM_conference_EN_revised_第十八轮备份.docx'
CANDIDATE=HERE/'MPSMA-DEIM_conference_EN_revised_第十九轮候选.docx'
OLD='888f5a17e991885b8581b16684e0a1d28ce06ab5ecd3f7176d2acd22e2551420'
NEW='6c603477dd95225525e4505fbf8c126f9b349e1ba5fa6cffaf87df4672238969'
ORIGINAL='ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def ref(path):
    return {'path':str(path),'sha256':digest(path)}

def write_new(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def replace_text(path,text):
    temp=path.with_name(path.name+'.rev19.tmp')
    assert not temp.exists()
    with temp.open('x',encoding='utf-8',newline='\n') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)

def main():
    original=PAPER/'稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup=PAPER/'稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert digest(CURRENT)==digest(BACKUP)==OLD and digest(CANDIDATE)==NEW
    assert digest(original)==digest(original_backup)==ORIGINAL
    pending=read(HERE/'nineteenth_revision_pending_qa.json')
    qa_path=HERE/'nineteenth_revision_visual_qa.json'
    qa=read(qa_path)
    proof_path=HERE/'independent_saved_array_verification.json'
    proof=read(proof_path)
    fields_path=HERE/'saved_evaluation_binding_fields.json'
    fields=read(fields_path)
    audit_path=HERE/'saved_array_log_matching.json'
    audit=read(audit_path)
    prediction_path=HERE/'s1_saved_prediction_rank_comparison.json'
    prediction=read(prediction_path)
    explanation=HERE/'历史评价数组与日志核查说明.md'
    assert pending['candidate_sha256']==qa['candidate_sha256']==NEW
    assert qa['status']=='VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages']==list(range(1,11))
    assert qa['pdf_sha256']==digest(HERE/'rendered_word/manuscript.pdf')
    assert proof['status']=='ALL_48_CORRESPONDING_ROWS_VERIFIED' and proof['arrays_checked']==48
    assert proof['source_audit_sha256']==fields['source_sha256']==digest(audit_path)
    assert proof['prediction_audit_sha256']==digest(prediction_path)
    assert proof['maximum_abs_difference']<1e-12 and proof['n2_independent_fmean_max_error']<1e-12
    assert [(x['run'],x['epoch']) for x in proof['selected_best_log_arrays']]==[('N2',300)]
    assert not fields['files_with_checkpoint_or_hash_named_fields'] and fields['files']==49
    for item in audit['saved_evaluations']:
        assert 'error' not in item and digest(Path(item['path']))==item['sha256']
    for item in audit['source_logs'].values():
        assert digest(Path(item['path']))==item['sha256']
    for item in audit['cpu'].values():
        assert digest(Path(item['path']))==item['sha256']
        assert digest(Path(item['metrics_path']))==item['metrics_sha256']
    for item in [prediction['historical'],prediction['cpu']]:
        assert digest(Path(item['path']))==item['sha256']
    with ZipFile(CURRENT) as a,ZipFile(CANDIDATE) as b:
        assert b.testzip() is None and a.namelist()==b.namelist()
        assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['word/document.xml']
    previous=INDEX/'主要结论证据索引_第十八轮.json'
    index_json=INDEX/'主要结论证据索引_第十九轮.json'
    index_txt=INDEX/'主要结论证据索引_第十九轮.txt'
    locations=HERE/'manuscript_version_locations.json'
    receipt=HERE/'nineteenth_revision_patch.json'
    assert not any(x.exists() for x in [index_json,index_txt,locations,receipt])
    paths=[PAPER/'当前修订入口.txt',REC/'修订说明与未解决问题.txt',REC/'目标验收与阻塞记录.txt',
           REMOTE/'结果并入与验收清单.txt']
    previous_text=[p.read_text(encoding='utf-8-sig') for p in paths]
    assert previous_text[3].count('当前稿件SHA256：'+OLD)==1
    for i,path in enumerate(paths):
        saved=HERE/f'第十九轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path,saved)
    temp=CURRENT.with_name(CURRENT.name+'.rev19.tmp')
    assert not temp.exists()
    shutil.copy2(CANDIDATE,temp)
    assert digest(temp)==NEW
    os.replace(temp,CURRENT)
    assert digest(CURRENT)==NEW and digest(original)==digest(original_backup)==ORIGINAL
    catalog={'revision':19,'previous_catalog':ref(REC/'第十七轮证据表述复核_20261105/manuscript_version_locations.json'),
             'versions':[{'revision':18,'sha256':OLD,'retained_file':ref(BACKUP)},
                         {'revision':19,'sha256':NEW,'retained_file':ref(CURRENT)}],
             'note':'Revision 18 did not change the revision-17 manuscript. Locate older bytes by their SHA and prior catalogs.'}
    write_new(locations,catalog)
    index={'revision':19,'status':'SAVED_EVALUATION_LOG_PROVENANCE_DOCUMENTED',
           'created_utc':datetime.now(timezone.utc).isoformat(),'manuscript':ref(CURRENT),
           'previous_index':ref(previous),'version_locations':ref(locations),'previous_manuscript':ref(BACKUP),
           'evaluation_array_audit':ref(audit_path),'independent_verification':ref(proof_path),
           'binding_field_audit':ref(fields_path),'saved_prediction_comparison':ref(prediction_path),
           'explanation':ref(explanation),'visual_qa':ref(qa_path),
           'finding':'Forty-two periodic and six latest saved evaluation arrays have 12-metric summaries matching their corresponding log rows within 1e-12; N2 at epoch 300 is the only selected best row covered.',
           'limits':'No explicit checkpoint hash field was found. The saved S1 standalone evaluation and predictions are not cryptographically bound to a checkpoint; exact historical code, data and weight identities remain unproven.',
           'models_run':0,'forwards':0,'evaluator_runs':0,'optimizer_updates':0,'submission_ready':False,
           'remaining':'A matched BASE/FULL fair comparison is incomplete and requires explicit authorization for new training.'}
    write_new(index_json,index)
    index_txt.write_text('主要结论证据索引 第十九轮\n当前稿件SHA256：'+NEW+'\n'
        '六组42份周期数组和6份latest数组的12项汇总均匹配相应日志行；在所选最佳行中仅N2零基300有对应周期数组。\n'
        '49份所查评价数据未发现显式checkpoint/hash命名字段；S1独立评价与CPU保存数组并非逐元素一致。不能由此认证历史权重。\n'
        '详情：'+str(explanation)+'\n机器可读索引：'+str(index_json)+'\n'
        '未生成新预测或运行评价；公平BASE/FULL仍缺，尚不能投稿。\n',encoding='utf-8')
    header=('【最新稿件入口：2026-11-05，第十九轮历史评价追溯】\n'
        '当前修订稿：'+str(CURRENT)+'\nSHA256：'+NEW+'\n'
        '第4.6节说明48份周期或末轮评价数组的12项汇总与对应日志一致，所选最佳行中仅N2零基300有周期数组；数组没有显式权重哈希。第十八轮稿件已备份，10页逐页检查。\n'
        '详情见修订记录_20261103/第十九轮历史评价追溯_20261105/历史评价数组与日志核查说明.md；最新证据索引为主要结论证据索引_第十九轮.txt及JSON。\n'
        '本轮只汇总既有数组和比较既有预测，无新模型、评价或训练。公平主对照仍待授权与完成，尚不能投稿。\n\n')
    replace_text(paths[0],header+previous_text[0])
    note=('\n第十九轮：历史评价数组与日志对应（2026-11-05）\n'
        '六组共42份定期和6份latest历史评价数组的12项汇总均唯一对应同编号/末轮日志行，最大绝对差约1.11e-16；预设匹配阈值1e-12。所选最佳行中仅N2零基300有周期评价数组；49份所查评价数据未见显式checkpoint/hash命名字段，不能认证历史权重。S1独立评价不与任何完整日志行严格匹配，也不与CPU保存数组逐元素一致；已存预测按顺序比较有类别位置差异，不推断同权重。\n'
        '第4.6节补充追溯范围和权重限制；第十八轮备份SHA256：'+OLD+'；当前稿件SHA256：'+NEW+'。6表、7图、6式和5组批注锚点保留，10页逐页检查。只汇总现成数组和比较既有预测，无新模型、评价或训练。公平BASE/FULL仍缺，尚不能投稿。\n')
    for i in [1,2]:
        replace_text(paths[i],previous_text[i]+note)
    lead=('最新历史评价核查（2026-11-05）：48份周期或末轮评价数组的12项汇总与日志对应，所选最佳行仅N2零基300有周期数组；未发现显式权重哈希字段。此证据不替代公平主对照或认证历史权重。详见修订记录_20261103/第十九轮历史评价追溯_20261105。\n')
    replace_text(paths[3],lead+previous_text[3].replace('当前稿件SHA256：'+OLD,'当前稿件SHA256：'+NEW))
    write_new(receipt,{'status':'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA','revision':19,
                       'manuscript':ref(CURRENT),'previous_manuscript':ref(BACKUP),
                       'evidence_index':ref(index_json),'visual_qa':ref(qa_path),'original_preserved':True})
    print(json.dumps({'status':'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA',
                      'manuscript_sha256':digest(CURRENT),'index_sha256':digest(index_json),'submission_ready':False},ensure_ascii=False))

if __name__=='__main__':
    main()
