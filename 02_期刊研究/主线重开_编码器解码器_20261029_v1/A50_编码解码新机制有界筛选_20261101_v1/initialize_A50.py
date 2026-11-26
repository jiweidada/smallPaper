from pathlib import Path
import hashlib, json, shutil, subprocess
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
LIB = BASE.parent / '文献_相关原始论文_20261004'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
receipt = BASE / 'A49_下一主线有界筛选_20261101_v1' / '18_A49终点与历史保留校验回执.json'
assert sha(receipt) == 'ef353c583d8ebc36d2b22707c21d6a857f43a6a9de8e950f38b344d2d2e9ba79'
old = json.loads(receipt.read_text(encoding='utf-8'))
preserved = {}
archive = ROOT / 'before_A50'
archive.mkdir(exist_ok=True)
for name in ['01_候选判别表.json', '15_当前状态与结果校验.json']:
    p = BASE / name
    expected = old['current_files_sha256'][str(p)]
    assert sha(p) == expected
    dest = archive / name
    assert not dest.exists()
    shutil.copy2(p, dest)
    assert sha(dest) == expected
    preserved[str(p)] = {'preserved_path':str(dest), 'sha256':expected}
save(ROOT / '00_A50开始与历史保留.json', {
    'status':'BOUNDED_METHOD_SCREEN_NO_NEW_GPU_CALLS', 'goal_status':'active',
    'mainline_confirmed':False, 'previous_terminal_receipt':str(receipt),
    'previous_terminal_receipt_sha256':sha(receipt), 'prior_canonical_inputs_resolved':preserved,
    'cumulative_model_forwards_before_A50':144941,
    'cumulative_optimizer_updates_including_capacity_before_A50':117059,
    'cumulative_formal_epochs_before_A50':144,
    'A49_scope':'原版/box-ignore/support-ignore均12轮结束，候选固定终点未过门槛；仅淘汰该具体配方',
    'candidates':[{'id':'A50_ENCODER_MERGING','state':'HOLD_PRIOR_OVERLAP_UNTRAINED',
       'reason':'SaMer已用训练实例框提供token合并先验，泛用实例纯度合并不能作为新算法增量'},
      {'id':'A50_DECODER_FULL_EDGE_SHAPE_MATCHING','state':'SOURCE_AND_CLOSEST_METHOD_CHECK_UNTRAINED',
       'question':'不增加推理分支时，原生FDR完整边界分布用于query-GT匹配，是否优于点框和同矩高斯近似？',
       'not_claimed':'概率匹配、Wasserstein、期望绝对误差或分布回归本身均不是新原理；FDR bins未被证明是校准后验'}],
    'new_training_queued':False})
texts = ROOT / 'selected_text'
texts.mkdir(exist_ok=True)
papers = [
 ('23_Uncertainty_aware_DETR__arXiv-2507.14855v1.pdf', 'https://arxiv.org/pdf/2507.14855v1'),
 ('105_D_FINE_ICLR2025_Publisher.pdf','https://openreview.net/pdf?id=MFZjrTFE7h'),
 ('22_NWD_tiny-object__arXiv-2110.13389.pdf','https://arxiv.org/abs/2110.13389'),
 ('21_SA-Matching_DETR_CVPRFindings2026.pdf', None)]
entries=[]
for name,url in papers:
    p=LIB/name
    assert p.exists()
    t=texts/(p.stem+'.txt')
    assert not t.exists()
    subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe','-layout',str(p),str(t)],check=True)
    entries.append({'pdf':str(p),'sha256':sha(p),'source_url':url,'text':str(t),
       'acquisition':'reused_existing_library_PDF_no_redownload',
       'version_status':'exact version needs embedded-header/primary verification; filename alone not sufficient',
       'reading_scope':'extracted only; selected reading follows separately'})
save(ROOT/'02_复用原文登记_v1.json', entries)
bridge = BASE/'A49_下一主线有界筛选_20261101_v1'/'remote_A49_bounded_transfers.py'
dest=ROOT/'remote_A50.py'
assert not dest.exists()
dest.write_text(bridge.read_text(encoding='utf-8').replace('/root/uav_a49_20261101_v1','/root/uav_a50_20261101_v1'),encoding='utf-8')
print(json.dumps({'phase':'A50','historical_inputs_preserved':2,'existing_pdfs_reused':len(entries),
 'new_model_forwards':0,'new_training':0,'bridge_started':False},ensure_ascii=False))
