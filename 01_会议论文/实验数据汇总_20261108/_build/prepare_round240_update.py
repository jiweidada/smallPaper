"""Prepare a40-row, immutable-evidence-only data advance from200 to240."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'update_round200.py').read_text(encoding='utf-8')
replacements = [
    ('independent round200', 'independent round240'),
    ('ROUNDS = 200', 'ROUNDS = 240'),
    ('lineage_second_recovery_through_round200.json', 'lineage_second_recovery_through_round240.json'),
    ('resume_monitor_20261109_002/milestone_round200_live.json', 'resume_monitor_20261110_003/milestone_round240_live.json'),
    ("SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round165_verified_20261109'", "SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round200_verified_20261110'"),
    ("FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round200_verified_20261110'", "FINAL_SNAPSHOT = OUT / 'verified_snapshots/NO_MSCA_round240_verified_20261110'"),
    ("['completed_rounds'] == 165", "['completed_rounds'] == 200"),
    ("len(previous_manifest['files']) == 20", "len(previous_manifest['files']) == 25"),
    ('c451348bb627f7be74fd9b56deaa36b7299742537e63aaed01505bed9dfaeb11', 'b210f4fcaf5acb09fffbc8af8cb6f484c73b2a854136352227a087c04e132545'),
    ("recorded_attempts'] == 323400", "recorded_attempts'] == 388080"),
    ("['new_completed_rounds'] == 36", "['new_completed_rounds'] == 76"),
    ("['new_optimizer_update_attempts'] == 58212", "['new_optimizer_update_attempts'] == 122892"),
    ("worker['actual_loop_epoch'] == worker['completed_epochs'] == 200", "worker['actual_loop_epoch'] == worker['completed_epochs'] == 240"),
    ("worker['optimizer_step_attempts'] == 323501 and worker['batch'] == 101", "worker['optimizer_step_attempts'] == 388081 and worker['batch'] == 1"),
    ("worker['local_backup_verified']['epoch'] == 199", "worker['local_backup_verified']['epoch'] == 239"),
    ("epochs.accepted_stage('NO_MSCA', 199)", "epochs.accepted_stage('NO_MSCA', 239)"),
    ("rows[-1]['optimizer_step_attempts'] == 323400", "rows[-1]['optimizer_step_attempts'] == 388080"),
    ("best['epoch_index'] == 198", "best['epoch_index'] == 238"),
    ('rows[198]', 'rows[238]'),
    ("rows[index]['metrics']['AP']) == 198", "rows[index]['metrics']['AP']) == 238"),
    ('0.19650041568758386', '0.21032901073722304'),
    ('0.13120383278867362', '0.1393851562602276'),
    ('0.19665533181774233', '0.21033699779236342'),
    ("/epoch0164')", "/epoch0199')"),
    ("[('NO_MSCA_SECOND_RECOVERY_R200', CERTIFICATE), ('NO_MSCA_SECOND_RECOVERY_LIVE_R201', LIVE_PATH), ('NO_MSCA_OBSERVER003_IDENTITY', IDENTITY)]",
     "[('NO_MSCA_SECOND_RECOVERY_R240', CERTIFICATE), ('NO_MSCA_SECOND_RECOVERY_LIVE_R241', LIVE_PATH)]"),
    ('epoch_index_0based=199, remaining_rounds=120', 'epoch_index_0based=239, remaining_rounds=80'),
    ('optimizer_step_attempts=323400', 'optimizer_step_attempts=388080'),
    ('NO_MSCA_SECOND_RECOVERY_R200', 'NO_MSCA_SECOND_RECOVERY_R240'),
    ('NO_MSCA_SECOND_RECOVERY_LIVE_R201', 'NO_MSCA_SECOND_RECOVERY_LIVE_R241'),
    ('independently_verified_round200', 'independently_verified_round240'),
    ("'previous_rounds': 165, 'new_verified_rounds': 35, 'new_verified_updates': 56595", "'previous_rounds': 200, 'new_verified_rounds': 40, 'new_verified_updates': 64680"),
    ('PROGRESS_ONLY_AP_BEST_WITHIN_FIRST200_NOT_A_FINAL_ENDPOINT', 'PROGRESS_ONLY_AP_BEST_WITHIN_FIRST240_NOT_A_FINAL_ENDPOINT'),
    ("'round_1based': 199, 'epoch_index_0based': 198, 'cutoff_rounds': 200", "'round_1based': 239, 'epoch_index_0based': 238, 'cutoff_rounds': 240"),
    ('new_verified_rounds=36, new_verified_optimizer_updates=58212', 'new_verified_rounds=76, new_verified_optimizer_updates=122892'),
    ('cumulative_recovered_rounds_since29=171, cumulative_recovered_optimizer_updates_since29=276507', 'cumulative_recovered_rounds_since29=211, cumulative_recovered_optimizer_updates_since29=341187'),
    ("'actual_epoch_index_0based': 200, 'round_1based': 201", "'actual_epoch_index_0based': 240, 'round_1based': 241"),
    ("'native_accepted_rounds_observed': 200", "'native_accepted_rounds_observed': 240"),
    ('Immutable002 milestone200 process observation on30715. Round201 was training at this instant; metrics stop at independently audited200. Mutable003 latest/status hashes are not permanent evidence.',
     'Immutable003 milestone240 process observation on30715. Round241 was training at this instant; metrics stop at independently audited240. No004 or mutable latest/status hash is permanent evidence.'),
    ("'last_native_accepted_round': 200, 'independent_metric_snapshot_rounds': 200", "'last_native_accepted_round': 240, 'independent_metric_snapshot_rounds': 240"),
    ("'next_observer_identity_source_id': 'NO_MSCA_OBSERVER003_IDENTITY',\n        'next_observer_pid': 40816, 'next_observer_target_round': 240,", "'milestone_observer_identity_source_id': 'NO_MSCA_OBSERVER003_IDENTITY',\n        'milestone_observer_pid': 40816, 'milestone_observer_target_round': 240,\n        'milestone_observer_window_status': 'COMPLETED_AT240',"),
    ('仅独立核验200轮', '仅独立核验240轮'),
    ('本次指标仅截至独立200轮', '本次指标仅截至独立240轮'),
    ('VERIFIED_AT_ROUND200_ACCEPTANCE', 'VERIFIED_AT_ROUND240_ACCEPTANCE'),
    ('latest_observed_running_round=201', 'latest_observed_running_round=241'),
    ('metrics cut off at independent200', 'metrics cut off at independent240'),
    ('len(original_rows) == 1125', 'len(original_rows) == 1160'),
    ("['round_1based']) == 165", "['round_1based']) == 200"),
    ('zip(rows[165:200], refs[165:200])', 'zip(rows[200:240], refs[200:240])'),
    ('len(additional) == 35', 'len(additional) == 40'),
    ("[165, 200], 'rounds_included': [166, 200]", "[200, 240], 'rounds_included': [201, 240]"),
    ('Authorized30715 continuation is RUNNING in the immutable milestone200 observation at round201. Only accepted and independently audited rounds1-200 are exported; round201 and later are excluded. The stage-local best199 is progress only. Old discarded partial165 work remains historical.',
     'Authorized30715 continuation is RUNNING in the immutable003 milestone240 observation at round241. Only accepted and independently audited rounds1-240 are exported; round241 and later are excluded. The stage-local best239 is progress only. Old discarded partial165 work remains historical.'),
    ('Second recovery200 independent certificate retained; observed201 excluded; original1125 CSV bytes retained; latest200 and best199 distinct', 'Second recovery240 independent certificate retained; observed241 excluded; original1160 CSV bytes retained; latest240 and best239 distinct'),
    ("report['rows'] = 1160", "report['rows'] = 1200"),
    ('_build/update_round200_plan.json', '_build/update_round240_plan.json'),
    ("'total_updates': 323400, 'previous_snapshot_rounds': 165", "'total_updates': 388080, 'previous_snapshot_rounds': 200"),
    ("'update_tag': 'round200'", "'update_tag': 'round240'"),
    ('ROUND200_DATA_UPDATED', 'ROUND240_DATA_UPDATED'),
    ("'per_epoch_rows': 1160", "'per_epoch_rows': 1200"),
]
for old, new in replacements:
    assert text.count(old) >= 1, old
    text = text.replace(old, new)
text = text.replace("    assert live_wrapper['exit_code'] == 0\n", "    assert live_wrapper['exit_code'] == 0\n    assert sha(LIVE_PATH) == '809ab33296ee7e61be57f3f4a6967de1c7a1eeedd89ce1550ac030699a5baa1f'\n")
start = text.index("    text = (OUT / '汇总说明.md').read_text")
end = text.index("    replace_json(OUT / '_build/update_round240_plan.json'", start)
text = text[:start] + """    text = (OUT / '汇总说明.md').read_text(encoding='utf-8').replace(
        'NO_MSCA 的来源快照已独立核验至第 200 轮，进度 AP 为 19.6500%',
        'NO_MSCA 的来源快照已独立核验至第 240 轮，进度 AP 为 21.0329%')
    text += '\\n2026-11-10 240轮更新。NO_MSCA 独立证据截止第240轮，累计388,080次更新；相对200轮快照只追加201–240轮的40行及64,680次更新。第240轮AP为21.0329%、APs为13.9385%；截至该证据点的局部AP最佳在第239轮，AP为21.0337%。本页12指标和最新张量/预测/数组均对应第240轮；局部最佳权重与12指标另标进度，均不是最终320轮主/次端点。\\n'
    text += '\\n003已结束窗口的固定 milestone_round240_live.json 记录同一30715 worker413/supervisor412进入第241轮（服务器UTC '+live['observed_server_utc']+'）；该时点观察与独立指标截止240分开。新004与任何mutable latest/status均不作为本次永久来源。164轮恢复段累计76轮/122,892次更新；29/164两次恢复、旧partial165至少1,401次弃用更新及更多未知物理工作量、单seed/验证集选点/非官方VisDrone/非无中断位级等同限制全部保留。NO_MSCA的24个320轮终点单元格继续为空。\\n'
    replace_bytes(OUT / '汇总说明.md', text.encode('utf-8'))
    text = (OUT / '逐轮明细说明.md').read_text(encoding='utf-8').replace('共 1160 行', '共 1200 行').replace('前 200 行', '前 240 行')
    text = text.replace('第166–200行来自epoch0199累计日志，并与独立200轮证书核对。第201轮及之后指标不纳入本快照。',
                        '第166–200行来自epoch0199累计日志，第201–240行来自epoch0239累计日志，并与独立240轮证书核对。第241轮及之后指标不纳入本快照。')
    text += '\\n2026-11-10 240轮更新精确保留原1160行字节前缀及全部原字段/格式，只从接受的epoch0239追加201–240轮40行。BASE/FULL/NO_MPS各320行，NO_MSCA240行，总计1200行；14,400项原始0–1指标均逐行回溯。第239轮局部最佳与第240轮实际进度分开，241及之后排除。旧行状态仍代表各自历史快照。\\n'
    replace_bytes(OUT / '逐轮明细说明.md', text.encode('utf-8'))
""" + text[end:]
start = text.index("        'progress_notes': [")
end = text.index("        'new_source_sha256':", start)
text = text[:start] + """        'progress_notes': [
            '30715续训运行；独立截止240轮，固定003现场见第241轮。',
            '本页为第240轮：AP21.0329%、APs13.9385%，均为进度。',
            '截至240轮局部最佳在第239轮，AP21.0337%；非320轮端点。',
            '164轮恢复段76轮/122892次更新；累计240轮/388080次。',
            '29/164两次恢复；旧partial165至少1401次，更多物理工作量未知。',
            '完整状态/RNG恢复已核验；单seed/验证集选点/非官方VisDrone限制保留。',
        ],
""" + text[end:]
target = root / 'update_round240.py'
compile(text, str(target), 'exec')
with target.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
workbook = (root / 'update_progress_workbook.mjs').read_text(encoding='utf-8')
old = '`${localDate}，第${rounds}轮已独立核验。320轮终点保持空缺。`'
assert workbook.count(old) == 1
workbook = workbook.replace(old, '`${localDate}，30715续训运行；独立截止${rounds}轮，固定现场第241轮。320轮终点空缺。`')
with (root / 'update_progress_workbook_round240.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(workbook)
render = (root / 'verify_round160_full_renders.mjs').read_text(encoding='utf-8').replace('round160', 'round240')
with (root / 'verify_round240_full_renders.mjs').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(render)
print('ROUND240_DATA_AND_WORKBOOK_SCRIPTS_PREPARED')
