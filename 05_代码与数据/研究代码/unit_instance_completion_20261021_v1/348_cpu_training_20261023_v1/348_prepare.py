"""Exclusive 348 plan creation; no model forward or optimizer update."""
import ast
import importlib
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
C = importlib.import_module('348_common')

paths = sorted(C.HERE.glob('*.py'))
for path in paths:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

auth = {'recorded_utc': C.now(), 'source': '本聊天最新用户消息',
        'user_text': '347 我核过了，没有问题，可以开训。但只能在本机 CPU 上训练，现在还不能上远程。',
        'authorized': ['train347.py；本机CPU串行；N/S/G/R各seed1和seed2；每个max_steps=1218',
                       '每个run新目录；完成核对full_budget_completed=true后封存',
                       '中断从最近完整断点恢复，不重开；不调参、不加轮、不选best',
                       '八run全部完成后停下，等主控核对'],
        'not_authorized': ['远程/SSH/GPU', 'val548评估、G Δ、bootstrap、12项判定（本阶段不执行）'],
        'historical_note': '既有封存中的未授权措辞是先前阶段状态，原文不改；本记录绑定当前新增授权。'}
C.write(C.HERE / '348_用户授权与执行范围.json', auth)

old_seals = [C.WORK / '345_codex_contracts_20261022_v1/345_交付封存SHA256_v3.json',
             C.WORK / '346_codex_cpu_boundary_20261022_v1/346_交付封存SHA256.json',
             C.S347 / '347_交付封存SHA256.json']
for seal in old_seals:
    C.check(seal.is_file(), 'missing old seal: ' + str(seal))
cache_manifest = C.S5 / '342_活动缓存7566_逐条SHA总清单.json'
C.verify_record(cache_manifest, 'd615ae4472ddbf877b87821021baeb8d28842d7c51722a29fbc9d4db3924369e')
order = [f'{arm}_seed{seed}' for seed in (1, 2) for arm in 'NSGR']
configs = C.read(C.S5 / '342_八run配置清单.json')['runs']
C.check({c['run_id'] for c in configs} == set(order), 'exactly eight frozen configs')
references = {}
for identity in order:
    refs = [('345 formal first2', C.S345 / '345_B_CPU_acceptance_runs' /
             f'345_{identity}_formal_order_first2' / '345_steps.json')]
    if identity == 'G_seed1':
        refs.append(('347 G_seed1 accepted first10', C.S347 / '347_subject_G_seed1_10/347_steps.json'))
    references[identity] = [{'path': str(p), 'sha256': C.sha(p), 'label': label} for label, p in refs]
records = list(old_seals) + [cache_manifest, C.HERE / '348_用户授权与执行范围.json',
                           C.S5 / '342_八run配置清单.json']
for refs in references.values():
    records.extend(C.Path(r['path']) for r in refs)
document = {'authorized_utc': auth['recorded_utc'], 'run_order': order,
            'per_run_steps': 1218, 'device': 'cpu', 'parallel_training_limit': 1,
            'entry': str(C.S347 / 'train347.py'),
            'sources': {str(p): C.sha(p) for p in paths},
            'frozen_records': [{'path': str(p), 'sha256': C.sha(p)} for p in sorted(set(records))],
            'old_seals': [{'path': str(p), 'sha256': C.sha(p)} for p in old_seals],
            'active_cache_manifest': {'path': str(cache_manifest), 'sha256': C.sha(cache_manifest)},
            'references': references,
            'observer': {'training_code_modified': False, 'CPU_threads': 2,
                         'progress': 'Read completed-step locals through a Python trace callback; no tensor/RNG mutation.',
                         'initial_checkpoint': 'Call the unchanged 347 save_state closure at the first loop boundary; check exact RNG and model equality before/after.',
                         'pause': 'Only explicit pause marker; checkpoint at a complete-step boundary.',
                         'resume': 'Same logical run, new attempt child directory, nearest complete SHA-verified 347 receipt; no reset.'},
            'stop_gate': 'After 8 full runs plus sealing; all evaluation waits for user review.'}
C.write(C.PLAN, document)
C.write(C.HERE / '348_执行绑定SHA.json', {'path': str(C.PLAN), 'sha256': C.sha(C.PLAN)})
C.plan()
C.emit({'status': '计划独占新建并绑定', 'runs': order, 'syntax_checked_files': len(paths),
        'model_forwards': 0, 'optimizer_steps': 0})
