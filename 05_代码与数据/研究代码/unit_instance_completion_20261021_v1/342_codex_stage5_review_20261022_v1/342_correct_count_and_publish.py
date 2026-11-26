"""Correct 6470+1096=7566 without overwriting any prior draft or receipt."""
import sys,importlib,json,difflib
sys.stdout.reconfigure(encoding='utf-8');C=importlib.import_module('342_stage5_context')
old_manifest=C.HERE/'342_活动缓存7576_逐条SHA总清单.json';d=C.read(old_manifest)
assert len(d['entries'])==6470+1096==7566
d['supersedes_count_misnamed_manifest']=C.evidence(old_manifest)
C.write('342_活动缓存7566_逐条SHA总清单.json',d)
timing=C.read(C.HERE/'342_缓存耗时与存储账本.json');assert timing['已执行']['active_cache_sizes_counted']==7576
timing['已执行']['active_cache_sizes_counted']=len(d['entries']);timing['supersedes_count_only']='342_缓存耗时与存储账本.json'
C.write('342_缓存耗时与存储账本_v2.json',timing)
old=C.WORK/'342_开训前审查包.md';newpath=C.HERE/'342_开训前审查包.md'
body=old.read_text(encoding='utf-8')
replacements={'7576':'7566','342_阶段5_result.json':'342_阶段5_result_v2.json',
              '342_阶段5_中文汇总.md':'342_阶段5_中文汇总_v2.md',
              '342_缓存耗时与存储账本.json':'342_缓存耗时与存储账本_v2.json'}
for a,b in replacements.items():body=body.replace(a,b)
body=body.replace('日期：2026-10-22。状态：','日期：2026-10-22。本目录同名文件为活动更正版；工作目录根部同名稿的缓存合计误写已作废，原稿保留。状态：',1)
C.text('342_开训前审查包.md',body)
C.text('342_审查包缓存合计更正.diff',''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),body.splitlines(True),fromfile='WORK/342_开训前审查包.md (inactive draft)',tofile='phase5/342_开训前审查包.md (active)')))
result=C.read(C.HERE/'342_阶段5_result.json');result['package_path']=str(newpath);result['package_sha256']=C.sha(newpath)
for row in result['checklist_342_section_VI']:
    if row['item']==2:row['entries']=7566;row['evidence']=str(C.HERE/'342_活动缓存7566_逐条SHA总清单.json')
assert result['已执行']['active_cache_SHA_checks']==7566
result['supersedes']='342_阶段5_result.json（合计数/活动包路径更正；实际SHA检查数本来就是7566）'
C.write('342_阶段5_result_v2.json',result)
summary=(C.HERE/'342_阶段5_中文汇总.md').read_text(encoding='utf-8')
for a,b in replacements.items():summary=summary.replace(a,b)
summary=summary.replace(old.as_posix(),newpath.as_posix())
summary+='\n活动审查包位于本阶段5目录；根部同名稿因合计误写保留作废。7566=6470+1096，条目内容与逐条SHA未改变。更正记录：[342_result_阶段5合计更正.json]('+ (C.HERE/'342_result_阶段5合计更正.json').as_posix()+')。\n'
C.text('342_阶段5_中文汇总_v2.md',summary)
ledger=C.read(C.HERE/'342_审查包证据文件SHA256.json')
ledger['files'].pop(str(old_manifest));ledger['files'].pop(str(old))
for p in (newpath,C.HERE/'342_活动缓存7566_逐条SHA总清单.json',C.HERE/'342_缓存耗时与存储账本_v2.json'):
    ledger['files'][str(p)]=C.sha(p)
ledger['已执行']['evidence_files']=len(ledger['files'])
ledger['supersedes']='342_审查包证据文件SHA256.json'
C.write('342_审查包证据文件SHA256_v2.json',ledger)
C.write('342_result_阶段5合计更正.json',{'status':'已更正；原稿保留作废','correct_total':7566,'development_entries':6470,'val_entries':1096,
    'incorrect_draft_total':7576,'data_entries_unchanged':True,'data_file_SHA_values_unchanged':True,
    'inactive_draft':C.evidence(old),'active_package':C.evidence(newpath),
    'inactive_receipts':['342_阶段5_result.json','342_活动缓存7576_逐条SHA总清单.json','342_缓存耗时与存储账本.json','342_阶段5_中文汇总.md'],
    'active_receipts':['342_阶段5_result_v2.json','342_活动缓存7566_逐条SHA总清单.json','342_缓存耗时与存储账本_v2.json','342_阶段5_中文汇总_v2.md'],
    '已执行':{'arithmetic_checks':1,'new_package_writes':1,'data_file_mutations':0,'model_forwards':0,'optimizer_steps':0},
    '未执行':['真实训练','GPU'],'未核':['最终交付核查'],'source_sha256':C.sources()})
# Publish a corrected validator, also exclusively new.
original=(C.HERE/'342_validate_and_seal.py').read_text(encoding='utf-8')
fixed=original.replace("p=C.WORK/'342_开训前审查包.md'", "p=C.HERE/'342_开训前审查包.md'")
for a,b in replacements.items():fixed=fixed.replace(a,b)
fixed=fixed.replace('342_审查包证据文件SHA256.json','342_审查包证据文件SHA256_v2.json')
fixed=fixed.replace('342_result_阶段5交付核查.json','342_result_阶段5交付核查_v2.json')
fixed=fixed.replace('development/val 7566条SHA','development/val 7566条SHA')
fixed=fixed.replace("'已执行':{'local_file_link_checks'", "'count_correction_receipt':str(C.HERE/'342_result_阶段5合计更正.json'),'已执行':{'local_file_link_checks'")
C.text('342_validate_and_seal_v2.py',fixed)
print('corrected total 7566; active package:',newpath,flush=True)
