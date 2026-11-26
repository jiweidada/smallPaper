"""Python 3.10 f-string compatibility; no change to evidence or statuses."""
import importlib,difflib,ast
C=importlib.import_module('342_stage5_context')
old=(C.HERE/'342_build_review_package.py').read_text(encoding='utf-8');lines=[]
for line in old.splitlines(True):
    if 'lines.append(f\'| {title.replace(' in line:
        lines.append("        label=title.replace('|',chr(92)+'|')\n")
        lines.append('        lines.append(f\'| {label} | {dev["stats"]["1"][key]} | {dev["stats"]["2"][key]} |\')\n')
    elif 'lines.append(f\'| {r["run_id"]}' in line:
        lines.append("        config_path=C.HERE/('342_预定配置_{}_seed{}.json'.format(r['arm'],r['global_seed']))\n")
        lines.append('        lines.append(f\'| {r["run_id"]} | {r["head_seed"]} | 6 | 16/False | 1218 | 19410 | {x} | {link(config_path)} |\')\n')
    else:lines.append(line)
new=''.join(lines);ast.parse(new);assert new!=old
C.text('342_build_review_package_v2.py',new)
C.text('342_审查包生成脚本兼容修正.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='342_build_review_package.py',tofile='342_build_review_package_v2.py')))
C.write('342_result_审查包生成首次语法实现失败.json',{'status':'实现失败','exception':'Python3.10 f-string expression part cannot include a backslash; caught by AST before any builder execution.',
    '已执行':{'AST_parse_attempts':1,'builder_executions':0,'model_forwards':0,'optimizer_steps':0},
    'repair':'另存v2，先计算标题转义和配置路径再插值；证据与状态不变。',
    '未执行':['真实训练','GPU'],'未核':['最终包验证'],'source_sha256':C.sources()})
