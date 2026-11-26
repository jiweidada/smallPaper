"""345c: preserve the failed profile and change metadata serialization only."""
import ast, difflib, importlib
C = importlib.import_module('345_context')
old = C.HERE / '345_flops_complete.py'
source = old.read_text(encoding='utf-8')
updated = source.replace(
    "    if isinstance(x,(bool,int,float,str)) or x is None:return x",
    "    if isinstance(x,float) and not math.isfinite(x):return 'nan' if math.isnan(x) else ('inf' if x>0 else '-inf')\n"
    "    if isinstance(x,(bool,int,float,str)) or x is None:return x")
begin = updated.index("    C.write('345_FLOPs手算预定口径.json'")
end = updated.index('    rows=[];calls=0', begin)
updated = updated[:begin] + """    policy=C.HERE/'345_FLOPs手算预定口径.json'
    assert policy.is_file()
    previous=C.read(C.HERE/'345_FLOPs部分标准输出登记.json')['rows']
    expected={(r['seed'],r['arm']):(r['registered'],r['manual'],r['sum']) for r in previous}
    C.write('345_FLOPs序列化修订口径_v2.json',{
        'authorization':str(C.SPEC/'345c_补充裁定_FLOPs写盘修正.txt'),
        'inherited_policy':{'path':str(policy),'sha256':C.sha(policy)},
        'metadata_encoding':'non-finite Python float constants only: inf/-inf/nan strings; tensor shape/dtype only; FLOPs integers unchanged',
        'expected_previous_stdout':previous,'source_sha256':C.sources()})
""" + updated[end:]
updated = updated.replace("                rows.append({'seed':seed", "                assert (registered,extra,registered+extra)==expected[(seed,arm)], ('345c total mismatch',seed,arm,registered,extra)\n                rows.append({'seed':seed")
updated = updated.replace("C.write('345_result_FLOPs覆盖补足.json'", "C.write('345_result_FLOPs覆盖补足_v2.json'")
updated = updated.replace("C.write('345_result_FLOPs实现失败_停止.json'", "C.write('345_result_FLOPs实现失败_停止_v2.json'")
updated = updated.replace("'GPU':'NA','已执行'", "'metadata_encoding':'inf/-inf/nan strings for non-finite scalar metadata; numeric FLOPs unchanged','all_8_match_previous_stdout':True,'policy_sha256':C.sha(policy),'GPU':'NA','已执行'")
original_ast = ast.parse(source)
new_ast = ast.parse(updated)
equal = {}
for name in ('tensors','count_output','manual','Counter'):
    a = next(n for n in original_ast.body if getattr(n,'name',None)==name)
    b = next(n for n in new_ast.body if getattr(n,'name',None)==name)
    equal[name] = ast.dump(a)==ast.dump(b)
assert all(equal.values()), equal
C.text('345_flops_complete_v2.py', updated)
C.text('345_FLOPs序列化修订.diff', ''.join(difflib.unified_diff(source.splitlines(True),updated.splitlines(True),fromfile=str(old),tofile=str(C.HERE/'345_flops_complete_v2.py'))))
C.write('345_FLOPs序列化修订范围.json', {'unchanged_AST':equal,'old_sha256':C.sha(old),'new_sha256':C.sha(C.HERE/'345_flops_complete_v2.py'),'calculation_changed':False,'model_changed':False,'optimizer_steps':0})
print('345c serialization-only source prepared; arithmetic/Counter AST unchanged',flush=True)
