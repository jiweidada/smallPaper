"""Create train347 without changing any old file; prove numerical equivalence."""
import ast,difflib,importlib
C=importlib.import_module('347_context')

def replace_once(text,before,after):
    assert text.count(before)==1,(before,text.count(before))
    return text.replace(before,after,1)

def main():
    original=(C.S345/'train345.py').read_text(encoding='utf-8')
    finish=original.index('"""',3)+3
    updated='''"""347 fixed-budget CPU entry, based on sealed train345.

Formal execution permits at most1218 steps; the private miniature cap remains6.
Current authorization executes G_seed1 ten steps for acceptance only.
Eight complete scientific runs require separate explicit user authorization.
"""'''+original[finish:]
    updated=replace_once(updated,'import argparse',
        "import sys as _bootstrap_sys\nfrom pathlib import Path as _BootstrapPath\n_bootstrap_sys.path.insert(0,str(_BootstrapPath(__file__).resolve().parent.parent/'345_codex_contracts_20261022_v1'))\nimport argparse")
    updated=replace_once(updated,"C = importlib.import_module('345_context')",
        "C = importlib.import_module('345_context')\nMETA = importlib.import_module('347_context')")
    updated=replace_once(updated,"CURRENT_CPU_CAP = {'formal': 2, 'mini_acceptance': 6}",
        "CURRENT_CPU_CAP = {'formal': 1218, 'mini_acceptance': 6}")
    updated=replace_once(updated,'    return config, order, curve, binding',
        "    binding = {**binding, '347_source_bundle_sha256': META.verify_sources(), 'entry_version': 'train347'}\n    return config, order, curve, binding")
    for name in ('initial_state','run_binding','checkpoint','steps','epoch_times','run_result'):
        updated=updated.replace('345_'+name,'347_'+name)
    updated=replace_once(updated,"'345_full_state_v1'","'347_full_state_v1'")
    updated=replace_once(updated,"'status': 'CPU短验收已执行；真实训练未执行'", "'status': 'CPU步骤执行完成'")
    updated=replace_once(updated,"'global_forwards': 0, 'real_training_steps': 0", "'global_forwards': 0")
    updated=replace_once(updated,"'input_binding_counts_this_process': dict(T.INPUT_COUNTS)}",
        "'input_binding_counts_this_process': dict(T.INPUT_COUNTS),\n              'full_budget_completed': step == HARD_TOTAL_STEPS}")
    updated=replace_once(updated,"description='345 CPU entry acceptance; 真实训练需另行授权'",
        "description='347 fixed-budget CPU entry; full scientific runs require separate authorization'")
    a=ast.parse(original);b=ast.parse(updated)
    old_defs={n.name:n for n in a.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    new_defs={n.name:n for n in b.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    unchanged={}
    for name in ('GuardError','require','verify_file','write_json','state_sha','batch_sha',
                 'validate_order','make_optimizer','check_cursor'):
        unchanged[name]=ast.dump(old_defs[name])==ast.dump(new_defs[name])
    assert all(unchanged.values()),unchanged
    old_loop=next(n for n in ast.walk(old_defs['execute']) if isinstance(n,ast.While))
    new_loop=next(n for n in ast.walk(new_defs['execute']) if isinstance(n,ast.While))
    assert ast.dump(old_loop)==ast.dump(new_loop),'numeric/cursor/checkpoint-trigger loop changed'
    def guards(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require']
    assert guards(old_defs['execute'])==guards(new_defs['execute']),'execution guard drift'
    old_validation=old_defs['validate_bindings'].body
    new_validation=new_defs['validate_bindings'].body
    assert [ast.dump(n) for n in old_validation[:-1]]==[ast.dump(n) for n in new_validation[:-2]]
    assert ast.dump(old_validation[-1])==ast.dump(new_validation[-1])
    C.text('train347.py',updated)
    C.text('347_train345_to_train347.diff',''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),
          fromfile=str(C.S345/'train345.py'),tofile=str(C.HERE/'train347.py'))))
    C.write('347_静态继承证明.json',{'status':'已验证（静态）','entire_training_while_loop_AST_equal':True,
            'all_execute_require_guards_AST_equal':True,'original_binding_checks_retained_in_order':True,
            'unchanged_helper_AST':unchanged,'formal_cap_change':[2,1218],'mini_cap_unchanged':6,
            'hard_budget_unchanged':1218,'CLI_flags_unchanged':['arm','seed','run_dir','device','max_steps'],
            'other_changes':['standalone local import bootstrap','new entry source SHA added to existing binding',
                'exclusive347 output names/schema','neutral execution status and full_budget_completed; actual physical step counts retained, intent not guessed'],
            'no_coordinate_or_loss_change':True,'old_sha256':C.sha(C.S345/'train345.py'),
            'new_sha256':C.sha(C.HERE/'train347.py'),'head_forwards':0,'optimizer_steps':0})
    C.write('347_用户授权与预定验收口径.json',{
            'direct_user_authorization':'按你修正的方案执行347：另存train347.py，正式CPU模式上限改为1218，保留其他守卫和SHA绑定，不改封存文件。用G_seed1验收10步，前2步对照现有345日志，完整10步对照最多跑10步的隔离参考进程。通过后封存并停下，八个完整run仍未授权。',
            'subject':{'arm':'G','seed':1,'max_steps':10,'mode':'formal','device':'cpu'},
            'reference':'fresh process, original sealed train345, in-memory formal cap2->10 only; hard cap1218 and source files unchanged; process rejects step11',
            'first2_reference':str(C.S345/'345_B_CPU_acceptance_runs/345_G_seed1_formal_order_first2/345_steps.json'),
            'full10_exact_comparison':['model tensor bytes/dtype/shape','optimizer state and param_groups','torch/numpy/python RNG',
                'all per-step log bytes including batch SHA/loss/lr/kappa','step/epoch/position/shuffle/initial identity'],
            'allowed_binding_difference':'subject adds entry_version=train347 and pinned347 source bundle SHA; original binding fields all equal',
            'timing_policy':'actual wall times separately recorded, never required bit-identical',
            'planned_counts':{'subject_optimizer_steps':10,'reference_optimizer_steps':10,'head_forwards':20,
                'cache_loads':320,'native_loss_calls':640,'scientific_full_runs':0},
            'guards_to_check':['1219 rejected','existing run refused','config SHA','manifest SHA','shuffle bytes',
                'formal rejects mini','both346 synthetic fixture kinds rejected','CLI rejects lr override','reference cap11 rejected'],
            'stop_rule':'any acceptance failure -> implementation failure and stop; no model/recipe repair or scientific full run',
            'not_authorized':['eight full runs','GPU','SSH','val/G Delta/bootstrap','heldout image/cache reads'],
            'source_sha256':C.sources()})
    seals=[C.S345/'345_交付封存SHA256_v3.json',C.S346/'346_交付封存SHA256.json']
    audit=[]
    for path in seals:
        doc=C.read(path)
        for name,item in doc['artifacts'].items():assert C.sha(name)==item['sha256'],name
        audit.append({'path':str(path),'sha256':C.sha(path),'verified_files':len(doc['artifacts'])})
    C.write('347_旧封存起始核验.json',{'status':'已验证','rows':audit,'head_forwards':0,'optimizer_steps':0})
    C.write('347_入口源码SHA绑定.json',{'sources':C.sources(),'old_seals':audit,
            'subject_entry':'train347.py','only_current_execution':'G_seed1 ten-step CPU acceptance; eight complete runs not authorized'})
    value=C.sha(C.HERE/'347_入口源码SHA绑定.json')
    C.text('347_bindings_anchor.py',f'"""347 source binding trust anchor; included in final seal."""\nBUNDLE_SHA256={value!r}\n')
    print('347 preparation PASS: numerical loop and execution guards unchanged; formal cap1218, miniature6; optimizer steps=0',flush=True)

if __name__=='__main__':main()
