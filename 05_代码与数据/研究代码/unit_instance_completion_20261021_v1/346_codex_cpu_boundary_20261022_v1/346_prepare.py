"""Prepare an exclusively new boundary adapter; retain train345 unmodified."""
import ast,difflib,importlib,inspect
C=importlib.import_module('346_context')
E=C.E

def main():
    old_path=C.S345/'train345.py'
    old=inspect.getsource(E.execute)
    new=old.replace('def execute(', 'def execute_boundary(',1)
    new=new.replace("_mode='formal', _resume=None):", "_mode='formal', _resume=None, _case=None):",1)
    replacements=[
      ("    require(_mode in CURRENT_CPU_CAP and max_steps <= CURRENT_CPU_CAP[_mode], 'acceptance_step_cap')", """    require(_mode == 'formal', '346_formal_only')
    require(_case in C.CASES, '346_unknown_case')
    specification=C.CASES[_case]
    require(max_steps == specification['end'], '346_endpoint_not_whitelisted')
    require(_resume is not None, '346_fixture_required')
    C.verify_sources()
    verify_file(_resume['path'], _resume['sha256'], '346_resume_receipt')
    preliminary=C.read(_resume['path'])
    require(preliminary.get('execution_kind') == C.KIND and preliminary.get('case') == _case,
            '346_fixture_kind_or_case')
    require(specification['start'] <= preliminary['step'] < max_steps,
            'resume_endpoint_not_after_checkpoint')
    require(max_steps - preliminary['step'] <= specification['maximum_actual_updates'],
            '346_actual_update_cap')"""),
      ("    require(not run_dir.exists(), 'run_directory_exists', run_dir)", """    require(run_dir.is_relative_to(C.RUN_ROOT.resolve()), '346_run_directory_scope')
    require(not run_dir.exists(), 'run_directory_exists', run_dir)"""),
      ("    config, order, curve, binding = validate_bindings(arm, seed, _mode)", """    config, order, curve, binding = validate_bindings(arm, seed, _mode)
    binding=C.binding_for(binding,_case)
    require(preliminary['binding'] == binding, '346_preliminary_binding')"""),
      ("        model.load_state_dict(checkpoint['model'], strict=True)", """        require(checkpoint['format'] == C.KIND and checkpoint['synthetic_origin_step'] == specification['start'],
                '346_checkpoint_kind')
        require(checkpoint['actual_optimizer_updates_since_fixture'] == checkpoint['step']-specification['start'],
                '346_actual_update_count')
        model.load_state_dict(checkpoint['model'], strict=True)"""),
      ("        require(len(step_logs) == step, 'resume_log_length')", "        require(len(step_logs) == step-specification['start'], 'resume_log_length')"),
      ("'Actual wall times are a separate ledger. Byte equality covers deterministic per-step numeric/batch logs; wall times are not deterministic.'", "'Synthetic acceptance: timing covers executed batches only. Cursor/order sizes are not counts of past training. Skipped prefix has no fabricated logs or elapsed time.'"),
      ("        payload = {'format': '345_full_state_v1', 'binding': binding,", "        payload = {'format': C.KIND, 'binding': binding,"),
      ("                   'epoch_partial_seconds': epoch_partial_seconds}", """                   'epoch_partial_seconds': epoch_partial_seconds,
                   'synthetic_origin_step': specification['start'],
                   'actual_optimizer_updates_since_fixture': step-specification['start'],
                   'real_training_steps':0}"""),
      ("                   'position_in_epoch': batch_index * config['batch']})", """                   'position_in_epoch': batch_index * config['batch'],
                   'execution_kind':C.KIND,'case':_case,
                   'actual_optimizer_updates_since_fixture':step-specification['start']})"""),
      ("                                'batches': batch_index, 'windows': len(order['epochs'][epoch])})", """                                'cursor_batches_at_epoch_end': batch_index,
                                'order_epoch_windows': len(order['epochs'][epoch]),
                                'actual_batches_since_fixture':sum(r['epoch']==epoch for r in step_logs),
                                'actual_windows_since_fixture':sum(len(r['image_ids']) for r in step_logs if r['epoch']==epoch)})"""),
      ("    result = {'status': 'CPU短验收已执行；真实训练未执行',", "    result = {'status': '346构造边界CPU短验收；真实训练未执行',"),
      ("              'input_binding_counts_this_process': dict(T.INPUT_COUNTS)}", """              'input_binding_counts_this_process': dict(T.INPUT_COUNTS),
              'execution_kind':C.KIND,'case':_case,
              'synthetic_origin_step':specification['start'],
              'synthetic_unexecuted_prefix_steps':specification['start'],
              'actual_step_log_records_since_fixture':len(step_logs)}""")]
    for before,after in replacements:
        assert new.count(before)==1,('replacement scope',before,new.count(before))
        new=new.replace(before,after,1)
    new=new.replace('345_initial_state','346_initial_state').replace('345_run_binding','346_run_binding')
    new=new.replace('345_checkpoint','346_checkpoint').replace('345_steps','346_steps')
    new=new.replace('345_epoch_times','346_epoch_times').replace('345_run_result','346_run_result')
    old_ast=ast.parse(old).body[0];new_ast=ast.parse(new).body[0]
    a=next(n for n in ast.walk(old_ast) if isinstance(n,ast.While))
    b=next(n for n in ast.walk(new_ast) if isinstance(n,ast.While))
    # Only epoch timing field names/count labels change inside the loop. The
    # complete numerical, ID slicing, LR, optimizer and cursor logic must match.
    def strip_timing(node):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            fn=node.value.func
            if isinstance(fn,ast.Attribute) and fn.attr=='append' and isinstance(fn.value,ast.Name) and fn.value.id=='epoch_times':
                return ast.Pass()
        return node
    class Strip(ast.NodeTransformer):
        def visit_Expr(self,node):return strip_timing(node)
    aa=Strip().visit(ast.parse(ast.unparse(a)));bb=Strip().visit(ast.parse(ast.unparse(b)))
    assert ast.dump(aa)==ast.dump(bb),'training-loop AST drift beyond epoch-time metadata'
    header='''"""346 synthetic boundary acceptance only. No full-training launch path.

The loop is copied from sealed train345; the only in-loop edit renames epoch
timing metadata so a synthetic cursor cannot masquerade as executed history.
All loss/geometry/optimizer functions remain imported, never edited.
"""
import importlib,random,time
from pathlib import Path
C=importlib.import_module('346_context')
import numpy as np
import torch
torch.set_num_threads(2)
E=C.E;T=C.T
HARD_TOTAL_STEPS=E.HARD_TOTAL_STEPS
require=E.require;verify_file=E.verify_file;write_json=E.write_json
state_sha=E.state_sha;batch_sha=E.batch_sha
validate_bindings=E.validate_bindings;make_optimizer=E.make_optimizer;check_cursor=E.check_cursor

'''
    full=header+new+"\n\nif __name__=='__main__':\n    raise SystemExit('346 boundary acceptance worker only; real training is not authorized.')\n"
    ast.parse(full)
    C.text('train346.py',full)
    C.text('346_train345_to_train346.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
          fromfile=str(old_path)+':execute',tofile=str(C.HERE/'train346.py')+':execute_boundary')))
    C.write('346_数值循环继承证明.json',{'status':'已验证（静态AST）',
           'entire_while_loop_AST_equal_except_epoch_timing_metadata':True,
           'unchanged':['batch ID slicing','cache loader/collate','lr route','loss','backward','joint clip','optimizer step','step/epoch/batch cursor transitions','1218 hard-stop predicate','epoch checkpoint trigger'],
           'changed_outside_loop':['whitelisted synthetic fixture/cursor caps','receipt execution-kind binding','truthful local log length','distinct acceptance checkpoint schema','exclusive new output paths'],
           'old_source_sha256':C.sha(old_path),'new_source_sha256':C.sha(C.HERE/'train346.py'),
           'model_forwards':0,'optimizer_steps':0})
    C.write('346_用户指令与预定验收口径.json',{
           'direct_user_authorization':'你自己设定目标 本机验证看看',
           'interpretation':'仅承接上一条提出的本机CPU构造边界短验收；不授权真实训练、GPU或SSH。',
           'master_accepted_prior':'用户已确认345 FLOPs与B通过；恢复/篡改部分只读JSON未独立重跑。',
           'cases':C.CASES,'arms':list('NSGR'),'seeds':[1,2],
           'expected_total_actual_optimizer_updates':40,'expected_head_forwards':40,
           'expected_cache_loads':432,'expected_native_loss_calls':864,
           'input':'原345 SHA绑定的正式3235×6顺序；不使用mini文件，不变更缓存/模型/损失/学习率。',
           'fixture':'model/optimizer/RNG来自未训练公共初态，只设置明确标记的虚拟游标。被跳过前缀未执行，不填充历史loss或epoch耗时。',
           'independent_oracle':'枚举完整顺序的嵌套循环，逐步比对ID、epoch、批次和offset；两seed同case四臂比较真实batch SHA。',
           'negative_tests':['1219 budget','non-whitelisted endpoint205','completed1218 resume','346 checkpoint rejected by original345 binding','mini order rejected'],
           'stop_rule':'任一验收失败立即写实现失败并停下，不更改模型或配方后重试。',
           'scope_exclusions':['真实训练','val评估','G Delta','bootstrap','GPU','SSH','heldout图像或缓存'],
           'source_sha256':C.source_sha()})
    old_seal=C.S345/'345_交付封存SHA256_v3.json'
    for p,item in C.read(old_seal)['artifacts'].items():assert C.sha(p)==item['sha256'],p
    C.write('346_旧封存起始核验.json',{'status':'已验证','old_seal_path':str(old_seal),
            'old_seal_sha256':C.sha(old_seal),'verified_files':len(C.read(old_seal)['artifacts']),
            'head_forwards':0,'optimizer_steps':0})
    sources=C.source_sha()
    C.write('346_冻结源码与旧封存绑定.json',{'sources':sources,
            'old_seal':{'path':str(old_seal),'sha256':C.sha(old_seal)},'execution_kind':C.KIND,
            'cases':C.CASES,'user_authorization':'本次用户直接指令，范围见预定验收口径；真实训练0步'})
    value=C.sha(C.HERE/'346_冻结源码与旧封存绑定.json')
    C.text('346_bindings_anchor.py',f'"""346 CPU acceptance trust anchor."""\nBUNDLE_SHA256={value!r}\n')
    print('346 preparation PASS: numerical loop AST identical; old seal unchanged; optimizer_steps=0',flush=True)

if __name__=='__main__':main()
