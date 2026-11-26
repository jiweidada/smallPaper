"""CPU-only audit logic tests: synthetic events, no neural forward or training."""
import ast,hashlib,importlib.util,json,pathlib,sys,types
import torch
ROOT=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path('/25085400120/DEIM/experiments/small_object_encoder_decoder12_20261010_v2/code/training_audit.py')
OLD_MODEL=OLD.with_name('small_object_model.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def declarations(p):
    tree=ast.parse(p.read_text());out={}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):out[node.name]=ast.dump(node,include_attributes=False)
    return out
def main():
    out=ROOT/'mock_results';out.mkdir(exist_ok=False)
    constants=ast.parse(OLD_MODEL.read_text());prefixes=next(ast.literal_eval(n.value) for n in constants.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='EXTRA_PREFIXES' for x in n.targets))
    sys.modules['small_object_model']=types.SimpleNamespace(EXTRA_PREFIXES=prefixes)
    spec=importlib.util.spec_from_file_location('repaired_audit',ROOT/'training_audit.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    old,new=declarations(OLD),declarations(ROOT/'training_audit.py')
    assert old['forward']==new['forward'] and old['digest']==new['digest'] and old['__init__']==new['__init__']
    ref=out/'synthetic_reference.jsonl';ref.write_text(''.join(json.dumps({'attempt':i})+'\n' for i in range(1,101)))
    class Handle:
        def remove(self):pass
    class MockModel:
        training=True
        def __init__(self):self.values={p+'test_parameter':torch.nn.Parameter(torch.zeros(2)) for p in prefixes}
        def named_parameters(self):return self.values.items()
        def register_forward_pre_hook(self,*args,**kwargs):return Handle()
        def state_dict(self):return {n:p.detach().clone() for n,p in self.values.items()}
    class MockOptimizer:
        def register_step_pre_hook(self,*args):return Handle()
        def register_step_post_hook(self,*args):return Handle()
    results=[]
    for case in ['early_serial_gates','tiny_exact_movements']:
        d=out/case;d.mkdir();model=MockModel();opt=MockOptimizer();audit=mod.TrainingAudit(model,opt,d,ref,sha(ref));audit.steps=7;audit.attempts=10
        for n,p in model.named_parameters():p.grad=torch.full_like(p,1e-30)
        audit.before_step(opt,(),{})
        with torch.no_grad():
            for n,p in model.named_parameters():
                if case=='tiny_exact_movements' or n.startswith(prefixes[1]):p.copy_(torch.tensor([1e-30,-1e-30]))
        # Call the observer with synthetic events; no optimizer.step is called.
        audit.after_step(opt,(),{})
        row=audit.rows[0];assert (d/'parameters_actual_step8.pth').is_file()
        assert all(v['norm']==0 and v['norm64']>0 and v['nonzero_elements']==2 for v in row['gradients_after_native_unscale_clip'].values())
        expected_failure=case=='early_serial_gates';failed=False
        try:audit.epoch_complete(0,{},types.SimpleNamespace(updates=8))
        except RuntimeError as e:assert 'first completed epoch' in str(e);failed=True
        assert failed==expected_failure
        if expected_failure:assert (d/'failed_epoch1_parameters.pth').is_file()
        else:
            assert all(v>0 for v in row['delta_from_initial'].values())
            assert all(v==2 for v in row['changed_elements_from_initial'].values())
        audit.close()
        results.append({'case':case,'synthetic_eighth_event_not_rejected':True,'epoch1_guard_rejected_as_expected':failed,'FP32_norm_underflow_observed':True,'FP64_norm_and_exact_tensor_counts_correct':True,'real_optimizer_constructions':0,'real_optimizer_steps':0,'neural_model_forwards':0})
    result={'status':'PASS_CPU_SYNTHETIC_AUDIT_LOGIC_ONLY','cases':results,'original_input_hash_RNG_parity_methods_AST_exact':True,'original_constructor_AST_exact':True,'repair_sha256':sha(ROOT/'training_audit.py'),'original_audit_sha256':sha(OLD),'test_sha256':sha(__file__),'actual_optimizer_steps':0,'neural_forward_calls':0,'epochs_added':0,'formal_training_restart':False,'limits':['Mocked audit step/EMA counters are synthetic fixtures; no actual optimization or EMA update.','No real full-epoch AMP training integration or detection efficacy verified.','Changes affect observation/check timing only; scientific model sources unchanged.']}
    (out/'completion.json').write_text(json.dumps(result,indent=2));(out/'manifest.json').write_text(json.dumps({'files':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}},indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
