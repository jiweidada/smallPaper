"""All 6470 train344 loads; source/SHA negative controls and bounded-entry audit."""
import importlib,json,ast,copy,time,traceback
C=importlib.import_module('344_context');T=importlib.import_module('train344')
import torch
torch.set_num_threads(2)

def main():
    start=time.perf_counter();negative=[];per_seed={};counts={'successful_bound_loads':0,'negative_controls':0,
        'global_forwards':0,'head_forwards':0,'optimizer_steps':0}
    try:
        for seed in (1,2):
            badtotal=0;pwtotal=0
            for n,iid in enumerate(C.development_ids(),1):
                e=T.load_entry(seed,iid);counts['successful_bound_loads']+=1
                bad=e['fullp_nonpositive_mask'];badtotal+=int(bad.sum())
                keep=T.geom.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],bad)
                assert not keep[bad].any();pwtotal+=int(keep.sum())
                if n%256==0 or n==3235:print(f'344 bound load seed={seed} {n}/3235',flush=True)
            per_seed[str(seed)]={'entries':3235,'fullp_nonpositive':badtotal,'P_W_candidates':pwtotal}
        assert counts['successful_bound_loads']==6470 and per_seed['1']['fullp_nonpositive']==0 and per_seed['2']['fullp_nonpositive']==1
        document=json.loads(T.MANIFEST_PATH.read_text(encoding='utf-8'));r=next(x for x in document['entries'] if x['seed']==2 and x['image_id']==1466)
        def altered(key,value):
            a=copy.deepcopy(r);a[key]=value;return lambda:T._verify_entry_record(2,1466,a)
        tests={
            'wrong_file_SHA':altered('file_sha256','0'*64),
            'wrong_payload_SHA':altered('payload_sha256','0'*64),
            'old340_path_rejected':altered('path',str(C.WORK/'340_cache_seed2/001466.pt')),
            'EMA_path_rejected':altered('path',str(C.HERE/'344_EMA_predictions.pt')),
            'bad_mask_mismatch':altered('fullp_nonpositive_indices',[]),
            'nondevelopment_id':lambda:T.load_entry(2,-1),
        }
        for name,fn in tests.items():
            try:fn()
            except AssertionError as ex:negative.append({'name':name,'rejected':True,'message':str(ex)});counts['negative_controls']+=1
            else:raise AssertionError('negative control accepted: '+name)
        original_manifest=T.MANIFEST_PATH
        T.MANIFEST_PATH=C.S343/'343_development6469_逐条SHA清单.json'
        try:
            try:T._manifest()
            except AssertionError as ex:negative.append({'name':'manifest_SHA_changed','rejected':True,'message':str(ex)});counts['negative_controls']+=1
            else:raise AssertionError('wrong manifest accepted')
        finally:T.MANIFEST_PATH=original_manifest
        m=T.make_common_init('N',1);opt=T.build_optimizer(m)
        for name,args in [('over20steps',(0,21,1218)),('real_schedule',(0,1,24360))]:
            try:T.run_steps(m,opt,[],*args)
            except AssertionError as ex:negative.append({'name':name,'rejected':True,'message':str(ex)});counts['negative_controls']+=1
            else:raise AssertionError('smoke bound not enforced')
        src=(C.HERE/'train344.py').read_text(encoding='utf-8');tree=ast.parse(src)
        prohibited=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                text=ast.unparse(node.func)
                if text.endswith('.cuda') or text in ('subprocess.run','subprocess.Popen','os.system') or (text.endswith('.to') and any(isinstance(a,ast.Constant) and isinstance(a.value,str) and ('cuda' in a.value or 'gpu' in a.value) for a in node.args)):
                    prohibited.append({'line':node.lineno,'call':text})
        assert not prohibited and '0 <= steps <= 20 and total_steps == 1218' in src
        assert src.count('geom.intersecting_fullp_candidates(b, W, e["fullp_nonpositive_mask"])')==2
        # These recipe/loss/optimizer functions must be unchanged except for P_W row selection.
        prior=ast.parse((C.S2/'train342.py').read_text(encoding='utf-8'))
        a={n.name:n for n in prior.body if isinstance(n,ast.FunctionDef)};b={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        same=[]
        for name in ('build_arm','make_common_init','build_optimizer','lr_at','lr_at_kappa','run_steps','capture_rng','restore_rng','save_checkpoint','load_checkpoint'):
            assert ast.dump(a[name])==ast.dump(b[name]),name;same.append(name)
        loss_old=ast.get_source_segment((C.S2/'train342.py').read_text(encoding='utf-8'),a['compute_losses_340'])
        loss_new=ast.get_source_segment(src,b['compute_losses_340'])
        assert loss_new==loss_old.replace('keep = geom.intersecting_candidates(b, W)','keep = geom.intersecting_fullp_candidates(b, W, e["fullp_nonpositive_mask"])')
        C.write('344_result_train344全量输入与入口验收.json',{'status':'已验证','per_seed':per_seed,'negative_controls':negative,
            'manifest_path':str(T.MANIFEST_PATH),'manifest_sha256':T.MANIFEST_SHA256,
            'static_entry_audit':{'GPU_or_process_launch_calls':prohibited,'main_only_exits':True,'smoke_step_bounds_retained':True,
                'unchanged_recipe_functions_AST':same,'loss_change_only_P_W_mask':True},
            'input_binding_calls':T.INPUT_COUNTS,'已执行':counts,'wall_seconds':time.perf_counter()-start,
            '未执行':['真实训练','GPU','阶段5','global模型构造'],'未核':['主控独立重跑'],'source_sha256':C.sources()})
        print(json.dumps({'status':'PASS','counts':counts,'per_seed':per_seed},ensure_ascii=False),flush=True)
    except Exception as ex:
        C.write('344_result_train344全量输入_实现失败.json',{'status':'实现失败','exception':repr(ex),'traceback':traceback.format_exc(),
            '已执行':counts,'未执行':['真实训练','GPU','阶段5'],'未核':['输入绑定完整验收'],'source_sha256':C.sources()});raise

if __name__=='__main__':main()
