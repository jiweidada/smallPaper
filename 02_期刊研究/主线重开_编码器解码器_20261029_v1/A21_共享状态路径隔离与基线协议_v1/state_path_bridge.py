"""One AST observer point in the native normalized NC-SSD path; no author file edits."""
import ast
import __future__
import copy
import hashlib
import types
import torch


def plain(x):
    return x.detach().as_subclass(torch.Tensor).contiguous().cpu().clone()


def fingerprint(x):
    if isinstance(x, torch.Tensor):
        v=plain(x)
        return {'shape':list(v.shape),'dtype':str(v.dtype),
                'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()}
    return x


def compile_native_bridge(attention, source):
    """Insert only (KV,nomi)=observer(...) after the normalized-branch KV calculation."""
    tree=ast.parse(source.read_text(encoding='utf-8'),filename=str(source))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Mamba2')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='non_casual_linear_attn')
    fn=copy.deepcopy(original)
    branch=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=='ssd_linear_norm')
    positions=[i for i,n in enumerate(branch.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KV' for t in n.targets)]
    assert len(positions)==1
    i=positions[0]
    assert isinstance(branch.body[i].value,ast.BinOp) and isinstance(branch.body[i].value.op,ast.MatMult)
    assert isinstance(branch.body[i+1],ast.Assign) and ast.unparse(branch.body[i+1].value)=='Q @ KV'
    hook=ast.parse('KV, nomi = self._state_path_bridge(KV, nomi, Q_pre, K_pre, V, Q, D)').body[0]
    ast.copy_location(hook,branch.body[i]);branch.body.insert(i+1,hook)
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    scope=dict(type(attention).non_casual_linear_attn.__globals__)
    exec(compile(module,str(source),'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),scope)
    patched=scope['non_casual_linear_attn']
    original_dump=ast.dump(original,include_attributes=False)
    recovered=copy.deepcopy(fn)
    recovered_branch=next(n for n in recovered.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=='ssd_linear_norm')
    del recovered_branch.body[i+1]
    assert ast.dump(recovered,include_attributes=False)==original_dump
    return patched,{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'original_method_AST_sha256':hashlib.sha256(original_dump.encode()).hexdigest(),
       'instrumented_method_AST_sha256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),
       'AST_insertions':1,'source_file_modified':False,'intervention':'Replace only KV and/or weighted unrotated K mean; retain native Q, V/D local skip, outer input-derived gate and local paths'}


class StateBridge:
    def __init__(self):
        self.donor=None
        self.replace_state=False
        self.replace_mean=False
        self.records=[]
        self.captured=None

    def __call__(self,KV,nomi,Q_pre,K_pre,V,Q,D):
        mean=K_pre.mean(dim=-2,keepdim=True)
        self.captured={'state':plain(KV),'mean':plain(mean)}
        record={'native_state':fingerprint(KV),'native_mean':fingerprint(mean),
            'fixed_read_Q_pre':fingerprint(Q_pre),'fixed_rotated_Q':fingerprint(Q),
            'fixed_local_V':fingerprint(V),'fixed_local_D':fingerprint(D),
            'replace_state':self.replace_state,'replace_mean':self.replace_mean}
        if self.replace_state:
            assert self.donor is not None
            KV=self.donor['state'].to(device=KV.device,dtype=KV.dtype)
        if self.replace_mean:
            assert self.donor is not None
            mean=self.donor['mean'].to(device=K_pre.device,dtype=K_pre.dtype)
            nomi=Q_pre @ mean.transpose(-2,-1) + 1e-6
        record.update(applied_state=fingerprint(KV),applied_mean=fingerprint(mean),applied_denominator=fingerprint(nomi))
        self.records.append(record)
        return KV,nomi


def attach_recorder(attention,function,operator_records):
    def recorded(self,*args,**kwargs):
        names=['x','dt','A','B','C','D','H','W','relpos','last_kv']
        before={n:fingerprint(v) for n,v in zip(names,args)}
        before.update({k:fingerprint(v) for k,v in kwargs.items()})
        result=function(self,*args,**kwargs)
        operator_records.append({'inputs':before,'merged_state_plus_local_skip':fingerprint(result[0]),
                                 'returned_state':fingerprint(result[1])})
        return result
    attention.non_casual_linear_attn=types.MethodType(recorded,attention)
