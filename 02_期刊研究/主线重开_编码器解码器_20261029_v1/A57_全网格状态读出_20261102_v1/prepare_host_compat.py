"""Create a separate source compatibility copy; execute no external model code."""
from pathlib import Path
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
S=P.parent/'A56_尺度内对比读取_20261102_v1/post_terminal_source_screen/MambaSOD_pinned'
UP=P/'upstream_Vim_pinned'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=P/'host_compat_v1'
    target.mkdir(exist_ok=False)
    changes=[]
    def write(name,src,replacements):
        text=src.read_text('utf-8')
        for a,b in replacements:
            assert a in text,(name,a)
            text=text.replace(a,b)
        dst=target/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('x',encoding='utf-8') as f:f.write(text)
        changes.append({'target':name,'source':str(src),'source_sha256':sha(src),
                        'sha256':sha(dst),'replacements':[{'before':a,'after':b} for a,b in replacements]})
    normfix=('mamba_ssm.ops.triton.layernorm','mamba_ssm.ops.triton.layer_norm')
    write('model.py',S/'model.py',[("from ablation_modules import StandardCA, StandardSA", "# Unreleased ablation paths are unsupported; default use_ca/use_sa remain False."),normfix])
    write('encoder.py',S/'encoder.py',[("from ablation_modules import StandardFPN", "# Unreleased ablation path unsupported; default use_fpn remains False."),normfix])
    write('mamba_block.py',S/'Mamba_block.py',[normfix])
    write('cross_mamba.py',S/'Cross_mamba.py',[("from selective_scan_interface_ca import mamba_inner_fn","from scan_compat import cross_inner_fn as mamba_inner_fn")])
    write('vim/models_mamba.py',S/'models_mamba.py',[("from mamba_ssm.modules.mamba_simple import Mamba","from vim_mamba_compat import Mamba"),normfix])
    write('rope.py',UP/'vim/rope.py',[])
    write('vim_mamba_compat.py',UP/'mamba-1p1p1/mamba_ssm/modules/mamba_simple.py',[("from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj","from scan_compat import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj")])
    shutil.copy2(P/'scan_compat.py',target/'scan_compat.py')
    (target/'vim/__init__.py').write_text('',encoding='utf-8')
    receipt={'status':'STATIC_COMPATIBILITY_COPY_PREPARED_NOT_IMPORTED','files':changes,
             'source_baseline':'MambaSOD author commit dc5ca8b18d917a32205bb7e63c8d1882c30f91d9; default non-ablation code only',
             'upstream_Vim_commit':json.loads((P/'Vim_source_receipt.json').read_text())['commit'],
             'limitations':['Not a full-paper reproduction.','Legacy fused CrossMamba/Vim-v2 scan replaced by mathematically same unfused native ABI calls; numerical component verification still required.','Unreleased optional CA/SA/FPN ablations unsupported, never selected.'],
             'scan_compat_sha256':sha(P/'scan_compat.py'),'new_model_forwards':0,'new_optimizer_updates':0}
    with (P/'host_compat_source_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':receipt['status'],'source_files_copied':len(changes),'model_forwards':0}))


if __name__=='__main__':main()
