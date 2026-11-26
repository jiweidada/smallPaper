"""A57 source-derived alternatives, retaining one shared parameter topology."""
import types
import weakref
import torch
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from state_readout_v2 import encoded_feature_read,packed_query_read


def _scan(u,delta,A,B,C):
    return selective_scan_fn(u,delta,A,B,C,D=None,z=None,delta_bias=None,delta_softplus=False)


def install(model,arm):
    assert arm in ['B','M','Q']
    if arm=='B':return model
    decoder_ref=weakref.ref(model.decoder)

    def branch_forward(branch,queries,feat,feat_hw,num_groups=1):
        b,total_q,d=queries.shape
        decoder=decoder_ref()
        refs=decoder.reference_point.weight[:,:2].sigmoid()
        refs=refs.repeat(num_groups,1)[None].expand(b,-1,-1)
        assert refs.shape==(b,total_q,2)
        x,res=branch.feat_refine(feat,residual=None)
        context=x+res
        directions=[]
        for block,backward in [(branch.cross_q_reads_f.cross_fw,False),
                               (branch.cross_q_reads_f.cross_bw,True)]:
            raw=context.flip(1) if backward else context
            xy=1-refs if backward else refs
            normalized=block.norm(raw.to(block.norm.weight.dtype))
            mixer=block.mixer
            xz=F.linear(normalized,mixer.in_proj.weight,mixer.in_proj.bias).transpose(1,2)
            # Convolve the context before creating any read events.
            from scan_compat import _context
            u,z,pars,dt,Bdyn=_context(xz,mixer.conv1d.weight,mixer.conv1d.bias,
                                     mixer.x_proj.weight,mixer.dt_proj.weight,mixer.d_state)
            read_source=queries if arm=='Q' else raw
            read_source=block.norm_query(read_source.to(block.norm_query.weight.dtype))
            C=F.linear(F.linear(read_source,mixer.in_proj_q.weight,mixer.in_proj_q.bias),
                       mixer.q_proj.weight)
            with torch.autocast(device_type='cuda',enabled=False):
                delta=F.softplus(dt.float()+mixer.dt_proj.bias.float()[None,:,None])
                common={'u':u.float(),'delta':delta,'A':-mixer.A_log.float().exp(),
                        'B':Bdyn.float(),'z':z.float(),'D':mixer.D.float(),
                        'reference_xy':xy.float(),'hw':feat_hw,'scan':_scan}
                if arm=='Q':read=packed_query_read(**common,Cquery=C.float())
                else:read=encoded_feature_read(**common,Ccontext=C.float().transpose(1,2).contiguous())
            out=F.linear(read.to(raw.dtype),mixer.out_proj.weight,mixer.out_proj.bias)
            with torch.autocast(device_type='cuda',enabled=False):
                image=raw.float().transpose(1,2).reshape(b,d,*feat_hw)
                grid=(xy.float()*2-1).unsqueeze(2)
                local=F.grid_sample(image,grid,mode='bilinear',padding_mode='border',align_corners=False)
                local=local.squeeze(-1).transpose(1,2)
            local=local.to(out.dtype)
            directions.append(out+local)
        return branch.out_proj(directions[0]+directions[1])

    for layer in model.decoder.layers:
        for branch in layer.mqi.scale_branches:
            branch.forward=types.MethodType(branch_forward,branch)
    return model
