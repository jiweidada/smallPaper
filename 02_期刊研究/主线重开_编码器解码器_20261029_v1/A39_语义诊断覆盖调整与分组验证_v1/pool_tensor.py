import torch
import torch.nn.functional as F


def pool_tensor(feature,boxes,chunk_size=64):
    """Post-forward read-only pooling; no detector module is called."""
    assert feature.ndim==4 and feature.shape[0]==1
    boxes=boxes.clamp(0,1)
    inner_axis=(torch.arange(7,device=feature.device,dtype=feature.dtype)+.5)/7
    iy,ix=torch.meshgrid(inner_axis,inner_axis,indexing='ij')
    inner_t=torch.stack([ix,iy],-1)
    outer_axis=(torch.arange(15,device=feature.device,dtype=feature.dtype)+.5)/15*2-1
    oy,ox=torch.meshgrid(outer_axis,outer_axis,indexing='ij')
    outer_t=torch.stack([ox,oy],-1)
    ring=(ox.abs()>.5)|(oy.abs()>.5)
    fine=[]; contrast=[]; counts=[]
    for part in boxes.split(chunk_size):
        n=len(part); wh=part[:,2:]-part[:,:2]
        coordinates=part[:,None,None,:2]+inner_t[None]*wh[:,None,None,:]
        grid=(coordinates*2-1).reshape(1,n*7,7,2)
        sampled=F.grid_sample(feature,grid,mode='bilinear',padding_mode='border',align_corners=False)
        inside=sampled.reshape(feature.shape[1],n,7,7).permute(1,0,2,3).mean((2,3))
        center=(part[:,:2]+part[:,2:])*.5
        outer=center[:,None,None,:]+outer_t[None]*wh[:,None,None,:]
        valid=ring[None]&((outer>=0)&(outer<=1)).all(-1)
        grid=(outer*2-1).reshape(1,n*15,15,2)
        sampled=F.grid_sample(feature,grid,mode='bilinear',padding_mode='border',align_corners=False)
        sampled=sampled.reshape(feature.shape[1],n,15,15).permute(1,0,2,3)
        count=valid.sum((1,2)); denom=count.clamp_min(1).to(feature.dtype)
        outside=(sampled*valid[:,None]).sum((2,3))/denom[:,None]
        difference=torch.where((count>0)[:,None],inside-outside,torch.zeros_like(inside))
        fine.append(inside); contrast.append(difference); counts.append(count)
    return torch.cat(fine),torch.cat(contrast),torch.cat(counts)
