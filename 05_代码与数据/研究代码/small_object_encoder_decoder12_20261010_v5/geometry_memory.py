"""Fine geometry field and coordinate-aware sparse memory reading.

Ordinary feature/coordinate operators; no unverified manifold or frequency-loss
assumption. Labels/GT never enter this encoder or its memory reader.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class GeometryField(nn.Module):
    def __init__(self, fine_channels=64, semantic_channels=256, channels=64, degraded=False):
        super().__init__();self.degraded=degraded
        self.fine_projection=nn.Sequential(nn.Conv2d(fine_channels,channels,1,bias=False),nn.GroupNorm(8,channels),nn.SiLU())
        self.context_projection=nn.Sequential(nn.Conv2d(semantic_channels,channels,1,bias=False),nn.GroupNorm(8,channels),nn.SiLU())
        self.local_geometry=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False),nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,1))

    def forward(self,fine,semantic):
        if self.degraded:
            fine=F.interpolate(F.avg_pool2d(fine,2,2),size=fine.shape[-2:],mode='bilinear',align_corners=False)
        local=self.fine_projection(fine)
        context=F.interpolate(self.context_projection(semantic),size=local.shape[-2:],mode='bilinear',align_corners=False)
        return self.local_geometry(local)+context


class GeometryMemoryRead(nn.Module):
    """9 local samples, four heads, input-feature-cell coordinate chart.

The chart contains relative offsets, reference dimensions and periodic
sub-cell center phase. These describe predicted/reference coordinates only.
The radius floor is one geometry-memory cell; no GT scale is used.
"""
    def __init__(self,query_channels=256,geometry_channels=64,output_channels=256,heads=4):
        super().__init__()
        assert geometry_channels%heads==0
        self.channels=geometry_channels;self.heads=heads;self.head_channels=geometry_channels//heads
        grid=torch.tensor([[-1.,-1.],[0.,-1.],[1.,-1.],[-1.,0.],[0.,0.],[1.,0.],[-1.,1.],[0.,1.],[1.,1.]])
        self.register_buffer('relative_grid',grid,persistent=False)
        self.position=nn.Sequential(nn.Linear(8,geometry_channels),nn.SiLU(),nn.Linear(geometry_channels,geometry_channels))
        self.query=nn.Linear(query_channels,geometry_channels)
        self.key=nn.Linear(geometry_channels,geometry_channels)
        self.value=nn.Linear(geometry_channels,geometry_channels)
        self.output=nn.Linear(geometry_channels,output_channels)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,query,geometry,reference):
        if query.shape[:2]!=reference.shape[:2] or reference.shape[-1]!=4:
            raise ValueError('Expected B,Q,4 reference boxes and B,Q,C queries')
        b,q,_=query.shape;_,c,h,w=geometry.shape
        if c!=self.channels:raise ValueError('Geometry field channel mismatch')
        ref=reference.float();cell=ref.new_tensor([1/w,1/h])
        radius=torch.maximum(ref[...,2:].abs()*.5,cell)
        delta=radius[:,:,None,:]*self.relative_grid.to(ref.dtype)[None,None]
        locations=ref[:,:,:2,None].transpose(-1,-2)+delta
        # Grid/sample fp32 avoids half precision pixel-phase quantization.
        sampled=F.grid_sample(geometry.float(),2*locations-1,mode='bilinear',padding_mode='border',align_corners=False)
        sampled=sampled.permute(0,2,3,1)
        delta_cell=delta/cell
        log_size=torch.log1p(ref[...,2:].abs()/cell)[:,:,None,:].expand(-1,-1,9,-1)
        phase=2*math.pi*(ref[...,:2]/cell)
        phase_features=torch.cat((phase.sin(),phase.cos()),-1)[:,:,None,:].expand(-1,-1,9,-1)
        chart=torch.cat((delta_cell,log_size,phase_features),-1)
        tokens=sampled+self.position(chart)
        qh=self.query(query).reshape(b,q,self.heads,self.head_channels)
        kh=self.key(tokens).reshape(b,q,9,self.heads,self.head_channels)
        vh=self.value(tokens).reshape(b,q,9,self.heads,self.head_channels)
        weights=((qh[:,:,None]*kh).sum(-1)/math.sqrt(self.head_channels)).float().softmax(2).to(vh.dtype)
        descriptor=(weights[...,None]*vh).sum(2).reshape(b,q,self.channels)
        return self.output(descriptor).to(query.dtype)

