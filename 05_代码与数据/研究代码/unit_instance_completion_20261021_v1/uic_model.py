import math, torch, torch.nn as nn, torch.nn.functional as F
C=10; D=128; GRID=16; WIN=128

class ConvGN(nn.Module):
    def __init__(s,ci,co,k=3,st=1):
        super().__init__(); s.c=nn.Conv2d(ci,co,k,st,k//2,bias=False); s.g=nn.GroupNorm(8,co); s.a=nn.SiLU()
    def forward(s,x): return s.a(s.g(s.c(x)))
class ResBlock(nn.Module):
    def __init__(s,ch):
        super().__init__(); s.c1=nn.Conv2d(ch,ch,3,1,1,bias=False); s.g1=nn.GroupNorm(8,ch); s.c2=nn.Conv2d(ch,ch,3,1,1,bias=False); s.g2=nn.GroupNorm(8,ch); s.a=nn.SiLU()
    def forward(s,x):
        y=s.a(s.g1(s.c1(x))); y=s.g2(s.c2(y)); return s.a(x+y)
class Encoder(nn.Module):
    def __init__(s,d=D):
        super().__init__(); s.s1=ConvGN(3,32,st=2); s.s2=ConvGN(32,64,st=2); s.s3=ConvGN(64,d,st=2); s.res=ResBlock(d)
    def forward(s,x):
        x=s.s1(x); x=s.s2(x); x=s.s3(x); x=s.res(x); return x
class MassHead(nn.Module):
    def __init__(s,d=D,cls=C):
        super().__init__(); s.conv=nn.Conv2d(d,cls,1)
        nn.init.constant_(s.conv.weight,0); nn.init.constant_(s.conv.bias, math.log(math.expm1(1.0/256)))
    def forward(s,f):
        m=F.softplus(s.conv(f)); return m.flatten(2)  # [B,cls,256]
class QueryDecoder(nn.Module):
    def __init__(s,d=D,cls=C,nq=4,heads=4,layers=2,ff=512):
        super().__init__(); s.nq=nq; s.d=d
        s.q=nn.Parameter(torch.randn(nq,d)*0.02)
        dec=nn.TransformerDecoderLayer(d,heads,ff,dropout=0.0,batch_first=True,norm_first=True)
        s.dec=nn.TransformerDecoder(dec,layers)
        s.register_buffer("ref_center",torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]],dtype=torch.float32))
        s.cls_head=nn.Linear(d,cls); s.box_head=nn.Linear(d,4)
    def _init(s):
        nn.init.constant_(s.cls_head.weight,0)
        with torch.no_grad(): s.cls_head.bias.copy_(torch.full((C,),math.log(1e-4/(1-1e-4))))
        nn.init.constant_(s.box_head.weight,0); nn.init.constant_(s.box_head.bias,0)
    def forward(s,memory,qinit=None):
        b=memory.shape[0]; q=s.q.unsqueeze(0).expand(b,-1,-1).clone()
        if qinit is not None: q=q+qinit
        out=s.dec(q,memory); logits=s.cls_head(out); delta=s.box_head(out)
        cxcywh=torch.cat([s.ref_center.unsqueeze(0).expand(b,-1,-1), torch.full_like(delta[...,2:],0.25)],-1)
        return logits, cxcywh+delta, out
class CandidateEncoder(nn.Module):
    def __init__(s,d=D):
        super().__init__(); s.l1=nn.Linear(15,d); s.l2=nn.Linear(d,d)
    def forward(s,x): return F.silu(s.l2(F.silu(s.l1(x))))
class AuxPlan(nn.Module):
    def __init__(s,d=D):
        super().__init__(); s.Wq=nn.Linear(d,d,bias=False); s.Wk=nn.Linear(d,d,bias=False); s.kap=nn.Sequential(nn.Linear(2*d,d),nn.SiLU(),nn.Linear(d,1))
    def _init(s):
        nn.init.constant_(s.Wq.weight,0); nn.init.xavier_uniform_(s.Wk.weight)
        nn.init.constant_(s.kap[-1].weight,0); nn.init.constant_(s.kap[-1].bias,0)
class QueryToPTokens(nn.Module):
    # 2 layers of query->P cross-attention only (317 counts these as MHA + 1 preLN each), out_proj zero-init.
    # matches 317 algebraic S addition 133888: 2*(3dd+3d+dd+d + 2d) + Linear(10,128,nobias)
    def __init__(s,d=D,heads=4):
        super().__init__()
        s.attn=nn.ModuleList([nn.MultiheadAttention(d,heads,dropout=0.0,batch_first=True) for _ in range(2)])
        s.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        for a in s.attn:
            nn.init.constant_(a.out_proj.weight,0); nn.init.constant_(a.out_proj.bias,0)
    def forward(s,queries,ptokens):
        x=queries
        for a,nm in zip(s.attn,s.norm):
            y,_=a(x,ptokens,ptokens); x=nm(x+y)
        return x
class CountHead(nn.Module):
    def __init__(s,d=D,cls=C):
        super().__init__(); s.fc=nn.Linear(cls,d,bias=False); nn.init.constant_(s.fc.weight,0)
    def forward(s,nc): return s.fc(nc)

def d_out(boxes, cell_centers):
    cx,cy,w,h=boxes.unbind(-1); px=cell_centers[:,0].unsqueeze(0); py=cell_centers[:,1].unsqueeze(0)
    dx=torch.clamp(torch.abs(px-cx.unsqueeze(1))-w.unsqueeze(1)/2,min=0)/w.unsqueeze(1).clamp(min=1e-6)
    dy=torch.clamp(torch.abs(py-cy.unsqueeze(1))-h.unsqueeze(1)/2,min=0)/h.unsqueeze(1).clamp(min=1e-6)
    return dx+dy

def shared_operator(Cmat, kappa, mbar, cand_cls):
    # Cmat:[N,L] cand x cell ; kappa:[C] per-class or [N]; mbar:[C,L]; cand_cls:[N]
    # Numerically stable: Z = exp(-k) + sum_l mbar*exp(-C) computed in log space via logsumexp.
    N,L=Cmat.shape
    kappa_j = kappa[cand_cls] if (kappa.dim()==1 and kappa.shape[0]==C) else kappa
    mrow=mbar[cand_cls]  # [N,L]
    # log terms: -kappa_j (scalar per cand) and log(mrow)-Cmat per cell; handle mrow==0 (-inf)
    tiny=torch.finfo(Cmat.dtype).tiny
    logm=torch.where(mrow>0, mrow.clamp_min(tiny).log(), torch.full_like(mrow,-float('inf')))
    logterms=torch.cat([(-kappa_j).unsqueeze(1), logm - Cmat], dim=1)   # [N, L+1]
    logZ=torch.logsumexp(logterms, dim=1)                                # [N]
    Z=torch.exp(logZ)
    expC=torch.exp(-Cmat); p0=torch.exp((-kappa_j-logZ)).clamp(min=0,max=1.0)
    p_ij=(mrow*expC)/Z.unsqueeze(1)
    u=torch.zeros(C,L,device=Cmat.device)
    for c in range(C):
        m=(cand_cls==c)
        if m.any(): u[c]=p_ij[m].sum(0)
    a=torch.ones(C,L,device=Cmat.device); nz=u>0; a[nz]=torch.clamp(mbar[nz]/u[nz],max=1.0)
    T=p_ij*a[cand_cls]; z=p0+((1-a[cand_cls])*p_ij).sum(1)
    r=torch.zeros(C,L,device=Cmat.device)
    for c in range(C):
        m=(cand_cls==c)
        if m.any(): r[c]=mbar[c]-T[m].sum(0)
    r=torch.clamp(r,min=0)
    return {"u":u,"a":a,"T":T,"z":z,"r":r,"p0":p0,"p_ij":p_ij}

class FourArm(nn.Module):
    def __init__(s, arm):
        super().__init__(); s.arm=arm
        s.enc=Encoder(); s.mass=MassHead(); s.dec=QueryDecoder()
        s.cand=CandidateEncoder(); s.aux=AuxPlan()
        if arm=="S": s.ptok=QueryToPTokens(); s.count=CountHead()
        if arm=="G": s.kappa_c=nn.Parameter(torch.zeros(C))
        s.dec._init(); s.aux._init()
    def build_cand_feats(s, P_boxes_xyxy_local, P_scores, P_cls):
        # [N,4] unclipped xyxy, onehot10, score1 = 15
        oh=F.one_hot(P_cls,C).float()
        return torch.cat([P_boxes_xyxy_local, oh, P_scores.unsqueeze(1)],dim=1)
    def forward(s, rgb, P=None, cell_centers=None):
        """rgb:[B,3,128,128]; P dict with boxes_xyxy_local[B,N,4], scores[B,N], cls[B,N]; W=1 local coords.
        Returns per-batch: pred_logits[B,4,C], pred_boxes[B,4,4], aux plan (C^A[..],k^A,r etc), and gen state."""
        f=s.enc(rgb); B=f.shape[0]
        mbar=s.mass(f).detach()  # stopgrad in generation per 317 sg(m)
        m=s.mass(f)              # for losses
        mem=f.flatten(2).transpose(1,2)  # [B,256,128]
        outs=[]
        for b in range(B):
            if P is not None:
                Pb=P["boxes_xyxy_local"][b]; Ps=P["scores"][b]; Pc=P["cls"][b]
                feats=s.build_cand_feats(Pb,Ps,Pc)
                a_j=s.cand(feats)  # [N,128]
                # aux plan C^A_ij = d_out(i,j) - <Wq a_j, Wk f_i>/sqrt128
                Wqa=s.aux.Wq(a_j)  # [N,128]
                Wkf=s.aux.Wk(f[b].reshape(D,-1).transpose(0,1))  # [256,128]
                dot=Wqa@Wkf.t()/math.sqrt(D)  # [N,256]
                # compute d_out from cxcywh boxes: convert xyxy->cxcywh
                x0,y0,x1,y1=Pb.unbind(-1); cxb=(x0+x1)/2; cyb=(y0+y1)/2; wb=(x1-x0).clamp(min=1e-6); hb=(y1-y0).clamp(min=1e-6)
                pb_cxcywh=torch.stack([cxb,cyb,wb,hb],-1)
                dO=d_out(pb_cxcywh, cell_centers)  # [N,256]
                CA=dO-dot
                kA=s.aux.kap(torch.cat([a_j, f[b].flatten(1).mean(1).unsqueeze(0).expand(a_j.shape[0],-1)],-1)).squeeze(-1)  # [N]
                # generation b depending on arm
                if s.arm=="N":
                    bmass=mbar[b]  # [C,256]
                    # N does not read P; gen uses m only
                    gen_res=None
                elif s.arm=="G":
                    # b=r^G: C^G=d_out fixed, kappa^G=kappa_c per class
                    gen=shared_operator(dO, s.kappa_c, mbar[b], Pc)
                    bmass=gen["r"]
                    gen_res=gen
                elif s.arm=="R":
                    gen=shared_operator(CA, kA, mbar[b], Pc)
                    bmass=gen["r"]
                    gen_res=gen
                else: # S
                    bmass=mbar[b]; gen_res=None
            else:
                a_j=None; CA=None; kA=None; bmass=mbar[b]; gen_res=None
            # generation mass summary (bmass depends on arm: N/S=mbar, G=r^G, R=r^A).
            # In this proxy the mass vector is exposed as bmass for the operator/parity checks;
            # full attention-value injection into the decoder is design-only here (see task4 pipeline spec).
            spat=torch.log(bmass.sum(0)+1e-4/256)  # [256] spatial logit contribution (recorded, not wired)
            qinit=None; ptokens=None
            if s.arm=="S" and P is not None:
                # P-token cross-attn
                ptokens=a_j.unsqueeze(0)  # [1,N,128]
                qcur=s.dec.q.unsqueeze(0)
                qadd=s.ptok(qcur,ptokens).squeeze(0)  # [4,128]
                # count head
                nc=torch.zeros(C,device=rgb.device)
                if P is not None:
                    for c in range(C): nc[c]=(Pc==c).sum().float()
                qadd=qadd+s.count(torch.log1p(nc))
                qinit=qadd.unsqueeze(0)
            logits,boxes,hidden=s.dec(mem[b].unsqueeze(0), qinit)
            out_b={"pred_logits":logits[0],"pred_boxes":boxes[0],"m":m[b],"mbar":mbar[b],
                   "a_j":a_j,"CA":CA,"kA":kA,"gen":gen_res}
            outs.append(out_b)
        return outs