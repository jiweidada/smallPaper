"""Independent scalar NumPy algorithms; no Torch, geom342, or frozen code imports."""
import numpy as np,math

def cap(mass,cost,null,classes,valid):
    m=np.asarray(mass,dtype=np.float64);cost=np.asarray(cost,dtype=np.float64);null=np.asarray(null,dtype=np.float64)
    B,C,I=m.shape;J=cost.shape[-1];p=np.zeros((B,I,J));p0=np.zeros((B,J));u=np.zeros_like(m)
    for b in range(B):
        for j in range(J):
            if not valid[b,j]:continue
            c=int(classes[b,j]);logs=[math.log(m[b,c,i])-cost[b,i,j] if m[b,c,i]>0 else -math.inf for i in range(I)]
            mx=max([-float(null[b,j])]+logs)
            weights=[math.exp(x-mx) for x in logs];n=math.exp(-float(null[b,j])-mx);den=n+sum(weights)
            p[b,:,j]=np.asarray(weights)/den;p0[b,j]=n/den;u[b,c]+=p[b,:,j]
    T=np.zeros_like(p);z=np.zeros_like(p0);explained=np.zeros_like(m)
    for b in range(B):
        for j in range(J):
            if not valid[b,j]:continue
            c=int(classes[b,j]);back=0.
            for i in range(I):
                a=min(1.,m[b,c,i]/u[b,c,i]) if u[b,c,i]>0 else 1.
                T[b,i,j]=a*p[b,i,j];back+=p[b,i,j]-T[b,i,j];explained[b,c,i]+=T[b,i,j]
            z[b,j]=p0[b,j]+back
    return dict(p=p,p0=p0,T=T,z=z,u=u,explained=explained,residual=np.maximum(m-explained,0),mass=m)

def geometry(boxes):
    a=np.asarray(boxes,dtype=np.float64);B,J,_=a.shape;out=np.zeros((B,256,J))
    for b in range(B):
        for row in range(16):
            for col in range(16):
                x=(2*col+1)/32;y=(2*row+1)/32
                for j in range(J):
                    x0,y0,x1,y1=map(float,a[b,j]);assert x1>x0 and y1>y0
                    dx=x0-x if x<x0 else x-x1 if x>x1 else 0.
                    dy=y0-y if y<y0 else y-y1 if y>y1 else 0.
                    out[b,row*16+col,j]=dx/(x1-x0)+dy/(y1-y0)
    return out
