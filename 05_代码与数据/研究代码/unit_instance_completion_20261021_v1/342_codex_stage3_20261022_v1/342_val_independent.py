"""Independent NumPy/Fraction oracles. No geom342/311/M0-solver imports."""
import numpy as np
from fractions import Fraction
import math,itertools

def brute_assignment(cost,legal,ids):
    n,m=cost.shape;best=None;answer=None;order=np.argsort(ids).tolist()
    def visit(j,used,owner,count,total):
        nonlocal best,answer
        if j==n:
            key=(-count,total,tuple(owner[x] if owner[x]>=0 else m for x in order))
            if best is None or key<best:best=key;answer=list(owner)
            return
        visit(j+1,used,owner+[-1],count,total)
        for g in range(m):
            if legal[j,g] and g not in used:
                visit(j+1,used|{g},owner+[g],count+1,total+Fraction(float(cost[j,g])))
    visit(0,set(),[],0,Fraction(0))
    return answer

def max_cardinality(legal):
    n,m=legal.shape;owners=[-1]*m
    def augment(j,seen):
        for g in np.flatnonzero(legal[j]):
            if g in seen:continue
            seen.add(int(g))
            if owners[g]<0 or augment(owners[g],seen):owners[g]=j;return True
        return False
    return sum(augment(j,set()) for j in range(n))

def certificate(cost,legal,ids,solution):
    n,m=cost.shape;owner=solution['cand_owner'];p=solution['dual_potentials'];s=n+m;t=s+1
    assert len(p)==n+m+2 and len(owner)==n
    assert solution['coverage']==max_cardinality(legal)
    chosen=[g for g in owner if g>=0];assert len(set(chosen))==len(chosen)
    fractions={(int(j),int(g)):Fraction(float(cost[j,g])) for j,g in np.argwhere(legal)}
    Q=1
    for value in fractions.values():Q=math.lcm(Q,value.denominator)
    sorted_ids=sorted(map(int,ids));B=m+1
    w={}
    for (j,g),value in fractions.items():
        rank=sorted_ids.index(int(ids[j]))
        w[j,g]=int(value*Q)*(B**n)+(g-m)*(B**(n-1-rank))
    shift=max(0,-min(w.values(),default=0))
    assert shift==solution['edge_cost_shift'] and Q==solution['cost_denominator']
    residual=[]
    for j,g in enumerate(owner):
        residual.append(p[j]-p[s] if g>=0 else p[s]-p[j])
    for g in range(m):residual.append(p[t]-p[n+g] if g in chosen else p[n+g]-p[t])
    for (j,g),value in w.items():
        if owner[j]==g:residual.append(-value-shift+p[n+g]-p[j])
        else:residual.append(value+shift+p[j]-p[n+g])
    assert min(residual,default=0)>=0
    exact_cost=sum((fractions[j,g] for j,g in enumerate(owner) if g>=0),Fraction(0))
    assert exact_cost==Fraction(solution['cost_integer_sum'],Q)
    return {'max_cardinality':solution['coverage'],'exact_residual_dual_feasible':True,
            'cost_exact_numerator':exact_cost.numerator,'cost_exact_denominator':exact_cost.denominator}

def raw640_boxes(normalized):
    # Float64 scalar source-space conversion, different from Torch unbind/stack.
    out=[]
    for cx,cy,w,h in normalized:
        out.append([(float(cx)-float(w)/2)*640,(float(cy)-float(h)/2)*640,
                    (float(cx)+float(w)/2)*640,(float(cy)+float(h)/2)*640])
    return np.asarray(out,dtype=np.float64).reshape(-1,4)

def window(final_normalized):
    # Enumerate candidate cells and count centers. No histogram flatten/unravel.
    boxes=raw640_boxes(final_normalized);centers=(boxes[:,:2]+boxes[:,2:])/2
    best=-1;selected=None
    for row in range(10):
        for col in range(10):
            count=0
            for x,y in centers:
                xin=col*64<=x<(col+1)*64 or (col==9 and x==640)
                yin=row*64<=y<(row+1)*64 or (row==9 and y==640)
                count+=int(xin and yin)
            if count>best:best=count;selected=(row,col)
    row,col=selected
    return selected,np.array([col*64,row*64,(col+1)*64,(row+1)*64],dtype=np.float64)

def gt_boxes(items,ow,oh):
    out=[]
    for a in items:
        x,y,w,h=a['bbox']
        out.append([float(x)/ow*640,float(y)/oh*640,(float(x)+float(w))/ow*640,(float(y)+float(h))/oh*640])
    return np.asarray(out,dtype=np.float64).reshape(-1,4)

def complete(gt,w):
    return np.array([all((x0>=w[0]-1e-6,y0>=w[1]-1e-6,x1<=w[2]+1e-6,y1<=w[3]+1e-6)) for x0,y0,x1,y1 in gt],dtype=bool)

def kg(gt,gw,w):
    out=np.zeros((len(gt),256),dtype=np.float64)
    for j in np.flatnonzero(gw):
        x0,y0,x1,y1=map(float,gt[j]);x=(x0+x1)/2;y=(y0+y1)/2
        ax=np.zeros(16);ay=np.zeros(16)
        for k in range(-1,17):
            ax[min(15,max(0,k))]+=max(0,1-abs((x-w[0])/(w[2]-w[0])*16-(k+.5)))
            ay[min(15,max(0,k))]+=max(0,1-abs((y-w[1])/(w[3]-w[1])*16-(k+.5)))
        out[j]=np.outer(ay,ax).reshape(-1)
    return out

def crop(pixels,w):
    # Direct original-pixel bilinear weights in float64. No normalized sampling
    # coordinates or grid_sample; x is column and y is row.
    a=np.asarray(pixels,dtype=np.float64)/255;h,ow=a.shape[:2]
    xs=[(w[0]+(u+.5)*(w[2]-w[0])/128)/640*ow-.5 for u in range(128)]
    ys=[(w[1]+(v+.5)*(w[3]-w[1])/128)/640*h-.5 for v in range(128)]
    xx,yy=np.meshgrid(xs,ys,indexing='xy');xx=np.clip(xx,0,ow-1);yy=np.clip(yy,0,h-1)
    ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);fx=(xx-ix)[...,None];fy=(yy-iy)[...,None]
    jx=np.minimum(ix+1,ow-1);jy=np.minimum(iy+1,h-1)
    result=(a[iy,ix]*(1-fx)+a[iy,jx]*fx)*(1-fy)+(a[jy,ix]*(1-fx)+a[jy,jx]*fx)*fy
    return result.transpose(2,0,1)

def crop_error_budget(ow,oh):
    # A-priori FP32 coordinate + bilinear arithmetic bound; independent direct
    # sampling is float64. Fixed formula, never fitted to observed differences.
    return 16*np.finfo(np.float32).eps*(ow+oh)+32*np.finfo(np.float32).eps
