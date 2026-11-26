"""Exact 296 M0: max cardinality, exact FP32-cost sum, owner-vector lex.

No epsilon/tolerance perturbation. Legal edges and costs are caller inputs.
The tertiary owner vector is ordered by the original flatten(query,class) ID.
Integer primal-dual flow returns a checkable residual-cost certificate.
"""
import heapq
import numpy as np

def exact_assignment(cost,legal,intrinsic_ids):
    a=np.asarray(cost,dtype=np.float32);legal=np.asarray(legal,dtype=bool)
    N,M=a.shape
    assert legal.shape==(N,M) and np.isfinite(a).all()
    ids=list(map(int,intrinsic_ids));assert len(ids)==N and len(set(ids))==N
    rank={j:r for r,j in enumerate(sorted(range(N),key=lambda j:ids[j]))}
    base=M+1;L=base**N
    coeff=[base**(N-1-rank[j]) for j in range(N)]
    pairs=[(int(j),int(g)) for j,g in np.argwhere(legal)]
    ratios={(j,g):float(a[j,g]).as_integer_ratio() for j,g in pairs}
    Q=max((q for p,q in ratios.values()),default=1)
    ci={(j,g):p*(Q//q) for (j,g),(p,q) in ratios.items()}
    weights={(j,g):ci[j,g]*L+(g-M)*coeff[j] for j,g in pairs}
    shift=max(0,-min(weights.values(),default=0))
    S=N+M;T=S+1;V=T+1;adj=[[] for _ in range(V)]
    def add(u,v,w):
        f=[v,len(adj[v]),1,w];r=[u,len(adj[u]),0,-w]
        adj[u].append(f);adj[v].append(r)
        return f
    for j in range(N):add(S,j,0)
    edges={(j,g):add(j,N+g,weights[j,g]+shift) for j,g in pairs}
    for g in range(M):add(N+g,T,0)
    potential=[0]*V;flow=0
    while True:
        dist=[None]*V;prev=[None]*V;dist[S]=0;heap=[(0,S)]
        while heap:
            d,u=heapq.heappop(heap)
            if dist[u]!=d:continue
            for k,e in enumerate(adj[u]):
                if not e[2]:continue
                v=e[0];reduced=e[3]+potential[u]-potential[v]
                assert reduced>=0
                nd=d+reduced
                if dist[v] is None or nd<dist[v]:
                    dist[v]=nd;prev[v]=(u,k);heapq.heappush(heap,(nd,v))
        if dist[T] is None:break
        far=max(d for d in dist if d is not None)
        for v in range(V):potential[v]+=far if dist[v] is None else dist[v]
        v=T
        while v!=S:
            u,k=prev[v];e=adj[u][k];e[2]-=1;adj[v][e[1]][2]+=1;v=u
        flow+=1
    owner=[-1]*N
    for (j,g),e in edges.items():
        if e[2]==0:owner[j]=g
    matched=[(j,g) for j,g in enumerate(owner) if g>=0]
    assert len(matched)==flow and len({g for j,g in matched})==flow
    assert all(e[3]+potential[u]-potential[e[0]]>=0 for u in range(V) for e in adj[u] if e[2])
    return {'cand_owner':owner,'coverage':flow,'matched_pairs':matched,
            'cost_integer_sum':sum(ci[j,g] for j,g in matched),'cost_denominator':Q,
            'owner_vector_lex':sum((g if g>=0 else M)*coeff[j] for j,g in enumerate(owner)),
            'dual_potentials':potential,'edge_cost_shift':shift,
            'intrinsic_candidate_ids':ids,'algorithm':'integer primal-dual max flow; exact composite objective; no epsilon'}
