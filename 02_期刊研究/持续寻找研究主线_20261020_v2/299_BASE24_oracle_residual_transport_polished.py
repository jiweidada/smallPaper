"""292 + 298: same oracle residual allocation, fixed solver precision polishing.
Root runs real data after review; synthetic-only never reads real cache contents.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '2'
sys.dont_write_bytecode = True
import argparse, gc, hashlib, importlib.util, json, pathlib, time, traceback
ROOT = pathlib.Path(__file__).resolve().parent
REUSE = ROOT / '283_BASE24_explanation_context_probe.py'
SHA283 = 'b2152b6b3e8065ce0d85ba5ea940bc6032ae1c0adefe58dc8d241c4e97f5b96e'
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != SHA283: raise RuntimeError('283 changed')
_spec = importlib.util.spec_from_file_location('_293_reuse283', REUSE)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)
R, P, H, np, torch = B.R, B.P, B.H, B.np, B.torch
need, sha, save = B.need, B.sha, B.save
from scipy.optimize import minimize
from scipy.special import logsumexp
import scipy
from scipy.optimize import _lbfgsb_py, _lbfgsb
PROTOCOL = ROOT / '298_B_熵运输求解器精度修订.txt'
INHERITED = {
 ROOT / '292_B_几何oracle残余强对照_事前协议.txt': 'e661f2936fab96a14aff6fab2485efe3096d60d1d84ec951c52229994071cf26',
 ROOT / '293_BASE24_oracle_residual_transport.py': '41af2daccced08552bdffffeb7b7d074882666a29d82056eb1b59787eda3741c'}
CONTEXT = ROOT / '284_BASE24_explanation_context_probe'
ANCHORS284 = {
 '00_sources_and_synthetic.json': '9afae6ba9a933575357a78e99fdd6bd505d44edebafc3136d470c4fe1ae792ef',
 '01_results.json': 'da0d380c27fdc76bf935955fad17edf3745a510d536208f2fc8927df6a1f3578',
 '02_output_sha256.json': 'ade4a6a8a104a27c4cc2e06f02c6b24dfbee4b64cffb9e50de5b9b0365c39809'}
SOLVER_SOURCES = {
 pathlib.Path(_lbfgsb_py.__file__).resolve(): '011e8f59fcf9c601c14fa1841bf5797bb4bdc2a37c08a6037bd0bf4a1a53fee0',
 pathlib.Path(_lbfgsb.__file__).resolve(): '0eca107a397e2dada7ebb1c67efdfe81341a0c6f2f134ee8b06a887f9176ca21'}
ABS, ACTIVE = 2e-8, 1e-9
POLISH_TOL, MAX_NEWTON, MAX_BACKTRACK, MAX_COORD_SWEEPS = 1e-12, 24, 40, 256
MAX_BRACKET, COORD_BISECTIONS = 64, 80
GRID, EPSILON, KAPPA = 16, 1., 0.
ARMS = ('I', 'I_cap', 'J')
COUNTS = dict(npz_batches=0, images=0, windows=0, convex_solves=0,
              prediction_PT_loads=0, model_builds=0, checkpoint_loads=0, model_forwards=0,
              autograd_calls=0, model_backwards=0, optimizer_steps=0, heldout_reads=0)


def scatter(centers, labels, window, size=GRID):
    x0, y0, x1, y1 = np.asarray(window, dtype=float); extent = np.array([x1-x0, y1-y0])
    need((extent > 0).all(), 'window extent')
    points = np.array([(x0+(x+.5)*extent[0]/size, y0+(y+.5)*extent[1]/size)
                       for y in range(size) for x in range(size)])
    weights = np.zeros((len(centers), size*size), dtype=float)
    for g, center in enumerate(centers):
        need(np.isfinite(center).all() and ((center >= [x0,y0]) & (center <= [x1,y1])).all(), 'complete GT center')
        u = (center-[x0,y0])*size/extent-.5; low = np.floor(u).astype(int); frac = u-low
        for dy in (0,1):
            for dx in (0,1):
                x = np.clip(low[0]+dx,0,size-1); y = np.clip(low[1]+dy,0,size-1)
                weights[g,y*size+x] += (frac[0] if dx else 1-frac[0])*(frac[1] if dy else 1-frac[1])
    need(np.allclose(weights.sum(1), 1, rtol=0, atol=1e-14), 'unit boundary mass')
    mass = np.zeros((10,size*size), dtype=float)
    for g,c in enumerate(labels): mass[int(c)] += weights[g]
    need(abs(mass.sum()-len(centers)) <= 1e-12*(1+len(centers)), 'total GT mass')
    return mass, weights, points


def geometry_cost(points, boxes):
    cost = np.full((len(points),len(boxes)), np.inf)
    good = B.valid_boxes(boxes)
    for j in np.flatnonzero(good):
        xy0, xy1 = boxes[j,:2], boxes[j,2:]
        outside = np.maximum(np.maximum(xy0-points,points-xy1),0)
        cost[:,j] = (outside/(xy1-xy0)).sum(1)
    return cost


def conditional(m, cost, alpha):
    logw = np.log(m)[:,None]-cost-alpha[:,None]
    ld = np.logaddexp(-KAPPA, logsumexp(logw,axis=0))
    return np.exp(logw-ld), np.exp(-KAPPA-ld), ld


def cap_once(m, t, z):
    rows = t.sum(1); factor = np.minimum(1,np.divide(m,rows,out=np.ones_like(m),where=rows>0))
    fixed = t*factor[:,None]; null = z+(t-fixed).sum(0)
    return fixed, null


def primal(m, cost, t, z):
    nz = t > 0
    need(np.isfinite(cost[nz]).all(), 'positive forbidden flow')
    ref = np.broadcast_to(m[:,None],t.shape)
    return float(np.sum(t[nz]*(cost[nz]+np.log(t[nz]/ref[nz])-1))
                 +np.sum(z[z>0]*(np.log(z[z>0])-1+KAPPA)))


def certify(m, cost, alpha, t, z, fdual):
    n = len(z); rows = t.sum(1); grad = m-rows
    fixed, null = cap_once(m,t,z)
    p = primal(m,cost,t,z); pf = primal(m,cost,fixed,null); dual = -fdual-n
    gap = pf-dual; budget = ABS*(1+abs(pf)+abs(dual))
    pg = np.where(alpha>ACTIVE, np.abs(grad), np.maximum(-grad,0))
    scale = 1+float(m.max(initial=0))
    station = 0.
    if np.any(t>0):
        nz = t>0; ref = np.broadcast_to(m[:,None],t.shape)
        # Infer the per-column multiplier independently from z, not cached denominator.
        terms = np.log(t[nz]/ref[nz])+cost[nz]+np.broadcast_to(alpha[:,None],t.shape)[nz]
        terms -= np.broadcast_to(np.log(z)[None,:]+KAPPA,t.shape)[nz]
        station = float(np.max(np.abs(terms)))
    checks = dict(nonnegative=bool((t>=0).all() and (z>=0).all() and (alpha>=0).all()),
        column=float(np.max(np.abs(t.sum(0)+z-1),initial=0)),
        row_excess=float(np.max(np.maximum(rows-m,0),initial=0)),
        projected_gradient=float(np.max(pg,initial=0)),
        complementarity=float(np.max(np.abs(alpha*grad),initial=0)), stationarity=station,
        feasible_column=float(np.max(np.abs(fixed.sum(0)+null-1),initial=0)),
        feasible_row_excess=float(np.max(np.maximum(fixed.sum(1)-m,0),initial=0)),
        primal=p, feasible_primal=pf, dual=dual, feasible_gap=gap, gap_budget=budget,
        gap_upper=max(0.,gap)+budget, feasibility_repair_L1=float(np.abs(t-fixed).sum()))
    need(checks['nonnegative'] and checks['column']<=ABS and checks['row_excess']<=ABS*scale
         and checks['projected_gradient']<=ABS*scale and checks['complementarity']<=ABS*(1+abs(p))
         and station<=ABS*10 and checks['feasible_column']<=ABS and checks['feasible_row_excess']<=ABS*scale
         and -budget<=gap<=budget, 'KKT/primal/dual certificate failed: '+str(checks))
    checks['passed'] = True
    return checks


def dual_state(m, cost, alpha):
    t,z,ld=conditional(m,cost,alpha)
    f=float(ld.sum()+np.dot(m,alpha)); g=m-t.sum(1)
    scale=1+float(m.max(initial=0))
    pg=np.where(alpha>ACTIVE,np.abs(g),np.maximum(-g,0))
    merit=max(float(np.max(pg,initial=0))/scale,
              float(np.max(np.maximum(-g,0),initial=0))/scale,
              float(np.max(np.abs(alpha*g),initial=0))/(1+abs(f)))
    return f,g,t,z,merit


def polish_dual(m, cost, initial, coordinate_only=False):
    """Refine the SAME convex dual. No modified objective or certificate tolerances."""
    alpha=np.maximum(0,np.array(initial,dtype=float,copy=True))
    f,g,t,z,merit=dual_state(m,cost,alpha)
    initial_f,initial_merit=f,merit
    stats=dict(initial_merit=merit,initial_objective=f,newton_steps=0,backtracking_evaluations=0,
               coordinate_sweeps=0,coordinate_only=coordinate_only,newton_fallback_reason=None)
    for _ in range(0 if coordinate_only else MAX_NEWTON):
        if merit<=POLISH_TOL: break
        hessian=np.diag(t.sum(1))-t@t.T
        free=(alpha>0)|(g<0); direction=np.zeros_like(alpha); usable=False
        for __ in range(len(alpha)+1):
            ids=np.flatnonzero(free)
            if not len(ids): break
            try: step=np.linalg.solve(hessian[np.ix_(ids,ids)],-g[ids])
            except np.linalg.LinAlgError:
                stats['newton_fallback_reason']='singular_Hessian'; break
            if not np.isfinite(step).all():
                stats['newton_fallback_reason']='nonfinite_direction'; break
            wrong=(alpha[ids]==0)&(step<0)
            if wrong.any(): free[ids[wrong]]=False; continue
            direction[ids]=step; usable=True; break
        if not usable or float(g@direction)>=0:
            stats['newton_fallback_reason']=stats['newton_fallback_reason'] or 'no_feasible_descent'; break
        negative=direction<0
        length=min(1.,float(np.min(-alpha[negative]/direction[negative]))) if negative.any() else 1.
        accepted=False
        for __ in range(MAX_BACKTRACK):
            proposal=np.maximum(0,alpha+length*direction)
            nf,ng,nt,nz,nmerit=dual_state(m,cost,proposal)
            roundoff=64*np.finfo(float).eps*(1+abs(f))
            armijo=f+1e-4*float(g@(proposal-alpha))+roundoff
            stats['backtracking_evaluations']+=1
            if np.isfinite(nf) and nf<=armijo and (nmerit<merit or nmerit<=POLISH_TOL):
                alpha,f,g,t,z,merit=proposal,nf,ng,nt,nz,nmerit; accepted=True; break
            length*=.5
        if not accepted:
            stats['newton_fallback_reason']='backtracking_exhausted'; break
        stats['newton_steps']+=1
    # Monotone coordinate minimizers give a fixed deterministic fallback.
    base=m[:,None]*np.exp(-cost)
    for _ in range(MAX_COORD_SWEEPS):
        if merit<=POLISH_TOL: break
        before=f; weights=base*np.exp(-alpha[:,None])
        for i in range(len(m)):
            other=1.+weights.sum(0)-weights[i]
            need(np.isfinite(other).all() and (other>0).all(),'coordinate null denominator')
            def row_at(a):
                w=base[i]*np.exp(-a)
                return float(np.sum(w/(other+w)))
            if row_at(0)<=m[i]: alpha[i]=0.
            else:
                low=0.; high=max(1.,float(alpha[i]))
                for __ in range(MAX_BRACKET):
                    if row_at(high)<=m[i]: break
                    high*=2.
                else: raise RuntimeError('coordinate root bracket exhausted')
                for __ in range(COORD_BISECTIONS):
                    mid=(low+high)*.5
                    if row_at(mid)>m[i]: low=mid
                    else: high=mid
                alpha[i]=(low+high)*.5
            weights[i]=base[i]*np.exp(-alpha[i])
        f,g,t,z,merit=dual_state(m,cost,alpha)
        need(f<=before+64*np.finfo(float).eps*(1+abs(before)),'coordinate objective increased')
        stats['coordinate_sweeps']+=1
    stats.update(final_merit=merit,final_objective=f,objective_change=f-initial_f,
                 max_alpha_change=float(np.max(np.abs(alpha-initial),initial=0)),
                 converged=bool(merit<=POLISH_TOL))
    need(stats['converged'],'fixed high-precision polishing exhausted: '+str(stats))
    return alpha,stats


def solve_class(mass, cost):
    need(mass.ndim==1 and cost.shape[0]==len(mass) and (mass>=0).all() and np.isfinite(mass).all(), 'transport input')
    need(not np.isnan(cost).any() and (cost>=0).all(), 'nonnegative geometry cost')
    n = cost.shape[1]; active = np.flatnonzero(mass>0)
    result = {a:dict(T=np.zeros(cost.shape),z=np.ones(n)) for a in ARMS}
    alpha_full = np.zeros(len(mass))
    if not len(active) or not n:
        cert = dict(passed=True, analytic=True, reason='mass_zero_null_only' if not len(active) else 'no_conditions',
                    gap_upper=0., feasibility_repair_L1=0., row_excess=0., feasible_gap=0.)
        return result, alpha_full, cert
    m, c = mass[active],cost[active]
    ti,zi,_ = conditional(m,c,np.zeros(len(m)))
    tc,zc = cap_once(m,ti,zi)
    def fg(a):
        t,z,ld = conditional(m,c,a)
        return float(ld.sum()+np.dot(m,a)), m-t.sum(1)
    opt = minimize(fg,np.zeros(len(m)),jac=True,bounds=[(0,None)]*len(m),method='L-BFGS-B',
                   options=dict(ftol=1e-15,gtol=1e-10,maxiter=4000,maxls=60))
    COUNTS['convex_solves'] += 1
    need(bool(opt.success), 'convex solver did not converge: '+str(opt.message))
    polished,polishing=polish_dual(m,c,opt.x)
    tj,zj,_ = conditional(m,c,polished); cert = certify(m,c,polished,tj,zj,fg(polished)[0])
    cert.update(iterations=int(opt.nit), evaluations=int(opt.nfev), solver_success=bool(opt.success), message=str(opt.message))
    cert['polishing']=polishing
    for a,t,z in (('I',ti,zi),('I_cap',tc,zc),('J',tj,zj)):
        result[a]['T'][active] = t; result[a]['z'] = z
    alpha_full[active] = polished
    ri = np.maximum(0,m-ti.sum(1)); rc = m-tc.sum(1); rj = np.maximum(0,m-tj.sum(1))
    need(np.max(np.abs(rc-ri),initial=0)<=ABS, 'onepass residual equals independent clip')
    need(np.max(rj-ri,initial=0)<=ABS*(1+float(m.max())), 'joint raw pointwise contract')
    need(np.max(np.maximum(tc.sum(1)-m,0),initial=0)<=ABS
         and np.max(np.abs(tc.sum(0)+zc-1),initial=0)<=ABS, 'onepass feasible')
    return result, alpha_full, cert


def normalized(residual, bound, mass_total, nconditions):
    total = float(residual.sum())
    if total==0:
        return None, 'empty_mass_null_only' if mass_total==0 else 'exact_zero_residual'
    if total<=bound: return None, 'numerically_unresolved'
    return residual/total, 'no_conditions' if nconditions==0 else 'resolved'


def allocation(mass, points, boxes, labels):
    cost = geometry_cost(points,boxes)
    raw = {a:mass.copy() for a in ARMS}; nulls = {a:np.ones(len(boxes)) for a in ARMS}
    alpha = np.zeros_like(mass); certs=[]
    for c in range(10):
        jj = np.flatnonzero(labels==c)
        try: solved,aa,cert = solve_class(mass[c],cost[:,jj])
        except Exception as exc: raise RuntimeError(f'transport category {c}, mass {mass[c].sum()}, conditions {len(jj)}') from exc
        alpha[c]=aa
        certs.append(dict(category=c,conditions=len(jj),mass=float(mass[c].sum()),**cert))
        for a in ARMS:
            raw[a][c] = mass[c]-solved[a]['T'].sum(1); nulls[a][jj]=solved[a]['z']
    residual = {a:np.maximum(0,raw[a]) for a in ARMS}
    mass_total = float(mass.sum()); n = len(boxes)
    gap_upper = sum(c['gap_upper'] for c in certs); repair = sum(c['feasibility_repair_L1'] for c in certs)
    bounds = {'I':ABS*(1+mass_total+n), 'I_cap':ABS*(1+mass_total+n), 'J':float(np.sqrt(2*n*gap_upper)+repair)}
    attentions={}; summaries={}
    for a in ARMS:
        att,status = normalized(residual[a],bounds[a],mass_total,n); attentions[a]=att
        summaries[a]=dict(mass=mass_total, conditions=n, raw_signed_total=float(raw[a].sum()),
            residual_total=float(residual[a].sum()), negative_clip_mass=float(np.maximum(-raw[a],0).sum()),
            null_total=float(nulls[a].sum()), normalized_status=status, normalization_error_bound=bounds[a],
            conservation_error=float(abs(raw[a].sum()-(mass_total-n+nulls[a].sum()))))
        need(summaries[a]['conservation_error']<=ABS*(1+mass_total+n),'mass/null conservation')
    need(np.max(np.abs(residual['I']-residual['I_cap']),initial=0)<=ABS,'global onepass residual identity')
    point_violation=float(np.max(residual['J']-residual['I_cap'],initial=0))
    need(point_violation<=ABS*(1+mass_total),'global J pointwise residual contract')
    total_i=float(residual['I_cap'].sum()); total_j=float(residual['J'].sum())
    lam=total_j/total_i if total_i>0 else None
    common=float(np.abs(residual['J']-lam*residual['I_cap']).sum()) if lam is not None else None
    l1=float(np.abs(attentions['J']-attentions['I_cap']).sum()) if all(attentions[a] is not None for a in ('J','I_cap')) else None
    return dict(raw=raw,r=residual,z=nulls,alpha=alpha,attention=attentions,certificates=certs,summaries=summaries,
                comparisons=dict(max_pointwise_J_minus_Icap=point_violation,common_scale_lambda=lam,
                    common_scale_L1_error=common,normalized_L1_J_minus_Icap=l1))


def metric_row(weights, label, raw, residual, attention):
    return dict(signed_class_raw=float(weights@raw[label]), class_raw=float(weights@residual[label]),
                spatial_raw=float(weights@residual.sum(0)),
                class_normalized=None if attention is None else float(weights@attention[label]),
                spatial_normalized=None if attention is None else float(weights@attention.sum(0)))


def cohort_report(rows):
    result={}
    for name in ('all','primary_Geo0','geometry_covered','primary_Geo0_tiny','geometry_covered_tiny','secondary_D0_P'):
        subset=[r for r in rows if name=='all' or r[name]]
        arms={}
        for a in ARMS:
            arms[a]={}
            for metric in ('signed_class_raw','class_raw','spatial_raw','class_normalized','spatial_normalized'):
                values=[r['arms'][a][metric] for r in subset if r['arms'][a][metric] is not None]
                arms[a][metric]=dict(n=len(values),sum=float(sum(values)),mean=float(np.mean(values)) if values else None)
        paired={}
        for metric in ('signed_class_raw','class_raw','spatial_raw','class_normalized','spatial_normalized'):
            differences=[r['arms']['J'][metric]-r['arms']['I_cap'][metric] for r in subset
                         if r['arms']['J'][metric] is not None and r['arms']['I_cap'][metric] is not None]
            paired[metric]=dict(n=len(differences),sum=float(sum(differences)),
                                mean=float(np.mean(differences)) if differences else None)
        result[name]=dict(GT=len(subset),arms=arms,paired_J_minus_Icap=paired)
    return result


def probe_image(old, ctx, archive, hw, tv, arrays):
    index=old['image_index']; prefix=f'i{index}_final'
    gt64=archive[prefix+'_GT']; labels=archive[prefix+'_GT_labels']; final64=archive[prefix+'_cached_boxes']
    need(np.array_equal(gt64.astype(np.float32).astype(float),gt64)
         and np.array_equal(final64.astype(np.float32).astype(float),final64),'FP32 recovery')
    gt=torch.from_numpy(gt64.astype(np.float32)); final=torch.from_numpy(final64.astype(np.float32))
    gtxy=R.pixels(gt,hw,tv).numpy().astype(float); finalxy=R.pixels(final,hw,tv).numpy().astype(float)
    wh=torch.tensor([hw[1],hw[0]],dtype=torch.float32)
    centers=(gt[:,:2]*wh).numpy().astype(float); qcenters=(final[:,:2]*wh).numpy().astype(float)
    tiny=(gt64[:,2:]*np.array([hw[1],hw[0]])).min(-1)<4
    need(len(gt64)==old['GT']==ctx['GT'] and tiny.sum()==old['actual_sub4']==ctx['actual_sub4'],'GT identity')
    flat=np.array(old['post']['P_flat'],dtype=np.int64); pboxes=np.array(old['post']['P_boxes_xyxy'],dtype=float).reshape(-1,4)
    need(np.array_equal(pboxes,finalxy[flat//10]),'fullP query boxes')
    cells=R.cell_ids(qcenters[flat//10],hw); gcells=R.cell_ids(centers,hw); grid=R.make_grid(hw)
    primary=set(ctx['thresholds']['0.5']['definitions']['primary_G_geo0']['full_image_zero_degree_GT'])
    secondary=set(ctx['thresholds']['0.5']['definitions']['secondary_D0_P']['full_image_zero_degree_GT'])
    windows={}
    for arm in B.WINDOWS:
        window=old['windows'][arm]; cell=window['cell']
        if cell is None: windows[arm]=dict(NA=True); continue
        need(cell==ctx['frozen_cells'][arm] and np.array_equal(grid[cell],window['xyxy_pixels']),'original window')
        full,_,_=R.visibility(grid[cell],gtxy,gcells,cell); gg=np.flatnonzero(full); pp=np.flatnonzero(cells==cell)
        context=ctx['thresholds']['0.5']['definitions']['primary_G_geo0']['windows'][arm]
        need(gg.tolist()==context['complete_GT'] and flat[pp].tolist()==context['supplies'],'complete/supply identity')
        mass,weights,points=scatter(centers[gg],labels[gg],grid[cell])
        try: solved=allocation(mass,points,pboxes[pp],flat[pp]%10)
        except Exception as exc: raise RuntimeError(f'image {index}, window {arm}, cell {cell}') from exc
        stem=f'i{index}_{arm}'; arrays[stem+'_mass']=mass; arrays[stem+'_GT_weights']=weights
        arrays[stem+'_GT_ids']=gg; arrays[stem+'_condition_flat']=flat[pp]; arrays[stem+'_dual_alpha']=solved['alpha']
        rows=[]
        for k,g in enumerate(gg):
            row=dict(GT=int(g),label=int(labels[g]),actual_input_sub4=bool(tiny[g]),primary_Geo0=int(g) in primary,
                geometry_covered=int(g) not in primary,primary_Geo0_tiny=int(g) in primary and bool(tiny[g]),
                geometry_covered_tiny=int(g) not in primary and bool(tiny[g]),secondary_D0_P=int(g) in secondary,
                arms={a:metric_row(weights[k],labels[g],solved['raw'][a],solved['r'][a],solved['attention'][a]) for a in ARMS})
            row['J_minus_Icap']={key:None if row['arms']['J'][key] is None or row['arms']['I_cap'][key] is None
                else row['arms']['J'][key]-row['arms']['I_cap'][key] for key in row['arms']['J']}
            rows.append(row)
        for a in ARMS:
            arrays[stem+'_'+a+'_raw_signed']=solved['raw'][a]; arrays[stem+'_'+a+'_residual']=solved['r'][a]
            arrays[stem+'_'+a+'_null']=solved['z'][a]
            if solved['attention'][a] is not None: arrays[stem+'_'+a+'_attention']=solved['attention'][a]
        windows[arm]=dict(NA=False,cell=int(cell),xyxy=grid[cell].tolist(),complete_GT=gg.tolist(),condition_flat=flat[pp].tolist(),
            conditions=len(pp),invalid_conditions=int((~B.valid_boxes(pboxes[pp])).sum()),
            summaries=solved['summaries'],certificates=solved['certificates'],comparisons=solved['comparisons'],
            GT_rows=rows,cohorts=cohort_report(rows),null_by_condition={a:solved['z'][a].tolist() for a in ARMS})
        COUNTS['windows']+=1
    return dict(image_index=index,GT=len(gt64),actual_sub4=int(tiny.sum()),windows=windows)


def synthetic():
    def alloc(m,c): return solve_class(np.array(m,dtype=float),np.array(c,dtype=float))
    rr,aa,cc=alloc([],np.empty((0,0))); need(cc['passed'],'empty')
    rr,aa,cc=alloc([0],[[0,0]]); need(np.array_equal(rr['J']['z'],[1,1]),'zero mass null')
    rr,aa,cc=alloc([1],np.empty((1,0))); need(rr['J']['T'].shape==(1,0),'no conditions')
    rr,aa,cc=alloc([1],[[0]]); need(abs(rr['J']['T'][0,0]-.5)<ABS,'single undercapacity')
    rr,aa,cc=alloc([.2],[[0,0]]); need(abs(rr['J']['T'].sum()-.2)<ABS and rr['I']['T'].sum()>.2,'single overload')
    m=np.array([.2,1,1]); c=np.array([[0,0],[1,8],[4,2]],float)
    rr,aa,cc=alloc(m,c); need(cc['passed'] and rr['J']['T'].sum(1)[0]<=.2+ABS,'multi capacity KKT')
    ri=np.maximum(0,m-rr['I']['T'].sum(1)); rc=m-rr['I_cap']['T'].sum(1)
    need(np.max(np.abs(ri-rc))<ABS,'onepass exact')
    ss,_,_=alloc([.08,.12,1,1],np.vstack((c[0],c[0],c[1:])))
    for arm in ARMS:
        collapsed=np.vstack((ss[arm]['T'][:2].sum(0),ss[arm]['T'][2:]))
        need(np.max(np.abs(collapsed-rr[arm]['T']))<ABS*5 and np.max(np.abs(ss[arm]['z']-rr[arm]['z']))<ABS*5,'split mass invariance')
    # Distinct class slots of one query each retain unit supply; no collapse by q.
    mass=np.zeros((10,1)); mass[0,0]=mass[1,0]=1
    x=allocation(mass,np.array([[0.,0.]]),np.array([[-1.,-1,1,1],[-1.,-1,1,1]]),np.array([0,1]))
    need(x['summaries']['I']['conditions']==2 and abs(x['summaries']['I']['null_total']-1)<ABS,'multiclass units')
    pts=np.array([[0.,0],[64,64],[0,64],[64,0],[32,32]])
    mm,ww,grid=scatter(pts,np.array([0,0,0,0,0]),[0,0,64,64])
    need(abs(mm.sum()-5)<1e-14 and np.allclose(ww@grid,np.clip(pts,2,62),atol=1e-14,rtol=0),'boundary mass projection')
    box=np.array([[1.,1,4,4],[0,0,0,2]])
    cost=geometry_cost(grid,box); cost2=geometry_cost(grid/2,box/2)
    need(np.array_equal(cost,cost2) and np.isinf(cost[:,1]).all(),'scale invariance/invalid null')
    rj=np.maximum(0,m-rr['J']['T'].sum(1))
    need(np.max(rj-ri)<=ABS and np.max(rj/rj.sum()-ri/ri.sum())>1e-6,'relative-weight counterexample')
    # Same mathematical group: NA normalization must not induce unpaired means.
    rows=[]
    for jv,iv in ((None,.9),(.6,.2)):
        metric={key:iv for key in ('signed_class_raw','class_raw','spatial_raw','class_normalized','spatial_normalized')}
        jmetric={key:jv for key in metric}
        rows.append(dict(primary_Geo0=True,geometry_covered=False,primary_Geo0_tiny=True,
                         geometry_covered_tiny=False,secondary_D0_P=True,arms={'I':metric,'I_cap':metric,'J':jmetric}))
    paired=cohort_report(rows)['all']['paired_J_minus_Icap']['class_normalized']
    need(paired['n']==1 and abs(paired['mean']-.4)<1e-14,'paired NA-safe summary')
    need(normalized(np.array([1e-10]),1e-8,1.,1)==(None,'numerically_unresolved')
         and normalized(np.zeros(1),0.,0.,1)==(None,'empty_mass_null_only')
         and normalized(np.zeros(1),0.,1.,1)==(None,'exact_zero_residual'),'zero vs unresolved normalization')
    return dict(empty=True,zero_mass_null=True,no_conditions=True,single_under_capacity=True,single_over_capacity=True,
        multirow_KKT=True,onepass_equals_clip=True,equal_cost_mass_split=True,multiclass_supply=True,
        boundary_unit_scatter=True,physical_scale_cost=True,pointwise_vs_relative_counterexample=True)


def synthetic_polishing():
    # Known optimum: all costs zero, M=1, N=5, alpha*=log(N-M)=log(4).
    m=np.full(4,.25); cost=np.zeros((4,5)); exact=np.full(4,np.log(4.))
    approximate=exact.copy(); approximate[0]+=2e-7
    f0,_,_,_,_=dual_state(m,cost,exact)
    fb,gb,tb,zb,mb=dual_state(m,cost,approximate)
    relative_change=abs(fb-f0)/(1+abs(f0))
    need(relative_change<1e-13 and mb>ABS,'flat objective but failed KKT synthetic')
    original_failed=False
    try: certify(m,cost,approximate,tb,zb,fb)
    except RuntimeError: original_failed=True
    need(original_failed,'synthetic must trigger original precision certificate')
    fixed,stats=polish_dual(m,cost,approximate)
    ff,_,tf,zf,_=dual_state(m,cost,fixed)
    cert=certify(m,cost,fixed,tf,zf,ff)
    need(cert['passed'] and np.max(np.abs(fixed-exact))<1e-9,'analytic optimum after polish')
    mc=np.array([.2,1.,1.]); cc=np.array([[0.,0.],[1.,8.],[4.,2.]])
    ac,cs=polish_dual(mc,cc,np.zeros(3),coordinate_only=True)
    fc,_,tc,zc,_=dual_state(mc,cc,ac)
    need(certify(mc,cc,ac,tc,zc,fc)['passed'] and cs['coordinate_sweeps']>0,'coordinate fallback precision')
    me=np.array([.2,1.]); ce=np.array([[0.,0.],[20.,20.]])
    ae,es=polish_dual(me,ce,np.zeros(2))
    fe,_,te,ze,_=dual_state(me,ce,ae)
    need(certify(me,ce,ae,te,ze,fe)['passed'] and ae[1]==0,'inactive bound retained')
    return dict(flat_objective_original_certificate_failed=original_failed,
                relative_objective_error=relative_change,initial_scaled_KKT=mb,
                precision_polish=stats,coordinate_fallback=cs,inactive_bound=es,
                same_original_certificate_passed=True,real_failure_matrix_read=False)


def sources():
    files=B.source_manifest()
    for path,digest in INHERITED.items():
        need(sha(path)==digest,'inherited 292/293 changed'); files[str(path)]=digest
    for path,digest in SOLVER_SOURCES.items():
        need(sha(path)==digest,'solver source/binary changed'); files[str(path)]=digest
    for name,digest in ANCHORS284.items():
        path=CONTEXT/name; need(sha(path)==digest,'284 anchor'); files[str(path.resolve())]=digest
    result=B.read(CONTEXT/'01_results.json'); need(result['state']=='complete' and all(result['sources_unchanged'].values()),'284 complete')
    index=B.read(CONTEXT/'02_output_sha256.json')
    for name,digest in result['artifact_sha256'].items():
        need(index[name]==digest and sha(CONTEXT/name)==digest,'284 artifact')
        files[str((CONTEXT/name).resolve())]=digest
    files[str(PROTOCOL)]=sha(PROTOCOL); files[str(pathlib.Path(__file__).resolve())]=sha(pathlib.Path(__file__))
    return files


def aggregate(batches):
    report={}
    for letter in ('A','B'):
        images=[im for b in batches if b['letter']==letter for im in b['images']]
        if not images: continue
        arms={}
        for arm in B.WINDOWS:
            windows=[im['windows'][arm] for im in images if not im['windows'][arm]['NA']]
            allrows=[g for w in windows for g in w['GT_rows']]
            pooled=cohort_report(allrows); graphmeans={}
            for cohort in pooled:
                graphmeans[cohort]={}
                for a in ARMS:
                    graphmeans[cohort][a]={}
                    for metric in pooled[cohort]['arms'][a]:
                        vals=[w['cohorts'][cohort]['arms'][a][metric]['mean'] for w in windows
                              if w['cohorts'][cohort]['arms'][a][metric]['mean'] is not None]
                        graphmeans[cohort][a][metric]=dict(inputs=len(vals),mean=float(np.mean(vals)) if vals else None)
                graphmeans[cohort]['paired_J_minus_Icap']={}
                for metric in pooled[cohort]['paired_J_minus_Icap']:
                    vals=[w['cohorts'][cohort]['paired_J_minus_Icap'][metric]['mean'] for w in windows
                          if w['cohorts'][cohort]['paired_J_minus_Icap'][metric]['mean'] is not None]
                    graphmeans[cohort]['paired_J_minus_Icap'][metric]=dict(inputs=len(vals),mean=float(np.mean(vals)) if vals else None)
            arms[arm]=dict(windows=len(windows),pooled_GT=pooled,input_mean=graphmeans,
                totals={a:{key:float(sum(w['summaries'][a][key] for w in windows)) for key in
                    ('mass','conditions','residual_total','raw_signed_total','null_total','negative_clip_mass')} for a in ARMS},
                normalization_status={a:{status:sum(w['summaries'][a]['normalized_status']==status for w in windows)
                    for status in ('resolved','no_conditions','empty_mass_null_only','exact_zero_residual','numerically_unresolved')} for a in ARMS},
                common_scale_L1=[w['comparisons']['common_scale_L1_error'] for w in windows],
                normalized_L1=[w['comparisons']['normalized_L1_J_minus_Icap'] for w in windows])
        # Same image, two windows: average within image first; not 64 independent windows.
        graph_two={}
        for cohort in ('primary_Geo0','geometry_covered','primary_Geo0_tiny','geometry_covered_tiny'):
            graph_two[cohort]={}
            for a in ARMS:
                vals=[]
                for im in images:
                    vv=[im['windows'][arm]['cohorts'][cohort]['arms'][a]['class_normalized']['mean'] for arm in B.WINDOWS
                        if not im['windows'][arm]['NA'] and im['windows'][arm]['cohorts'][cohort]['arms'][a]['class_normalized']['mean'] is not None]
                    if vv: vals.append(float(np.mean(vv)))
                graph_two[cohort][a]=dict(inputs=len(vals),mean=float(np.mean(vals)) if vals else None)
            paired=[]
            for im in images:
                vv=[im['windows'][arm]['cohorts'][cohort]['paired_J_minus_Icap']['class_normalized']['mean'] for arm in B.WINDOWS
                    if not im['windows'][arm]['NA'] and im['windows'][arm]['cohorts'][cohort]['paired_J_minus_Icap']['class_normalized']['mean'] is not None]
                if vv: paired.append(float(np.mean(vv)))
            graph_two[cohort]['paired_J_minus_Icap']=dict(inputs=len(paired),mean=float(np.mean(paired)) if paired else None)
        report[letter]=dict(inputs=len(images),windows=arms,two_windows_mean_within_image=graph_two)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--synthetic-only',action='store_true')
    parser.add_argument('--out',type=pathlib.Path,default=ROOT/'300_BASE24_oracle_residual_transport_polished')
    args=parser.parse_args(); started=time.monotonic(); resources=H.resource_limits()
    resources['scipy']=scipy.__version__
    need(resources['priority']=='below_normal' and torch.version.cuda is None,'CPU resource contract')
    need(scipy.__version__=='1.13.1','frozen scipy version')
    for path,digest in SOLVER_SOURCES.items(): need(sha(path)==digest,'frozen solver before synthetic')
    for path,digest in INHERITED.items(): need(sha(path)==digest,'inherited sources before synthetic')
    tests=dict(original_twelve=synthetic(),precision_revision=synthetic_polishing())
    if args.synthetic_only: print(json.dumps(dict(tests=tests,counts=COUNTS,resources=resources))); return
    synthetic_solves=COUNTS['convex_solves']; COUNTS['convex_solves']=0
    out=args.out.resolve(); need(out.parent==ROOT and out.name.startswith('300_') and not out.exists(),'new300 only'); out.mkdir()
    stage='sources'; files={}; batches=[]; artifacts={}
    try:
        files=sources(); _,_,tv=R.load_post_api()
        save(out/'00_sources_and_synthetic.json',dict(source_sha256=files,synthetic=tests,synthetic_solves=synthetic_solves,
             resources=resources,parameters=dict(grid=GRID,epsilon=EPSILON,kappa=KAPPA,ABS=ABS,ACTIVE=ACTIVE,
                 POLISH_TOL=POLISH_TOL,MAX_NEWTON=MAX_NEWTON,MAX_BACKTRACK=MAX_BACKTRACK,
                 MAX_COORD_SWEEPS=MAX_COORD_SWEEPS,MAX_BRACKET=MAX_BRACKET,COORD_BISECTIONS=COORD_BISECTIONS)))
        for letter,seed in (('A',1),('B',2)):
            for batch in range(8):
                P.M.guard(); stage=f'{letter}_batch{batch:02d}'
                old=B.read(ROOT/'263_BASE24_residual_window_probe'/(stage+'.json')); ctx=B.read(CONTEXT/(stage+'.json'))
                for key in ('letter','seed','batch','shape','input_sha256','prediction_sha256','source_ids'):
                    need(old[key]==ctx[key],'source identity '+key)
                need(old['letter']==letter and old['seed']==seed and old['batch']==batch,'batch order')
                need([x['image_index'] for x in old['images']]==[x['image_index'] for x in ctx['images']]==list(range(4)),'image order')
                hw=tuple(old['shape'][2:]); arrays={}
                with np.load(ROOT/'244_BASE24_cached_displacement_floor_probe_fixed'/(stage+'_arrays.npz'),allow_pickle=False) as ar:
                    images=[probe_image(o,c,ar,hw,tv,arrays) for o,c in zip(old['images'],ctx['images'])]
                report=dict(letter=letter,seed=seed,batch=batch,shape=old['shape'],input_sha256=old['input_sha256'],
                    prediction_sha256=old['prediction_sha256'],source_ids=old['source_ids'],images=images)
                save(out/(stage+'.json'),report); np.savez(out/(stage+'_arrays.npz'),**arrays)
                for name in (stage+'.json',stage+'_arrays.npz'): artifacts[name]=sha(out/name)
                batches.append(report); COUNTS['npz_batches']+=1; COUNTS['images']+=len(images)
                del arrays,old,ctx; gc.collect(); print(stage,'complete',flush=True)
        need(COUNTS['npz_batches']==16 and COUNTS['images']==64,'exact development scope')
        unchanged={p:sha(pathlib.Path(p))==digest for p,digest in files.items()}; need(all(unchanged.values()),'source changed')
        need(all(v==0 for k,v in COUNTS.items() if k not in ('npz_batches','images','windows','convex_solves')),'zero model/PT/heldout')
        save(out/'01_results.json',dict(state='complete',summaries=aggregate(batches),counts=COUNTS,resources=resources,
            source_unchanged=unchanged,artifact_sha256=artifacts,seconds=time.monotonic()-started,
            oracle_GT_mass=True,geometry_only_cost=True,AP=False,learnability_proven=False,
            transport_necessity_proven=False,training_authorized=False,numerical_gate_only=True))
        save(out/'02_output_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    except BaseException:
        save(out/'FAILED.json',dict(state='implementation_failure_no_method_conclusion',stage=stage,
            counts=COUNTS,traceback=traceback.format_exc(),seconds=time.monotonic()-started)); raise


if __name__=='__main__': main()
