"""Same-forward cache audit of existing ordinary positive DDF, not a new method.

Output-coordinate derivatives only, not network-parameter interference or AP.
"""
import datetime,gc,hashlib,json,os,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/source_allocation_frozen_20261020_v1'))
import diagnostic as d
torch,np=d.torch,d.np
DATA=WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/isolated_v3'
OUT=ROOT/'ddf_geometry_cache_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):d.native.save_json(p,v)
def project(up,scale):
    step=(up[0].abs()*scale.abs()+1)**(2/30)
    return torch.cat([-up[0].abs()*scale.abs()*2,*[1-step**i for i in range(15,0,-1)],torch.zeros_like(up[0][None]),*[step**i-1 for i in range(1,16)],up[0].abs()*scale.abs()*2],0)
def decode(z,ref,bins,scale):
    means=(z.softmax(-1)*bins).sum(-1)
    return torch.cat([ref[:,:2]+(means[:,2:]-means[:,:2])*ref[:,2:]/(2*scale),
                      ref[:,2:]*(1+(means[:,:2]+means[:,2:])/scale)],-1)
def aligned_iou(a,b):return d.box_iou(d.box_cxcywh_to_xyxy(a),d.box_cxcywh_to_xyxy(b))[0].diagonal()
def subset_row(teacher_relation,weight,dot1,dot5,cos1,cos5,drift):
    active=weight>0;eps=1e-12
    return dict(pairs=len(weight),active_pairs=int(active.sum()),teacher_better=int((teacher_relation>1e-6).sum()),teacher_worse=int((teacher_relation < -1e-6).sum()),teacher_tied=int((teacher_relation.abs()<=1e-6).sum()),
        student_IoU_weight_sum=float(weight.sum()),
        T1_opposing=int((active&(dot1 < -eps)).sum()),T5_opposing=int((active&(dot5 < -eps)).sum()),
        T1_aligned=int((active&(dot1>eps)).sum()),T5_aligned=int((active&(dot5>eps)).sum()),
        T1_aligned_T5_opposing=int((active&(dot1>eps)&(dot5 < -eps)).sum()),
        T1_opposing_T5_aligned=int((active&(dot1 < -eps)&(dot5>eps)).sum()),
        cosine_T1_median=float(cos1[active].median()) if active.any() else None,
        cosine_T5_median=float(cos5[active].median()) if active.any() else None,
        student_softened_mean_edge_drift_pixels_median=float(drift.median()) if len(drift) else None)
def main():
    limits=d.native.resource_limits();assert limits['available_RAM_GiB']>8
    # Analytic softened-KL derivative checked independently before real data.
    torch.manual_seed(20102026)
    ss=torch.randn(3,4,33,dtype=torch.float64,requires_grad=True);tt=torch.randn_like(ss)
    for temp in [1.,5.]:
        loss=(torch.nn.functional.kl_div((ss/temp).log_softmax(-1),(tt/temp).softmax(-1),reduction='none').sum(-1)*temp**2).sum()
        grad=torch.autograd.grad(loss,ss)[0]
        assert float((grad-temp*((ss/temp).softmax(-1)-(tt/temp).softmax(-1))).abs().max())<1e-14
    OUT.mkdir(exist_ok=False)
    protocol=dict(utc=d.now(),scope='BASE24online GPU co-saved development00-07; ordinary native union pairs at aux0/aux1, fixed same teacher final; not DN, network parameter gradients or heldout',
        source='Same-forward cached corners, refs, teacher_corners/teacher_logits, boxes and targets with original native GPU analysis SHA; never use CPUdata10 indices',
        original='Native DDF T=5, positive weights are detached student IoU. Negatives use teacher maxscore but are outside this matched-GT geometry audit. Teacher is final layer. Verify co-saved teacher corners.',
        controls='Only original T5 versus canonical unsoftened T1 with identical original positive weights and normalization. No temperature search, learned head or posthoc sign gate.',
        derivative='g_KL=T*(softmax(student/T)-softmax(teacher/T))*studentIoU*original_positive_normalization. Compare dot/cos with gradient of original5L1+2GIoU w.r.t same studentcorner logits; negative dot implies adverse first-order output-logit geometry direction under direct -g_KL, not network-update harm.',
        normalization='Native positive coefficient sqrt(4K*8/B)/(sqrt(4K*8/B)+sqrt(4(BQ-K)*8/B))/(4K); geometry uses original num_boxes_go=K. Omit common criterionDDFscalar in directional signs, do not compare total loss magnitudes.',
        groups='All ordinary union pairs; teacherbetter/worse/tied by IoUgap1e-6; shortside<4 andothers. All groups reported. Numerical dot sign tolerance1e-12. Counts are repeated pair exposures, not unique independentGT.',
        decision='Screen an unexplored existing term only. Temperature drift, adverse output gradients, or teacherworse frequency alone are not novelty/harm/training evidence. If no consistent distinct pattern from T1, do not prioritize a temperature mechanism. No new detector training.',
        limits=limits,model_forwards=0,training_steps=0,script_sha256=sha(__file__),synthetic_KL_derivative_pass=True)
    save(OUT/'01_protocol.json',protocol)
    rows=[];sources=[];reconstruction_error=0.
    for side,seed in [('A',1),('B',2)]:
        for batch in range(8):
            pp=DATA/side/f'frozen_predictions/development/batch{batch:02d}_predictions.pt';ap=DATA/side/f'development/batch{batch:02d}_analysis.json'
            analysis=json.loads(ap.read_text(encoding='utf8'));fingerprint=sha(pp);assert fingerprint==analysis['prediction_sha256']
            sources.append(dict(seed=seed,batch=batch,prediction_sha256=fingerprint,analysis_sha256=sha(ap)))
            v=torch.load(pp,map_location='cpu',weights_only=False);outputs=v['outputs'];bins=project(outputs['up'],outputs['reg_scale']).double();scale=float(outputs['reg_scale'])
            qs=[];bs=[];gs=[]
            for image,item in enumerate(analysis['inputs']):
                u=item['branches']['final']['native_union'];qs.extend(u['pair_query']);gs.extend(u['pair_GT']);bs.extend([image]*len(u['pair_query']))
            bs=torch.tensor(bs,dtype=torch.long);qs=torch.tensor(qs,dtype=torch.long);gs=torch.tensor(gs,dtype=torch.long)
            gt=torch.stack([v['targets'][i]['boxes'][g] for i,g in zip(bs.tolist(),gs.tolist())]).double();K=len(qs);B,Q=outputs['pred_boxes'].shape[:2]
            assert K==sum(len(x['branches']['final']['native_union']['pair_query']) for x in analysis['inputs'])
            norm=(4*K*8/B)**.5/((4*K*8/B)**.5+(4*(B*Q-K)*8/B)**.5)/(4*K)
            for layer,student in enumerate(outputs['aux_outputs']):
                assert torch.equal(student['teacher_corners'],outputs['pred_corners'])
                ref=student['ref_points'][bs,qs].double();z=student['pred_corners'][bs,qs].double().reshape(K,4,33).requires_grad_();teacher=student['teacher_corners'][bs,qs].double().reshape(K,4,33)
                pred=decode(z,ref,bins,scale);old=student['pred_boxes'][bs,qs].double()
                reconstruction_error=max(reconstruction_error,float((pred-old).abs().max()))
                assert reconstruction_error<1e-5
                weight=aligned_iou(old,gt).detach();teacher_iou=aligned_iou(outputs['pred_boxes'][bs,qs].double(),gt);relation=teacher_iou-weight
                geometry=(5*(pred-gt).abs().sum()+2*(1-d.generalized_box_iou(d.box_cxcywh_to_xyxy(pred),d.box_cxcywh_to_xyxy(gt)).diagonal()).sum())/K
                gg=torch.autograd.grad(geometry,z)[0].detach()
                dots={};cos={}
                with torch.no_grad():
                    for temp in [1,5]:
                        kd=temp*((z/temp).softmax(-1)-(teacher/temp).softmax(-1))*weight[:,None,None]*norm
                        dot=(gg*kd).sum((1,2));den=gg.flatten(1).norm(dim=1)*kd.flatten(1).norm(dim=1)
                        dots[temp]=dot;cos[temp]=torch.where(den>0,dot/den,torch.zeros_like(dot))
                    moment1=(z.softmax(-1)*bins).sum(-1);moment5=((z/5).softmax(-1)*bins).sum(-1)
                    drift=((moment5-moment1).abs()*ref[:,[2,3,2,3]]/scale*640).mean(-1)
                    tiny=gt[:,2:].min(-1).values*640<4
                    groups={'all':torch.ones(K,dtype=torch.bool),'teacher_better':relation>1e-6,'teacher_worse':relation < -1e-6,'teacher_tied':relation.abs()<=1e-6,'shortside_lt4':tiny,'shortside_ge4':~tiny}
                    row=dict(seed=seed,batch=batch,layer=layer,normalization=norm,groups={k:subset_row(relation[m],weight[m],dots[1][m],dots[5][m],cos[1][m],cos[5][m],drift[m]) for k,m in groups.items()})
                rows.append(row)
            del v,outputs,student,z,pred,geometry,gg;gc.collect()
        print(json.dumps(dict(seed=seed,rows=sum(x['seed']==seed for x in rows),reconstruction_error=reconstruction_error)),flush=True)
    totals=[]
    countkeys=['pairs','active_pairs','teacher_better','teacher_worse','teacher_tied','student_IoU_weight_sum','T1_opposing','T5_opposing','T1_aligned','T5_aligned','T1_aligned_T5_opposing','T1_opposing_T5_aligned']
    for seed in [1,2]:
        totals.append(dict(seed=seed,groups={g:{k:sum(x['groups'][g][k] for x in rows if x['seed']==seed) for k in countkeys} for g in rows[0]['groups']}))
    save(OUT/'02_results.json',dict(protocol=protocol,rows=rows,pooled=totals,sources=sources,box_reconstruction_max_error=reconstruction_error,novel_method_established=False))
    print(json.dumps(totals,ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save(OUT/('failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),blind_retry=False))
        raise
