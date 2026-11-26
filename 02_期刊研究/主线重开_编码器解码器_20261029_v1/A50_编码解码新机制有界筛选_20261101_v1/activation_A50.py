from pathlib import Path
import hashlib,json,os,time,traceback
import numpy as np
import torch
from runtime_A50 import ROOT,native,validate_files
from edge_shape_matching import physical_support,expected_error,native_cost,EdgeShapeMatcher


def main():
    plan=validate_files('activation_protocol_FROZEN_v1.json')
    with (ROOT/'activation_once.lock').open('x') as f:f.write(str(os.getpid()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    start=time.monotonic()
    train=native.VisDroneDataset('train')
    order=sorted(range(len(train)),key=lambda i:hashlib.sha256(('A50-shape:'+train.rows[i]['file']).encode()).hexdigest())[:64]
    model,criterion,_=native.create_components('B')
    checkpoint=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    assert checkpoint['arm']=='B' and checkpoint['epoch']==12
    model.load_state_dict(checkpoint['ema']['module'],strict=True)
    model.requires_grad_(False);model.eval();model.cuda()
    criterion=criterion.cuda();criterion.eval()
    decoder=model.decoder
    reg_max=decoder.reg_max;up=decoder.up.detach().item();scale=decoder.reg_scale.detach().item()
    captured={}
    def hook(module,args,output):
        captured.update(pred_boxes=output[0][-1],pred_logits=output[1][-1],
                        pred_corners=output[2][-1],ref_points=output[3][-1])
    handle=decoder.decoder.register_forward_hook(hook)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
    m=EdgeShapeMatcher(criterion.matcher,reg_max,up,scale,'M')
    q=EdgeShapeMatcher(criterion.matcher,reg_max,up,scale,'Q')
    from engine.edgecrafter.box_ops import box_cxcywh_to_xyxy,box_iou
    totals={'GT':0,'small_GT':0,'M_vs_B_changed':0,'Q_vs_B_changed':0,'Q_vs_M_changed':0,
            'small_M_vs_B_changed':0,'small_Q_vs_B_changed':0,'small_Q_vs_M_changed':0}
    rows=[];shape_errors=[];max_decode=0.;base_identity=True
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for offset in range(0,64,4):
            images,targets=native.collate([train[i] for i in order[offset:offset+4]])
            images=images.cuda();targets=native.targets_to_device(targets,'cuda')
            captured.clear();outputs=model(images)
            assert torch.equal(outputs['pred_boxes'],captured['pred_boxes'])
            assert torch.equal(outputs['pred_logits'],captured['pred_logits'])
            prob,support=physical_support(captured,reg_max,up,scale)
            decoded=(prob*support).sum(-1)
            discrepancy=(decoded-box_cxcywh_to_xyxy(outputs['pred_boxes'])).abs().max().item()
            max_decode=max(max_decode,discrepancy)
            assert discrepancy<=2e-6, ('native_affine_decode_identity',discrepancy)
            bidx=criterion.matcher(outputs,targets)['indices'];midx=m(captured,targets)['indices'];qidx=q(captured,targets)['indices']
            for j,target in enumerate(targets):
                g=len(target['boxes'])
                gt=box_cxcywh_to_xyxy(target['boxes'])
                direct=native_cost(criterion.matcher,{k:captured[k][j] for k in ('pred_logits','pred_boxes')},target)
                from scipy.optimize import linear_sum_assignment
                ii,jj=linear_sum_assignment(direct.cpu().numpy())
                original={int(t):int(i) for i,t in zip(*bidx[j])}
                assert original=={int(t):int(i) for i,t in zip(ii,jj)}
                owner=[]
                for pairs in (bidx[j],midx[j],qidx[j]):
                    current=torch.full((g,),-1,dtype=torch.long)
                    current[pairs[1]]=pairs[0]
                    owner.append(current)
                sz=target['orig_size'].float()
                small=(target['boxes'][:,2]*sz[1]*target['boxes'][:,3]*sz[0]<32**2).cpu()
                diff={'M_vs_B':owner[1]!=owner[0],'Q_vs_B':owner[2]!=owner[0],'Q_vs_M':owner[2]!=owner[1]}
                dm=expected_error(prob[j],support[j],gt,'M')
                dq=expected_error(prob[j],support[j],gt,'Q')
                bounded_gap=((-torch.expm1(-dq))-(-torch.expm1(-dm))).abs()
                owned=owner[0]>=0
                shape_errors.extend(bounded_gap[owner[0][owned].cuda(),torch.arange(g,device='cuda')[owned.cuda()]].cpu().tolist())
                totals['GT']+=g;totals['small_GT']+=int(small.sum())
                row={'file':train.rows[order[offset+j]]['file'],'GT':g,'small_GT':int(small.sum()),
                     'B_owners':owner[0].tolist(),'M_owners':owner[1].tolist(),'Q_owners':owner[2].tolist()}
                for key,d in diff.items():
                    totals[key+'_changed']+=int(d.sum());totals['small_'+key+'_changed']+=int((d&small).sum())
                    row[key+'_changed']=int(d.sum());row['small_'+key+'_changed']=int((d&small).sum())
                rows.append(row)
            native.save_json(ROOT/'activation_progress.json',{'status':'RUNNING','images':offset+4,
                 'whole_model_forwards':native.MODEL_COUNTER['forwards'],'optimizer_updates':0})
    handle.remove()
    after=model.state_dict()
    assert all(torch.equal(v,after[k].cpu()) for k,v in before.items())
    assert native.MODEL_COUNTER['forwards']==16
    fraction=totals['small_Q_vs_M_changed']/totals['small_GT']
    active=totals['small_Q_vs_M_changed']>=20 and fraction>=.01
    result={'status':'PASS_ACTIVATION_ONLY' if active else 'STOP_INSUFFICIENT_DISTINCT_ASSIGNMENT',
        'interpretation':'No optimizer updates; not an AP benefit or causal attribution test.',
        'protocol_sha256':native.digest(ROOT/'activation_protocol_FROZEN_v1.json'),
        'checkpoint_sha256':native.digest(Path(plan['checkpoint'])),'totals':totals,
        'small_Q_vs_M_owner_change_fraction':fraction,'native_decode_max_absolute_error':max_decode,
        'native_base_assignment_identity':base_identity,'BN_buffers_unchanged':True,
        'bounded_Q_vs_M_cost_gap_on_native_matched_pairs_percentiles':dict(zip(['p0','p25','p50','p75','p95','p100'],
                    np.percentile(shape_errors,[0,25,50,75,95,100]).tolist())),
        'whole_model_forwards':16,'optimizer_updates':0,'model_parameters':sum(p.numel() for p in model.parameters()),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-start,'rows':rows}
    native.save_json(ROOT/'activation_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        native.save_json(ROOT/'activation_result.json',{'status':'FAILED_STOPPED_NO_AUTORETRY',
          'error':repr(e),'traceback':traceback.format_exc(),'whole_model_forwards':native.MODEL_COUNTER['forwards'],
          'optimizer_updates':0})
        raise
