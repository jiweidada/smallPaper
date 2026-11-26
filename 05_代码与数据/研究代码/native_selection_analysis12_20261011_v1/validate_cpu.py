"""Arithmetic validation including tied native indices and displaced selected boxes."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
from selection_core import validate_record,neighborhood,image_description,verify_grid_centers,SHAPES

HERE=Path(__file__).resolve().parent
def record():
    logits=np.full((8400,10),-10,np.float32);logits[:300,0]=1
    anchors=np.zeros((8400,4),np.float32)
    ids=np.arange(300,dtype=np.int64)
    return dict(dense_class_logits=logits,dense_anchor_logits=anchors,selected_indices=ids,
        selected_anchor_logits=anchors[ids].copy(),selected_bbox_offsets=np.zeros((300,4),np.float32),
        original_hw=np.array([640,640]),input_hw=np.array([640,640]),image_id=np.array(7))
def reject(operation):
    try:operation()
    except AssertionError:return True
    raise AssertionError('Expected malformed record rejection')
def main(output):
    r=record();original={k:v.copy() for k,v in r.items()}
    validate_record(r)
    assert [len(a) for a in neighborhood(.5,.5,0)]==[1,1,1,3]
    assert [len(a) for a in neighborhood(0,0,1)]==[4,4,4,12]
    assert [len(a) for a in neighborhood(1,1,1)]==[4,4,4,12]
    assert [len(a) for a in neighborhood(.5,.5,1)]==[9,9,9,27]
    assert neighborhood(1/80,1/80,0)[0].tolist()==[81]
    target=[dict(id=1,image_id=7,category_id=0,bbox=[318,318,4,4])]
    iou=np.zeros((1,300));iou[0,0]=1
    result,audit=image_description(r,target,iou)
    assert not result['selected_neighborhood_sites'].any()
    assert result['initial_any_support'].all() and not result['selected_local_initial_support_count'].any()
    # Nonlocal selected location may regress a correct box: neighborhood
    # absence is explicitly not a proof that geometry has no coverage.
    assert all(np.array_equal(r[k],original[k]) for k in r)
    tied=record();tied['dense_class_logits'][300:600,0]=1
    tied['selected_indices']=np.arange(300,600,dtype=np.int64)
    tied['selected_anchor_logits']=tied['dense_anchor_logits'][tied['selected_indices']].copy()
    tr,ta=image_description(tied,[dict(id=2,image_id=7,category_id=0,bbox=[483.5,27.5,1,1])],np.zeros((1,300)))
    assert ta['native_cutoff_tied_sites']==600
    assert tr['best_max_class_rank_min'][0,0,0]==1 and tr['best_max_class_rank_max'][0,0,0]==600
    assert tr['selected_neighborhood_sites'][0,0,0]==1
    bad=record();bad['selected_indices'][0]=301
    reject(lambda:validate_record(bad))
    bad=record();bad['selected_indices'][0]=1
    reject(lambda:validate_record(bad))
    invalid=record();sites=np.concatenate(neighborhood(.5,.5,1)[:3])
    invalid['dense_anchor_logits'][sites]=np.inf
    invalid['selected_anchor_logits']=invalid['dense_anchor_logits'][invalid['selected_indices']].copy()
    v,va=image_description(invalid,target,np.zeros((1,300)))
    assert not v['valid_neighborhood_sites'].any()
    assert np.isnan(v['best_max_class_margin_to_cutoff']).all()
    er,ea=image_description(record(),[],np.zeros((0,300)))
    assert er['selected_neighborhood_sites'].shape==(0,2,4)
    grids=[]
    for height,width in SHAPES:
        row,col=np.meshgrid(np.arange(height),np.arange(width),indexing='ij')
        grids.append(np.stack([(col+.5)/width,(row+.5)/height],-1).reshape(-1,2))
    centers=np.concatenate(grids);grid_record=record()
    for dtype in (np.float32,np.float16):
        raw=np.concatenate([np.log(centers/(1-centers)),np.zeros((8400,2))],1).astype(dtype)
        grid_record['dense_anchor_logits']=raw
        assert verify_grid_centers(grid_record)<(.001 if dtype==np.float16 else 1e-6)
    bad_grid=record();reject(lambda:verify_grid_centers(bad_grid))
    for name in ('selection_core.py','analyze.py','validate_cpu.py'):
        compile((HERE/name).read_text(encoding='utf8'),str(HERE/name),'exec')
    p=HERE/'protocol.json'
    protocol=json.loads(p.read_bytes())
    for name,h in protocol['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    validation=dict(status='PASS_ACTUAL_INDEX_TIES_EDGES_INVALID_PRIORS_AND_NONLOCAL_GEOMETRY_CPU_FIXTURES',
        native_cutoff_tied_sites_fixture=600,actual_tied_ids_preserved=True,inputs_unchanged=True,
        missing_nearby_selection_is_not_missing_geometry_proof=True,empty_GT_supported=True,
        fixed12_records_analyzed=0,model_forwards=0,evaluator_calls=0,optimizer_updates=0,
        verification_scope='NumPy arithmetic fixtures and source syntax. Actual four fixed12 diagnostics remain unavailable.',
        code_sha256=protocol['files'])
    assert 'torch' not in sys.modules
    with Path(output).open('x',encoding='utf8') as f:json.dump(validation,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(validation['status'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)

