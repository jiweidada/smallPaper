"""Check range counts by direct edge inequalities and finite-logit constructions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(protocol_path,analysis,output):
    protocol_path,analysis,output=map(Path,(protocol_path,analysis,output))
    assert not output.exists()
    protocol=read(protocol_path);report=read(analysis/'report.json')
    assert report['protocol_sha256']==sha(protocol_path)
    manifest=read(analysis/'manifest.json')
    for name,h in manifest['files'].items(): assert sha(analysis/name)==h
    weights=np.array(report['weights']);assert len(weights)==33 and np.array_equal(weights,-weights[::-1])
    checks={}
    for seed,spec in protocol['inputs'].items():
        packet=Path(spec['packet']);ann=read(packet/'reference/val_annotations.json')
        images={int(i['id']):i for i in ann['images']};groups={k:[] for k in images}
        for a in ann['annotations']:groups[int(a['image_id'])].append(a)
        with np.load(analysis/f'seed{seed}_per_GT.npz',allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
        index={int(g):i for i,g in enumerate(data['annotation_id'])}
        assert len(index)==38759 and set(index)=={int(a['id']) for a in ann['annotations']}
        maximum=0.;constructed=0;checked=0;support=0
        for iid,im in sorted(images.items()):
            p=packet/f'baseline_eval/stages_epoch12/image_{iid:06d}.npz'
            assert sha(p)==report['bindings'][seed]['stage_files'][p.name]
            with np.load(p,allow_pickle=False) as z: ref=z['pre_after_first_layer_boxes'].astype(np.float64)
            entries=sorted(groups[iid],key=lambda a:int(a['id']));r=np.array([index[int(a['id'])] for a in entries])
            gt=np.array([a['bbox'] for a in entries],dtype=np.float64)/np.array([im['width'],im['height'],im['width'],im['height']])
            lower=gt[:,:2];upper=lower+gt[:,2:]
            center=ref[:,:2];size=ref[:,2:]
            # Direct coordinate intervals derived from d in[-4,4], scale4.
            ok=((lower[:,None,:]>=center[None,:,:]-1.5*size[None,:,:]) &
                (lower[:,None,:]<=center[None,:,:]+.5*size[None,:,:]) &
                (upper[:,None,:]>=center[None,:,:]-.5*size[None,:,:]) &
                (upper[:,None,:]<=center[None,:,:]+1.5*size[None,:,:])).all(-1)
            assert np.array_equal(ok.any(1),data['any_closed_range'][r])
            assert np.array_equal(ok.sum(1),data['reachable_query_count'][r])
            best=data['best_final_query'][r].astype(int)
            assert np.array_equal(ok[np.arange(len(r)),best],data['best_final_query_in_range'][r])
            distances=np.concatenate((4*(center[None,:,:]-lower[:,None,:])/size[None,:,:]-2,
                                      4*(upper[:,None,:]-center[None,:,:])/size[None,:,:]-2),axis=-1)
            magnitudes=np.max(np.abs(distances),axis=-1);chosen=magnitudes.argmin(1)
            assert np.allclose(magnitudes.min(1),data['minimum_max_abs_distance'][r],atol=1e-12,rtol=0)
            strict=magnitudes.min(1)<4-protocol['strict_interior_margin']
            assert np.array_equal(strict,data['any_strict_interior'][r])
            d=distances[np.arange(len(r)),chosen][strict]
            # Positive uniform mass at every bin plus endpoint masses gives finite logits.
            uniform=1e-10;prob=np.full((*d.shape,33),uniform/33)
            left=(4-d/(1-uniform))/8;right=1-left
            prob[...,0]+=(1-uniform)*left;prob[...,-1]+=(1-uniform)*right
            assert (prob>0).all() and np.isfinite(np.log(prob)).all()
            logits=np.log(prob);exponent=np.exp(logits-logits.max(-1,keepdims=True))
            recovered=(exponent/exponent.sum(-1,keepdims=True)*weights).sum(-1)
            error=float(np.max(np.abs(recovered-d))) if len(d) else 0.
            assert error<1e-12 and np.allclose(prob.sum(-1),1.,atol=1e-14,rtol=0)
            maximum=max(maximum,error);constructed+=int(strict.sum());checked+=len(r);support+=int(ok.any(1).sum())
        assert checked==38759
        checks[seed]=dict(GT_checked=checked,coordinate_inequality_reachable=support,
            finite_softmax_four_edge_constructions=constructed,max_abs_offset_roundtrip_error=maximum,
            coordinate_range_counts_and_final_best_query_flags_exact=True)
    result=dict(status='PASS_INDEPENDENT_FDR_RANGE_ORACLE_CHECK',checks=checks,protocol_sha256=sha(protocol_path),
        report_sha256=sha(analysis/'report.json'),verification_script_sha256=sha(__file__),
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
        material_passport=dict(dataset='VisDrone frozen val548/38759GT',weights='Existing same-seed BASE12 EMA',
            decoder='Native D-FINE3 layers/300 queries/reg_max32/reg_scale4/up0.5',
            inputs='Saved first-layer pre_bboxes plus original GT',scope=protocol['scope']),
        statistical_fallacy_checks=dict(total=11,checked=11,notes=[
            'No AP proxy','No efficacy or significance test','No best-epoch choice','Both bound seeds retained',
            'No query identity across runs','No oracle deployment claim','No root-cause claim',
            'Overlapping scale groups not summed','No independence assumption across GT/images',
            'No one-to-one matching or class/top100 constraints inferred','Positive finite logits prove algebraic reach only']),
        interpretation='Most under4-pixel failures are inside the native FDR range; expanding range or bins alone has no hard-range rationale. Learned optimization and information availability remain untested mediators.')
    assert 'torch' not in sys.modules
    with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('protocol','analysis','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(verify(a.protocol,a.analysis,a.output)['checks']),flush=True)
