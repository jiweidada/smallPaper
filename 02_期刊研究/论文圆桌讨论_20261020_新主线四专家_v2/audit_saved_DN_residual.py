"""Cheap falsification of the roundtable's DN residual transport proposal.

Only existing development predictions. No learned intervention or model call.
DN slot is selected by initial error only, never final outcome.
"""
import hashlib,json,tarfile
import numpy as np
from audit_saved_bin_response import load_arrays,DATA,OUT,ROOT
from audit_saved_support import xyxy,iou,offsets

def main():
    spec={'status':'exploratory developmental falsification, not validation or AP',
          'proposal':'initial-state comparable same-GT DN normalized correction transported to ordinary pre-reference',
          'selection':'same GT positive DN slot minimizing L-infinity distance of GT-minus-reference edge errors divided by reference width/height; ties first slot',
          'distance_prefixes':[.1,.25,.5,1.,2.,None],
          'transport':'ordinary_ref_edges + ordinary_ref_wh/DN_ref_wh * (DN_final_edges-DN_ref_edges)',
          'scope':'all ordinary final native union exposures, report capacity and repeated GT exposures separately',
          'limitations':['training-only GT linkage, not deployable inference','static virtual target quality, not benefit of learning it','no annotation-ID for original GT independence','development only; no heldout access'],
          'new_forward':0,'new_backward':0,'optimizer_updates':0,
          'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    (OUT/'21_DN修正量运输探索口径.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
    result={'spec':spec,'sources':{},'summaries':[]}
    for account,seed in [('A',1),('B',2)]:
        rows=[]
        with tarfile.open(DATA/f'{account}_frozen_predictions.tar.gz') as tf,tarfile.open(DATA/f'{account}_development.tar.gz') as ta:
            seal=json.load(tf.extractfile('development/seal.json'))
            for item in seal['files']:
                blob=tf.extractfile('development/'+item['relative']).read(); digest=hashlib.sha256(blob).hexdigest();assert digest==item['sha256']
                an=json.load(ta.extractfile(item['relative'].replace('_predictions.pt','_analysis.json')));assert an['prediction_sha256']==digest
                result['sources'][account+'/'+item['relative']]=digest
                d=load_arrays(blob);o=d['outputs'];dn=o['dn_outputs'][-1];ng=int(o['dn_meta']['dn_num_group'])
                np.testing.assert_array_equal(o['ref_points'],o['pre_outputs']['pred_boxes'])
                np.testing.assert_array_equal(dn['ref_points'],o['dn_pre_outputs']['pred_boxes'])
                for bi,r in enumerate(an['inputs']):
                    gtboxes=d['targets'][bi]['boxes'].astype(float);n=len(gtboxes)
                    pos=o['dn_meta']['dn_positive_idx'][bi];assert len(pos)==ng*n
                    gtrow=np.tile(np.arange(n),ng)
                    branch=r['branches']['final']['native_union']
                    for k,(q,g) in enumerate(zip(branch['pair_query'],branch['pair_GT'])):
                        gt=xyxy(gtboxes[g]);ro=o['ref_points'][bi,q].astype(float);reo=xyxy(ro);so=np.tile(ro[2:],2)
                        eo=(gt-reo)/so
                        cand=pos[gtrow==g];rd=dn['ref_points'][bi,cand].astype(float);red=xyxy(rd);sd=np.tile(rd[:,2:],(1,2))
                        ed=(gt-red)/sd;dist=np.max(np.abs(ed-eo),axis=-1);pick=int(np.argmin(dist));j=int(cand[pick])
                        teacher=xyxy(dn['pred_boxes'][bi,j].astype(float));target=reo+so/sd[pick]*(teacher-red[pick])
                        assert np.max(np.abs(offsets(ro,target)))<=4+1e-5
                        original=xyxy(o['pred_boxes'][bi,q].astype(float));original_iou=iou(original,gt)
                        assert abs(original_iou-branch['pair_iou'][k])<1e-4
                        rows.append({'GT_exposure':(item['batch'],bi,g),'initial_distance':float(dist[pick]),'capacity':bool(r['capacity_eligible']),
                                     'original_iou':original_iou,'transported_iou':iou(target,gt),'direct_DN_iou':iou(teacher,gt),
                                     'pre_iou':iou(reo,gt),'DN_pre_iou':iou(red[pick],gt),'teacher_slot':j})
        for capacity in ['ALL','eligible_only']:
            for cutoff in spec['distance_prefixes']:
                rs=[r for r in rows if (capacity=='ALL' or r['capacity']) and (cutoff is None or r['initial_distance']<=cutoff)]
                if not rs:continue
                summary={'seed':seed,'capacity':capacity,'initial_distance_max':cutoff,'pair_exposures':len(rs),
                         'unique_augmented_GT_exposures':len({r['GT_exposure'] for r in rs}),
                         'distance_q10_q50_q90':np.quantile([r['initial_distance'] for r in rs],[.1,.5,.9]).tolist(),
                         **{key:float(np.mean([r[key] for r in rs])) for key in ['original_iou','transported_iou','direct_DN_iou','pre_iou','DN_pre_iou']}}
                for t in [.5,.75]:
                    summary[f'original_ge_{t}']=sum(r['original_iou']>=t for r in rs)
                    for label in ['transported_iou','direct_DN_iou']:
                        summary[f'{label}_gained_{t}']=sum(r['original_iou']<t<=r[label] for r in rs)
                        summary[f'{label}_lost_{t}']=sum(r[label]<t<=r['original_iou'] for r in rs)
                result['summaries'].append(summary)
    (OUT/'22_CPU开发组DN修正量运输.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([r for r in result['summaries'] if r['capacity']=='ALL'],ensure_ascii=False))

if __name__=='__main__':main()
