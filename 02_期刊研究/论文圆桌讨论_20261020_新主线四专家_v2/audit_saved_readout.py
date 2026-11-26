"""Diagnostic readout substitutions of archived BASE24 distributions.

Fixed native union identities; not detector inference, recall, or AP.
"""
import hashlib,json,tarfile
import numpy as np
from audit_saved_bin_response import load_arrays,DATA,OUT,ROOT
from audit_saved_support import xyxy,ceiling,iou

def separated_peaks(p,bins):
    peaks=(p>np.concatenate((np.full((*p.shape[:-1],1),-1.),p[...,:-1]),-1)) & (p>=np.concatenate((p[...,1:],np.full((*p.shape[:-1],1),-1.)),-1))
    result=[]
    for v,peak in zip(p.reshape(-1,len(bins)),peaks.reshape(-1,len(bins))):
        i=int(np.argmax(v));other=np.flatnonzero(peak & (np.abs(bins-bins[i])>=.5) & (v>=.5*v[i]))
        result.append(any(np.min(v[min(i,j):max(i,j)+1])<=.5*min(v[i],v[j]) for j in other))
    return np.array(result).reshape(p.shape[:-1])

def main():
    spec={'scope':'BASE24 two seeds development only, fixed ordinary final union query/GT exposures',
          'hypothesis':'separated peaks may cause detrimental averaging; purely exploratory',
          'peak_definition':'two local maxima separated >=.5 normalized offset units, secondary height>=.5 primary, intervening valley<=.5 smaller height',
          'decoders':['original mean','argmax bin','discrete median','mode-centered coordinate windows radius .25/.5/1','global temperatures .5/.75/1.25/2'],
          'invalid_decoded_box':'nonpositive width/height => original box fallback, same denominator; invalid count reported',
          'limits':['not causal origin of failure','nonuniform bins are supervised point masses, not a calibrated density','score/LQE not recomputed','fixed pairs omit reassignment and other pool queries','not validation AP or training benefit','no heldout access'],
          'new_forwards':0,'new_backwards':0,'optimizer_updates':0,
          'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    (OUT/'26_多峰与解码探索口径.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
    bins_test=np.linspace(-4,4,33)
    uni=np.exp(-bins_test**2);bi=np.exp(-10*(bins_test-2)**2)+np.exp(-10*(bins_test+2)**2)
    assert not separated_peaks((uni/uni.sum())[None,:],bins_test)[0]
    assert separated_peaks((bi/bi.sum())[None,:],bins_test)[0]
    result={'spec':spec,'sources':{},'rows':[]}
    for account,seed in [('A',1),('B',2)]:
        all_rows=[]
        with tarfile.open(DATA/f'{account}_frozen_predictions.tar.gz') as tf,tarfile.open(DATA/f'{account}_development.tar.gz') as ta:
            seal=json.load(tf.extractfile('development/seal.json'))
            for item in seal['files']:
                blob=tf.extractfile('development/'+item['relative']).read();h=hashlib.sha256(blob).hexdigest();assert h==item['sha256']
                an=json.load(ta.extractfile(item['relative'].replace('_predictions.pt','_analysis.json')));assert h==an['prediction_sha256']
                result['sources'][account+'/'+item['relative']]=h
                saved=load_arrays(blob);o=saved['outputs'];regmax=o['pred_corners'].shape[-1]//4-1
                scale=float(o['reg_scale'].ravel()[0]);up=float(o['up'].ravel()[0]);step=(abs(up)*scale+1)**(2/(regmax-2))
                bins=np.array([-2*abs(up)*scale]+[1-step**i for i in range(regmax//2-1,0,-1)]+[0.]+[step**i-1 for i in range(1,regmax//2)]+[2*abs(up)*scale])
                for batch_index,r in enumerate(an['inputs']):
                    b=r['branches']['final']['native_union'];qs=np.array(b['pair_query']);gs=np.array(b['pair_GT'])
                    if not len(qs):continue
                    z=o['pred_corners'][batch_index,qs].astype(float).reshape(-1,4,regmax+1)
                    ref=o['ref_points'][batch_index,qs].astype(float);gt=xyxy(saved['targets'][batch_index]['boxes'][gs].astype(float))
                    original=xyxy(o['pred_boxes'][batch_index,qs].astype(float))
                    p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
                    mu=(p*bins).sum(-1);mode=bins[p.argmax(-1)]
                    means={'original':mu,'mode':mode,'median':bins[(np.cumsum(p,axis=-1)>=.5).argmax(-1)]}
                    for radius in [.25,.5,1.]:
                        v=p*(np.abs(bins-mode[...,None])<=radius);v/=v.sum(-1,keepdims=True)
                        means[f'window_{radius}']=(v*bins).sum(-1)
                    for temperature in [.5,.75,1.25,2.]:
                        v=np.exp((z-z.max(-1,keepdims=True))/temperature);v/=v.sum(-1,keepdims=True)
                        means[f'temperature_{temperature}']=(v*bins).sum(-1)
                    double=separated_peaks(p,bins)
                    methods={};bad={}
                    for name,mean in means.items():
                        edges=np.concatenate((ref[:,:2]-(scale/2+mean[:,:2])*ref[:,2:]/scale,ref[:,:2]+(scale/2+mean[:,2:])*ref[:,2:]/scale),-1)
                        invalid=np.any(edges[:,2:]<=edges[:,:2],axis=-1);edges[invalid]=original[invalid]
                        methods[name]=[iou(a,g) for a,g in zip(edges,gt)];bad[name]=invalid
                    np.testing.assert_allclose(methods['original'],b['pair_iou'],atol=1e-4)
                    for k in range(len(qs)):
                        base=methods['original'][k];cap=ceiling(ref[k],gt[k])
                        all_rows.append({'group':'good' if base>=.5 else ('bad_reachable' if cap>=.5 else 'bad_unreachable'),
                                         'double_edge_count':int(double[k].sum()),'methods':{n:v[k] for n,v in methods.items()},
                                         'invalid':{n:bool(v[k]) for n,v in bad.items()}})
        for group in ['ALL','bad_reachable','bad_unreachable','good']:
            rs=[r for r in all_rows if group=='ALL' or r['group']==group]
            for method in rs[0]['methods']:
                summary={'seed':seed,'group':group,'method':method,'pair_exposures':len(rs),
                         'any_separated_peaks':sum(r['double_edge_count']>0 for r in rs),
                         'edge_count':4*len(rs),'separated_peak_edges':sum(r['double_edge_count'] for r in rs),
                         'invalid_fallback':sum(r['invalid'][method] for r in rs),'mean_IoU':float(np.mean([r['methods'][method] for r in rs]))}
                for t in [.5,.75]:
                    summary[f'gained_{t}']=sum(r['methods']['original']<t<=r['methods'][method] for r in rs)
                    summary[f'lost_{t}']=sum(r['methods'][method]<t<=r['methods']['original'] for r in rs)
                result['rows'].append(summary)
    (OUT/'27_CPU开发组多峰与解码对照.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([r for r in result['rows'] if r['group']=='ALL'],ensure_ascii=False))

if __name__=='__main__':main()
