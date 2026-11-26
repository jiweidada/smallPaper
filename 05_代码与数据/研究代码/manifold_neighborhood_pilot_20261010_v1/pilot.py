"""Bounded, descriptive feature-neighborhood pilot on two existing images.

No manifold existence, curvature, causal root, deployable method or AP claim.
This measures neighborhoods and local PCA spectra, not t-SNE separation.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/manifold_neighborhood_pilot_20261010_v1'
CAP=ROOT/'outputs/model_artifact_recorder_20261010_v1/collection_seed1/capture_seed1_final132_images0_1'
STAGES={
 'stem':'backbone.stem@0.value',
 'backbone_stage0':'backbone.stages.0@0.value',
 'backbone_P3':'backbone.stages.1@0.value',
 'encoder_projection_P3':'encoder.input_proj.0@0.value',
 'encoder_FPN_P3':'encoder.fpn_blocks.1@0.value',
 'encoder_output_P3':'encoder@0.value.0',
 'decoder_projection_P3':'decoder.input_proj.0@0.value',
 'encoder_memory_P3':'decoder.enc_output@0.value'}
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def sample(feature,points):
    """Bilinear align_corners=False, border clamp. Coordinates normalized xy.
    This is an analysis sampling convention, not an exact receptive-field map.
    """
    c,h,w=feature.shape
    x=np.clip(points[:,0]*w-.5,0,w-1);y=np.clip(points[:,1]*h-.5,0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    dx=x-x0;dy=y-y0
    return ((1-dx)*(1-dy))[:,None]*feature[:,y0,x0].T+(dx*(1-dy))[:,None]*feature[:,y0,x1].T+((1-dx)*dy)[:,None]*feature[:,y1,x0].T+(dx*dy)[:,None]*feature[:,y1,x1].T

def unit(x):
    n=np.linalg.norm(x,axis=1);valid=n>0
    z=np.zeros_like(x);z[valid]=x[valid]/n[valid,None]
    return z,valid,n

def neighborhoods(query,reference,k=10):
    d=np.maximum(2-2*query@reference.T,0)
    order=np.argsort(d,axis=1,kind='stable')
    # Equal-radius Poincare remapping is strictly monotone in chord distance.
    # It cannot alter neighborhood ranks without changing representations/radii.
    r=.5;hyper=np.arccosh(1+2*r*r*d/(1-r*r)**2)
    assert np.array_equal(order,np.argsort(hyper,axis=1,kind='stable'))
    return order[:,:k],order[:,:32],d

def self_checks():
    feature=np.array([[[0,2],[4,6]]],float)
    assert np.array_equal(sample(feature,np.array([[.5,.5],[.25,.25],[.75,.75]])),[[3],[0],[6]])
    x=np.eye(3);idx,_,_=neighborhoods(x,x)
    assert np.array_equal(idx[:,0],np.arange(3))
    # A centered pair on a line has effective PCA dimension exactly one.
    a=np.array([[-1,0],[1,0]],float);ev=np.linalg.eigvalsh(a@a.T)
    assert (ev.sum()**2/(ev@ev))==1

def freeze():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol={
      'status':'FROZEN_BEFORE_FIRST_FEATURE_NEIGHBORHOOD_RESULTS',
      'frozen_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate'},
      'user_steering':'Consider manifold analysis as a different perspective for locating the small-object problem.',
      'source':{'capture_root':str(CAP),'source_manifest_sha256':sha(CAP/'manifest.json'),'prior_delivery_sha256':sha(ROOT/'outputs/model_artifact_recorder_20261010_v1/delivery.json')},
      'code_sha256':sha(__file__),'seed':1,'endpoint':132,'images':[0,1],
      'scope':'Existing two-image engineering fixture in ordinary EVAL. Method/instrumentation pilot only, not a representative manifold study or a causal gate. Latest32-image pool archive has logits/boxes but no feature vectors.',
      'stages':STAGES,
      'points':'All288 native annotations represented by normalized GT-center bilinear feature samples. Coordinate occupancy is an annotation proxy, not a pure-object embedding; overlapping centers flagged and retained.',
      'background':'For each image,64 points evenly selected in row-major order from40x40 regular input-grid centers outside every GT box expanded1.5x. Select before inspecting features. Points are unannotated coordinates, not proven pure receptive fields.',
      'primary':'L2-normalized features, leave-one-image-out k10 nearest neighbors; GT class purity minus available-class prior and background fraction minus reference-background prior. All input-short bins <4,4to8,8to16,ge16 and perclass/image retained.',
      'spectrum':'k32 other-image neighbors; center their normalized features; eigenvalues of32x32 Gram. Participation-ratio effective PCA dimension (sum lambda)^2/sum lambda^2 is a descriptive spectral statistic, not a certified intrinsic manifold dimension. Also top3 variance fraction.',
      'gauge_control':'For fixed equal Poincare radius r=.5, compare full neighbor order against spherical/chord order. No learned hyperbolic representation or variable-radii experiment. A monotone distance remapping alone cannot repair neighborhoods.',
      'zeros':'Zero-norm sampled vectors are flagged, retained in row records and excluded from cosine metrics with explicit valid denominators; no feature-dependent point replacement.',
      'curvature':'Not estimated in this pilot. Static local PCA nonflatness is not called intrinsic curvature; actual controlled perturbation trajectories would be needed.',
      'decision_boundary':'No primary success/novelty/AP or mechanism-priority gate from two images. No automatic detector training. Expand to bothseeds and prospectively frozen image panel before root/prioritization claims.',
      'budget':{'model_forward_calls':0,'model_backward_calls':0,'optimizer_steps':0,'training_epochs_added':0},
      'limits':['Channels and receptive fields differ by stage; reduced participation ratio is not alone semantic loss.','Cross-image neighbors reduce same-image leakage but2images remain confounded by scene/class/instance composition.','kNN sampling density, feature norm and background prior matter. No t-SNE/UMAP/graph geodesic is used as proof.','No feature values from failed/interrupted experiments are substituted.']}
    save(OUT/'protocol.json',protocol)
    print(json.dumps({'status':'FROZEN_MANIFOLD_NEIGHBORHOOD_PILOT','protocol_sha256':sha(OUT/'protocol.json')}))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','analyze']);args=p.parse_args()
    if args.mode=='freeze':freeze();return
    protocol=read(OUT/'protocol.json');assert sha(__file__)==protocol['code_sha256']
    self_checks()
    manifest=read(CAP/'manifest.json');assert sha(CAP/'manifest.json')==protocol['source']['source_manifest_sha256']
    for n,h in manifest['files'].items():assert sha(CAP/n)==h,n
    clouds=[];features={s:[] for s in STAGES};metadata=[]
    for image in (0,1):
        folder=CAP/f'image_{image:06d}';truth=read(folder/'GT_original.json');trace=read(folder/'trace.json')
        assert sha(folder/'GT_original.json')==read(folder/'GT_manifest.json')['sha256']
        wh=np.array(trace['metadata']['original_wh'],float)
        boxes=np.array([a['bbox'] for a in truth['annotations']],float)
        lo=boxes[:,:2]/wh;hi=(boxes[:,:2]+boxes[:,2:])/wh;center=(lo+hi)/2
        sizes=((hi-lo)*640).min(1);n=len(boxes)
        grid=(np.stack(np.meshgrid(np.arange(40),np.arange(40)),axis=-1).reshape(-1,2)+.5)/40
        half=(hi-lo)*.75;eligible=~((grid[:,None]>=center[None]-half[None])&(grid[:,None]<=center[None]+half[None])).all(-1).any(-1)
        background=grid[eligible];assert len(background)>=64
        background=background[np.floor(np.arange(64)*(len(background)-1)/63).astype(int)]
        points=np.concatenate((center,background));assert np.isfinite(points).all()
        labels=np.array([a['category_id'] for a in truth['annotations']]+[-1]*64)
        overlap=((center[:,None]>=lo[None])&(center[:,None]<=hi[None])).all(-1).sum(1)-1
        cloud={'image':image,'points':points,'labels':labels,'sizes':np.concatenate((sizes,np.full(64,np.nan))),'ids':[a['id'] for a in truth['annotations']]+[-1]*64,'overlap':np.concatenate((overlap,np.zeros(64,int))),'GT_count':n}
        clouds.append(cloud);metadata.append({'image':image,'GT_count':n,'background_count':64,'eligible_background_grid':int(eligible.sum())})
        with np.load(folder/'tensors.npz',allow_pickle=False) as z:
            assert np.array_equal(z['decoder.decoder.layers.0.native_spatial_shapes'],[[80,80],[40,40],[20,20]])
            for stage,key in STAGES.items():
                value=z[key]
                if stage=='encoder_memory_P3':value=value[:,:6400].reshape(1,80,80,256).transpose(0,3,1,2)
                assert value.ndim==4 and value.shape[0]==1 and np.isfinite(value).all()
                features[stage].append(sample(value[0].astype(np.float64),points))
    rows=[];stage_checks=[]
    for stage,ff in features.items():
      normalized=[unit(v) for v in ff]
      for ii,cloud in enumerate(clouds):
        other=1-ii;query,valid,norm=normalized[ii];reference,rv,_=normalized[other]
        refs=reference[rv];labels=clouds[other]['labels'][rv];assert len(refs)>=32
        order,local,_=neighborhoods(query[valid],refs);valid_indices=np.flatnonzero(valid)
        bgprior=float((labels<0).mean());spectra={}
        for pidx,neighbors in zip(valid_indices,local):
            center=refs[neighbors]-refs[neighbors].mean(0)
            eigen=np.linalg.eigvalsh(center@center.T)
            assert eigen.min()>=-1e-10*max(1,float(eigen.sum()))
            eigen=np.maximum(eigen,0);total=float(eigen.sum())
            spectra[int(pidx)]=(float(total*total/(eigen@eigen)) if total else None,float(eigen[-3:].sum()/total) if total else None)
        ranks={int(pidx):labels[neighbors] for pidx,neighbors in zip(valid_indices,order)}
        for j in range(len(query)):
            isbg=cloud['labels'][j]<0;size=cloud['sizes'][j]
            r={'stage':stage,'image':cloud['image'],'point':j,'annotation_id':cloud['ids'][j],'category':int(cloud['labels'][j]),'kind':'unannotated_coordinate' if isbg else 'GT_center','input_short_pixels':None if isbg else float(size),'input_size_bin':'background' if isbg else 'lt4' if size<4 else '4to8' if size<8 else '8to16' if size<16 else 'ge16','center_overlaps_other_GT':int(cloud['overlap'][j]),'channels':query.shape[1],'raw_feature_norm':float(norm[j]),'valid_cosine_sample':bool(valid[j]),'reference_points':len(refs),'reference_background_prior':bgprior,'reference_same_label_prior':float((labels==cloud['labels'][j]).mean()),'k10_background_fraction':None,'k10_same_label_fraction':None,'background_fraction_minus_prior':None,'same_label_fraction_minus_prior':None,'local_PCA_effective_dimension':None,'local_top3_variance_fraction':None}
            if valid[j]:
                neighbors=ranks[j];bg=float((neighbors<0).mean());same=float((neighbors==cloud['labels'][j]).mean())
                r.update(k10_background_fraction=bg,k10_same_label_fraction=same,background_fraction_minus_prior=bg-bgprior,same_label_fraction_minus_prior=same-r['reference_same_label_prior'],local_PCA_effective_dimension=spectra[j][0],local_top3_variance_fraction=spectra[j][1])
            rows.append(r)
        stage_checks.append({'stage':stage,'image':cloud['image'],'channels':query.shape[1],'points':len(query),'zero_norm_points':int((~valid).sum()),'complete_kNN_order_equal_fixed_radius_hyperbolic':True,'all_local_spectra_checked':int(valid.sum())})
    groups=[]
    for stage in STAGES:
      for image in (0,1):
        for size in ['lt4','4to8','8to16','ge16','background']:
            subset=[r for r in rows if r['stage']==stage and r['image']==image and r['input_size_bin']==size];valid=[r for r in subset if r['valid_cosine_sample']]
            result={'stage':stage,'image':image,'input_size_bin':size,'points':len(subset),'valid_cosine_points':len(valid)}
            for key in ['k10_background_fraction','background_fraction_minus_prior','k10_same_label_fraction','same_label_fraction_minus_prior','local_PCA_effective_dimension','local_top3_variance_fraction']:
                values=[r[key] for r in valid if r[key] is not None];result[key+'_mean']=float(np.mean(values)) if values else None
            groups.append(result)
    with (OUT/'per_point_neighborhood.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'status':'PASS_DESCRIPTIVE_TWO_IMAGE_FEATURE_NEIGHBORHOOD_PILOT','material_passport':protocol['material_passport'],'scientific_scope':'ANALYZED_PILOT_ONLY','seed':1,'endpoint':132,'images':metadata,'GT_count':sum(c['GT_count'] for c in clouds),'stages':list(STAGES),'groups':groups,'stage_checks':stage_checks,'rows':len(rows),'fixed_radius_distance_remapping_neighbor_changes':0,'manifold_existence_established':False,'curvature_measured':False,'root_cause_established':False,'AP_evidence':False,'training_launched':False,'model_forwards_added':0,'epochs_added':0,'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'source_files_verified':len(manifest['files'])}
    save(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['groups','stage_checks']},ensure_ascii=False))

if __name__=='__main__':main()
