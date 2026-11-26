"""Exploratory nested-capacity analysis of cached outputs, without a detector."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from a8_analysis import physical_query_matching
from verify_A8_results import maxflow_count

R=Path(__file__).resolve().parent
A=R/'A8_固定query的窗口更新诊断_v1'
O=R/'A9_已有预测的几何分类筛选容量分解_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def count(gt,gl,pr,pl,qid,threshold):
    matched=physical_query_matching(gt,gl,pr,pl,qid,threshold)
    # Independent network flow over independently assembled rectangle-IoU edges.
    edges=[]
    for b,label in zip(gt,gl):
        e=set()
        for p,l,q in zip(pr,pl,qid):
            if label!=l:continue
            w=max(0,min(b[2],p[2])-max(b[0],p[0]));h=max(0,min(b[3],p[3])-max(b[1],p[1]))
            ab=max(0,b[2]-b[0])*max(0,b[3]-b[1]);ap=max(0,p[2]-p[0])*max(0,p[3]-p[1])
            if w*h/max(ab+ap-w*h,1e-12)>=threshold:e.add(int(q))
        edges.append(sorted(e))
    result=int((matched>=0).sum())
    assert result==maxflow_count(edges)
    return result

def main():
    O.mkdir(exist_ok=True)
    assert not (O/'started.json').exists(),'Do not rerun an outcome analysis silently.'
    summary=json.loads((A/'summary.json').read_text(encoding='utf-8'))
    verify=json.loads((A/'verification.json').read_text(encoding='utf-8'))
    assert verify['status'].startswith('PASS_') and summary['global_full_detector_forwards']==200
    sources=[x for x in summary['rows'] if x['arm']=='NATIVE'];assert len(sources)==3
    protocol={'utc':datetime.now(timezone.utc).isoformat(),
      'status':'FROZEN_BEFORE_A9_CAPACITY_VALUES_AFTER_KNOWN_A8_RESULTS',
      'question':'Is remaining tiny-object failure mostly a lack of accurate raw query boxes, exclusion of their query IDs, or class-label availability in native TopQ?',
      'design':'Exploratory post-hoc analysis of3 author-train cases; A8 counts51/69 atIoU50 and12/69 atIoU75 are already known. Not an independent confirmatory study.',
      'cohorts':['all','lt8','8to16'],'thresholds':[0.5,0.75],
      'nested_pools':['class-agnostic all raw physical queries','class-agnostic unique query IDs retained by native TopQ','class-aware native TopQ'],
      'secondary':'Argmax-class prediction on all raw queries; not nested with native multiclass TopQ.',
      'primary':'<8 atIoU75; pooled capacity-gap(all-geometry minus native-class)/(GT minus native-class). >=.5 plus a gap in>=2images prioritizes ranking/class audit; <=.2 deprioritizes it; intermediate is inconclusive.',
      'interpretation':'Differences are between maximum matching capacities. They are not a per-GT causal error partition and cannot localize an encoder versus decoder failure.',
      'numerics':'Float32 center/size-to-corner conversion; actual native postprocess pixel boxes overwrite retained query IDs. Float64 conversion is a rounding sensitivity check, not a tuned alternative.',
      'sources':[{'file':x['output_file'],'sha256':x['output_sha256']} for x in sources],
      'new_detector_forwards':0,'new_fits':0,'main_detector_updates':0,'global_forwards_unchanged':200}
    save(O/'protocol.json',protocol)
    save(O/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(O/'protocol.json')})
    rows=[]
    for row in sources:
        path=A/'outputs'/row['output_file'];assert sha(path)==row['output_sha256']
        with np.load(path,allow_pickle=False) as z:
            raw=z['raw_boxes'];logits=z['raw_logits'];q=len(raw)
            # Preserve native selected-query coordinates exactly, avoiding an
            # artificial nested-pool difference from CPU/GPU rounding.
            full=np.concatenate([raw[:,:2]-raw[:,2:]*np.float32(.5),raw[:,:2]+raw[:,2:]*np.float32(.5)],axis=1)*np.float32(800)
            full=full.astype(np.float64)
            precise=np.concatenate([raw[:,:2].astype(float)-raw[:,2:].astype(float)/2,raw[:,:2].astype(float)+raw[:,2:].astype(float)/2],axis=1)*800
            for i,b in zip(z['query_ids'],z['boxes800']):full[i]=b
            gt=z['gt_boxes800'];labels=z['gt_labels'];sizes=z['gt_sizes800']
            retained=np.unique(z['query_ids'])
            for name,keep in [('all',np.ones(len(gt),bool)),('lt8',sizes<8),('8to16',(sizes>=8)&(sizes<16))]:
                for threshold in [0.5,0.75]:
                    target=gt[keep];actual=labels[keep];ones=np.ones(len(target),int)
                    allgeom=count(target,ones,full,np.ones(q,int),np.arange(q),threshold)
                    keptgeom=count(target,ones,full[retained],np.ones(len(retained),int),retained,threshold)
                    native=count(target,actual,z['boxes800'],z['labels'],z['query_ids'],threshold)
                    argmax=count(target,actual,full,logits.argmax(-1),np.arange(q),threshold)
                    fp64=count(target,ones,precise,np.ones(q,int),np.arange(q),threshold)
                    assert len(target)>=allgeom>=keptgeom>=native
                    rows.append({'case':row['case'],'cohort':name,'iou':threshold,'gt':len(target),'queries':q,
                      'unique_native_retained_queries':len(retained),'all_query_geometry':allgeom,
                      'retained_query_geometry':keptgeom,'native_class_TopQ':native,'all_query_argmax_class':argmax,
                      'geometry_capacity_shortfall':len(target)-allgeom,
                      'query_retention_capacity_gap':allgeom-keptgeom,'retained_class_capacity_gap':keptgeom-native,
                      'float64_all_geometry_sensitivity':fp64})
    aggregate=[]
    keys=['gt','all_query_geometry','retained_query_geometry','native_class_TopQ','all_query_argmax_class',
          'geometry_capacity_shortfall','query_retention_capacity_gap','retained_class_capacity_gap','float64_all_geometry_sensitivity']
    for name in protocol['cohorts']:
        for threshold in protocol['thresholds']:
            subset=[x for x in rows if x['cohort']==name and x['iou']==threshold]
            aggregate.append({'cohort':name,'iou':threshold,**{k:sum(x[k] for x in subset) for k in keys}})
    primary=next(x for x in aggregate if x['cohort']=='lt8' and x['iou']==.75)
    fraction=(primary['all_query_geometry']-primary['native_class_TopQ'])/max(1,primary['gt']-primary['native_class_TopQ'])
    images=sum(x['all_query_geometry']>x['native_class_TopQ'] for x in rows if x['cohort']=='lt8' and x['iou']==.75)
    if fraction>=.5 and images>=2:decision='RANKING_CLASS_CAPACITY_GAP_WORTH_CLOSEST_METHOD_AUDIT_NOT_NEW_ALGORITHM'
    elif fraction<=.2:decision='RANKING_CLASS_CAPACITY_NOT_PRIMARY_IN_THESE_CASES_GEOMETRY_UNRESOLVED'
    else:decision='INCONCLUSIVE_CAPACITY_DECOMPOSITION'
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_INDEPENDENT_FLOW_CROSSCHECKED',
      'decision':decision,'primary_fraction':fraction,'primary_positive_gap_images':images,
      'rows':rows,'aggregate':aggregate,'independent_flow_counts':len(rows)*5,
      'new_detector_forwards':0,'global_forwards_unchanged':200,'new_fits':0,'main_detector_updates':0,
      'scientific_mainline_validated':False,'scope':protocol['interpretation']}
    save(O/'summary.json',result)
    save(O/'completed.json',{'status':'COMPLETED','summary_sha256':sha(O/'summary.json'),
          'protocol_sha256':sha(O/'protocol.json'),'new_detector_forwards':0,'new_fits':0})
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
