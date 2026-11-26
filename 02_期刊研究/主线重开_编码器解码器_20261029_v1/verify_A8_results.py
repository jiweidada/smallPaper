"""Independent archive, graph-flow and rectangle-overlap audit; no model calls."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from collections import deque
import hashlib,json,tarfile
import numpy as np

R=Path(__file__).resolve().parent/'A8_固定query的窗口更新诊断_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def maxflow_count(edges):
    """Unit-capacity network flow, independent of the experiment's Kuhn matcher."""
    queries=sorted({q for e in edges for q in e});n=len(edges)
    qi={q:n+1+i for i,q in enumerate(queries)}
    sink=n+len(queries)+1;adj=[[] for _ in range(sink+1)]
    def add(u,v):
        adj[u].append([v,1,len(adj[v])]);adj[v].append([u,0,len(adj[u])-1])
    for g,e in enumerate(edges,1):
        add(0,g)
        for q in e:add(g,qi[q])
    for q in queries:add(qi[q],sink)
    flow=0
    while True:
        level=[-1]*len(adj);level[0]=0;queue=deque([0])
        while queue:
            u=queue.popleft()
            for v,cap,rev in adj[u]:
                if cap and level[v]<0:level[v]=level[u]+1;queue.append(v)
        if level[sink]<0:break
        index=[0]*len(adj)
        def send(u):
            if u==sink:return True
            while index[u]<len(adj[u]):
                e=adj[u][index[u]];v,cap,rev=e
                if cap and level[v]==level[u]+1 and send(v):
                    e[1]-=1;adj[v][rev][1]+=1;return True
                index[u]+=1
            return False
        while send(0):flow+=1
    return flow

def independent_count(gt,labels,boxes,classes,qids,keep,threshold):
    edges=[]
    for target,cls in zip(gt[keep],labels[keep]):
        good=set()
        for box,pred_cls,qid in zip(boxes,classes,qids):
            if cls!=pred_cls:continue
            iw=max(0,min(target[2],box[2])-max(target[0],box[0]))
            ih=max(0,min(target[3],box[3])-max(target[1],box[1]))
            a=max(0,target[2]-target[0])*max(0,target[3]-target[1])
            b=max(0,box[2]-box[0])*max(0,box[3]-box[1])
            overlap=iw*ih/max(a+b-iw*ih,1e-12)
            if overlap>=threshold:good.add(int(qid))
        edges.append(sorted(good))
    return maxflow_count(edges)

def coverage(mask,gt,tiny):
    center=0;fractions=[]
    h,w=mask.shape
    for b in gt[tiny]:
        x=(b[0]+b[2])/2;y=(b[1]+b[3])/2
        i=min(max(int(y//(800/h)),0),h-1);j=min(max(int(x//(800/w)),0),w-1)
        center+=int(mask[i,j])
        covered=0.0
        for i in range(h):
            for j in range(w):
                if not mask[i,j]:continue
                iw=max(0,min(b[2],(j+1)*800/w)-max(b[0],j*800/w))
                ih=max(0,min(b[3],(i+1)*800/h)-max(b[1],i*800/h))
                covered+=iw*ih
        fractions.append(covered/max((b[2]-b[0])*(b[3]-b[1]),1e-12))
    return center,float(np.mean(fractions))

def audit():
    rec=read(R/'backup/receipt.json');archive=R/'backup/A8_window_memory_results.tar.gz'
    assert sha(archive)==rec['archive_sha256'] and archive.stat().st_size==rec['archive_bytes']
    manifest=read(R/'backup/file_manifest.json')['files']
    with tarfile.open(archive,'r:gz') as t:
        for x in manifest:
            p=PurePosixPath(x['path']);assert not p.is_absolute() and '..' not in p.parts
            data=t.extractfile(x['path']).read()
            assert len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256']
            dest=(R/x['path']).resolve();assert dest.is_relative_to(R.resolve())
            if dest.exists():assert sha(dest)==x['sha256'],x['path']
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    frozen=read(R/'freeze_manifest.json')
    for x in frozen['files']:assert sha(R/x['file'])==x['sha256']
    s=read(R/'summary.json');done=read(R/'completed.json')
    assert done['summary_sha256']==sha(R/'summary.json')
    assert s['new_full_detector_forwards']==15 and s['global_full_detector_forwards']==200
    assert s['state_digest_before']==s['state_digest_after']
    entered=[json.loads(x) for x in (R/'forward_counter.jsonl').read_text().splitlines()]
    assert [x['global_count'] for x in entered]==list(range(186,201))
    cases=read(R/'cases.json')['cases'];case_by_name={x['file_name']:x for x in cases}
    native={};verified=0;aggregate={}
    for row in s['rows']:
        file=R/'outputs'/row['output_file'];assert sha(file)==row['output_sha256']
        z=np.load(file,allow_pickle=False)
        boxes=z['boxes800'].astype(float);logits=z['raw_logits'];raw=z['raw_boxes'].astype(float)
        flat=z['flat_ids'];q=len(logits)
        assert len(flat)==q==row['query_count'] and logits.shape==(q,12)
        assert np.array_equal(z['query_ids'],flat//12) and np.array_equal(z['labels'],flat%12)
        sig=1/(1+np.exp(-logits.astype(float)))
        expected=sig.reshape(-1)[flat]
        assert np.allclose(z['scores'],expected,rtol=0,atol=2e-7)
        omitted=np.ones(sig.size,bool);omitted[flat]=False
        assert expected.min()>=sig.reshape(-1)[omitted].max()-2e-7
        out=np.concatenate([raw[:,:2]-raw[:,2:]/2,raw[:,:2]+raw[:,2:]/2],axis=1)*800
        assert np.allclose(out[z['query_ids']],boxes,rtol=0,atol=1e-4)
        case=case_by_name[row['case']];ann=case['analysis_gt']
        gt=np.array([x['box800_xyxy'] for x in ann]);gl=np.array([x['class_id']+1 for x in ann])
        size=np.array([x['size800_from640'] for x in ann])
        assert np.array_equal(gt,z['gt_boxes800']) and np.array_equal(gl,z['gt_labels'])
        assert np.array_equal(size,z['gt_sizes800'])
        for name,keep in [('all',np.ones(len(size),bool)),('lt8',size<8),('8to16',(size>=8)&(size<16))]:
            for thresh in [0.5,0.75]:
                key=f'{name}_iou{int(thresh*100)}'
                count=independent_count(gt,gl,boxes,z['labels'],z['query_ids'],keep,thresh)
                assert count==row['counts'][key]['matched'] and int(keep.sum())==row['counts'][key]['gt']
                verified+=1
        mask=z['window_mask'];c,a=coverage(mask,gt,size<8)
        assert c==row['coverage']['tiny_center_covered']
        assert abs(a-row['coverage']['tiny_mean_area_covered'])<1e-12
        if row['arm']=='NATIVE':native[row['case']]={k:z[k].copy() for k in ['window_mask','raw_boxes','raw_logits','flat_ids']}
        else:
            base=native[row['case']]
            if row['arm']!='ALL_WINDOWS':assert mask.sum()==base['window_mask'].sum()
            if row['arm']=='REPLAY_NATIVE':
                assert np.array_equal(mask,base['window_mask'])
                assert np.max(abs(z['raw_boxes']-base['raw_boxes']))<=1e-6
                assert np.max(abs(z['raw_logits']-base['raw_logits']))<=1e-6
                assert np.array_equal(flat,base['flat_ids'])
        group=aggregate.setdefault(row['arm'],{'tiny_center_covered':0,'gt':0,'tiny_gt':0,
          'lt8_iou50':0,'lt8_iou75':0,'all_iou50':0,'all_iou75':0,'windows':[],'query_counts':[]})
        group['tiny_center_covered']+=c;group['gt']+=len(gt);group['tiny_gt']+=int((size<8).sum())
        group['windows'].append(int(mask.sum()));group['query_counts'].append(q)
        for key in ['lt8_iou50','lt8_iou75','all_iou50','all_iou75']:group[key]+=row['counts'][key]['matched']
    assert len(s['rows'])==15 and verified==90
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_ARCHIVE_ALL30_FILES_15_OUTPUTS_90_INDEPENDENT_FLOW_COUNTS_AND_WINDOW_GEOMETRY',
      'archive_sha256':rec['archive_sha256'],'archive_files':len(manifest),'independent_graph_counts':verified,
      'aggregate':aggregate,'new_detector_forwards':0,'new_fits':0,
      'audit_development_note':'Initial audit compared raw-query-order boxes against score-ranked boxes and failed. Corrected the independent audit to gather by the recorded physical query IDs; no tolerance, experiment output, protocol or frozen experiment code was changed.',
      'scope':'Independent endpoint arithmetic and artifact/identity integrity. Query seeds are clamped by inspected frozen code; no separate serialized per-arm seed-tensor digest was collected. Not AP/generalization/novelty validation.'}
    (R/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':audit()
