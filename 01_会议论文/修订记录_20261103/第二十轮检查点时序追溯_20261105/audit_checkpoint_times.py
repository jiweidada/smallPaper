"""Inspect saved checkpoint metadata and timestamps without loading tensor bytes or models."""
from collections import OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import io
import json
import pickle

HERE=Path(__file__).resolve().parent
REC=HERE.parent
SOURCE=REC/'第十九轮历史评价追溯_20261105/saved_array_log_matching.json'
RUN_AUDIT=REC/'run_audit.json'
OUTPUT=HERE/'checkpoint_time_trace.json'

class TensorMetadata:
    def __init__(self,storage,offset,size,stride,*args):
        self.shape=tuple(size)

class SafeMetadata(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='collections' and name=='OrderedDict': return OrderedDict
        if module=='torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'): return TensorMetadata
        if module=='torch' and name.endswith('Storage'): return name
        if module=='torch._utils' and name=='_rebuild_parameter': return lambda data,*args:data
        if module=='builtins' and name=='set': return set
        raise ValueError('unsupported pickle global: '+module+'.'+name)
    def persistent_load(self,pid):
        return ('inert_storage',pid)

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def load_metadata(path):
    with ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('/data.pkl')]
        assert len(names)==1 and z.getinfo(names[0]).file_size<32*1024*1024
        return SafeMetadata(io.BytesIO(z.read(names[0]))).load()

def stat(path):
    s=path.stat()
    return {'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns,
            'mtime_utc':datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),
            'creation_or_ctime_ns':s.st_ctime_ns}

def metadata(path):
    data=load_metadata(path)
    ema=data.get('ema',{})
    return {'path':str(path),'sha256':digest(path),'filesystem':stat(path),
            'saved_date':data.get('date'),'saved_last_epoch':data.get('last_epoch'),
            'saved_ema_updates':ema.get('updates') if isinstance(ema,dict) else None,
            'top_level_keys':sorted(data)}

def main():
    assert not OUTPUT.exists()
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    runs=json.loads(RUN_AUDIT.read_text(encoding='utf-8'))
    result=[]
    for run in ['N0','N1','N2','N3','N4','S1']:
        row=runs[run]
        cp=Path(row['source_dir'])/'best_stg2.pth'
        ckpt=metadata(cp)
        assert ckpt['sha256']==row['checkpoint']['sha256']
        assessments=[]
        for ev in [x for x in source['saved_evaluations'] if x['run']==run]:
            path=Path(ev['path'])
            assert digest(path)==ev['sha256']
            s=stat(path)
            assessments.append({'path':str(path),'sha256':ev['sha256'],'stored_date':ev['stored_date'],
                                'filesystem':s,'mtime_delta_vs_checkpoint_ns':s['mtime_ns']-ckpt['filesystem']['mtime_ns'],
                                'matched_log_epochs':[x['epoch'] for x in ev['complete_log_vector_matches']]})
        assert len(assessments)==8
        candidate=[x for x in assessments if x['matched_log_epochs']==[row['best_epoch']]]
        assert len(candidate)<=1
        nearest=min(assessments,key=lambda x:abs(x['mtime_delta_vs_checkpoint_ns']))
        result.append({'run':run,'best_log_epoch':row['best_epoch'],'checkpoint':ckpt,
                       'best_log_evaluation_file':candidate[0] if candidate else None,
                       'nearest_saved_eval_by_filesystem_mtime':nearest,'all_saved_evaluations':assessments})
    n2=next(x for x in result if x['run']=='N2')
    assert n2['best_log_evaluation_file'] is not None
    extra={}
    for name in ['best_stg1.pth','last.pth']:
        extra[name]=metadata(Path(runs['N2']['source_dir'])/name)
    code_paths=[Path(r'C:\Users\jiwei\Desktop\DEIM\engine\solver\det_solver.py'),
                Path(r'C:\Users\jiwei\Desktop\DEIM\engine\solver\_solver.py')]
    output={'status':'SAVED_CHECKPOINT_AND_EVALUATION_TIMESTAMPS_INSPECTED',
            'source_array_audit_sha256':digest(SOURCE),'source_run_audit_sha256':digest(RUN_AUDIT),
            'runs':result,'N2_additional_checkpoints':extra,
            'inspected_current_solver_files':[{'path':str(p),'sha256':digest(p)} for p in code_paths],
            'limits':['Filesystem timestamps and saved date strings can change during copying and are not cryptographic links.',
                      'Clock origin/timezone and historical source version are not authenticated.',
                      'Checkpoint tensor storage, detection model, predictions and evaluation code were not executed or regenerated.',
                      'Temporal proximity cannot prove that a saved checkpoint produced a particular evaluation.'],
            'models_run':0,'forwards':0,'evaluator_runs':0,'optimizer_updates':0}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
        f.write('\n')
    for row in result:
        candidate=row['best_log_evaluation_file']
        print(json.dumps({'run':row['run'],'best_epoch':row['best_log_epoch'],'checkpoint_date':row['checkpoint']['saved_date'],
            'last_epoch':row['checkpoint']['saved_last_epoch'],'ema_updates':row['checkpoint']['saved_ema_updates'],
            'matching_eval_date':candidate['stored_date'] if candidate else None,
            'matching_eval_mtime_delta_ms':candidate['mtime_delta_vs_checkpoint_ns']/1e6 if candidate else None,
            'nearest_saved_eval_name':Path(row['nearest_saved_eval_by_filesystem_mtime']['path']).name,
            'nearest_mtime_delta_seconds':row['nearest_saved_eval_by_filesystem_mtime']['mtime_delta_vs_checkpoint_ns']/1e9},ensure_ascii=False))

if __name__=='__main__':
    main()
