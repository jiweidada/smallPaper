"""Run directly from the deployed script directory; no model or CUDA calls."""
from pathlib import Path
import argparse,hashlib,importlib,json,sys,sysconfig
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
    assert Path(sys.path[0]).resolve()==HERE and not (HERE/'queue.py').exists()
    import torch
    import queue
    assert Path(queue.__file__).resolve()==(Path(sysconfig.get_path('stdlib'))/'queue.py').resolve()
    q=queue.Queue();q.put('native import witness');assert q.get_nowait()=='native import witness'
    import replay,diagnostic_queue
    assert Path(replay.__file__).resolve()==HERE/'replay.py' and Path(diagnostic_queue.__file__).resolve()==HERE/'diagnostic_queue.py'
    spec=json.loads((HERE/'diagnostic_protocol.json').read_bytes());sources={p.name:sha(p) for p in HERE.glob('*.py')}
    assert sources==spec['files'] and not torch.cuda.is_initialized()
    result=dict(status='PASS_REAL_ENTRYPOINT_STDLIB_QUEUE_AND_TORCH_IMPORT',source_sha256=sources,
        protocol_sha256=sha(HERE/'diagnostic_protocol.json'),entrypoint_directory=str(HERE),stdlib_queue_path=queue.__file__,
        torch_version=str(torch.__version__),CUDA_initialized=False,model_forwards=0,training_epochs=0,optimizer_updates=0)
    with Path(out).open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],torch=result['torch_version'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
