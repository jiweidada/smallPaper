from pathlib import Path
import hashlib,json,os,subprocess,time

ROOT=Path(__file__).resolve().parent
PYTHON='/opt/conda/envs/mamba/bin/python'
if __name__=='__main__':
    with (ROOT/'launch_once.lock').open('x') as f:f.write(str(os.getpid()))
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    assert not busy.strip(),busy
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
    with (ROOT/'supervisor_stdout.log').open('x') as out,(ROOT/'supervisor_stderr.log').open('x') as err:
        p=subprocess.Popen([PYTHON,'-u',str(ROOT/'supervise_A48.py')],cwd=ROOT,env=env,
                           stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
    ticks=int(Path('/proc/%s/stat'%p.pid).read_text().split(') ',1)[1].split()[19])
    result={'status':'SUPERVISOR_LAUNCHED','pid':p.pid,'start_ticks':ticks,'unix':time.time(),
            'protocol_sha256':hashlib.sha256((ROOT/'protocol_FROZEN.json').read_bytes()).hexdigest(),
            'scope':'prepare -> bounded capacity -> fresh B/M/Q each12 only; no auto retry',
            'launch_does_not_prove_training_started':True}
    (ROOT/'launch_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
