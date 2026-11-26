"""One finite OS-only trial; persistent guardian always restores original CPU affinity."""
import base64,datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path(CONFIG['root']);path=root/'artifacts/performance_CPU_locality_v1';assert not path.exists()
queue=json.loads((root/'queue_state.json').read_text());assert queue['state']=='training' and len(queue['active_jobs'])==3
path.mkdir();program='CONFIG='+repr(CONFIG)+'\n'+base64.b64decode('__SOURCE__').decode()
script=path/'trial.py';script.write_text(program)
with (path/'trial.log').open('xb') as log:
 process=subprocess.Popen([CONFIG['python'],'-u',str(script)],cwd=path,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'account':CONFIG['account'],'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
 'trial_directory':str(path),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'no_training_file_or_recipe_change':True}
(path/'launch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
