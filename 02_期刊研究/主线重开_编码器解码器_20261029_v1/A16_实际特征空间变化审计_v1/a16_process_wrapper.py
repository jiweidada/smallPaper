"""One bounded source-check/capture/source-check sequence; no retry or next queue."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
exit_path=ROOT.parent/'a16_spatial_response_v1_exit.json'
assert not exit_path.exists()
with (ROOT/'one_shot.lock').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()},f)
stages=[['a16_source_validation.py','before_capture'],['a16_capture_native_patches.py'],['a16_source_validation.py','after_capture']]
records=[];exit_code=0
for stage in stages:
    try:
        with (ROOT/(stage[0].replace('.py','')+'_'+(stage[1] if len(stage)>1 else 'capture')+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(ROOT/stage[0])]+stage[1:],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,timeout=600)
        exit_code=result.returncode
    except subprocess.TimeoutExpired:exit_code=124
    records.append({'stage':stage,'exit_code':exit_code})
    (ROOT/'wrapper_progress.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    if exit_code:break
exit_path.write_text(json.dumps({'status':'TERMINAL','exit_code':exit_code,'stages':records,'utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n',encoding='utf-8')
sys.exit(exit_code)
