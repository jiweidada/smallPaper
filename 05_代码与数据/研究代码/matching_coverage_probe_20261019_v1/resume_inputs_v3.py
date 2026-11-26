"""Explicit user resumption: only original CPU generator, no model forward."""
import datetime,hashlib,json
from control_probe import ctl,ROOTS,LOCAL,HERE
passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes());client=None
try:
 client=ctl.connect(ctl.ACCOUNTS['A']['port'],passwords);root=ROOTS['A'];sf=client.open_sftp();raw=(HERE/'resume_inputs_watch_v3.py').read_bytes()
 with sf.open(root+'/resume_inputs_watch_v3.py','wx') as f:f.write(raw)
 program='''import pathlib,json,os,signal,subprocess,time,hashlib
root=pathlib.Path(PROBE_ROOT);hold=json.loads((root/'user_hold_before_discussion_20261020.json').read_text())
assert not list(root.glob('results_seed*')) and not list(root.glob('isolated_results_seed*'))
assert hashlib.sha256((root/'resume_inputs_watch_v3.py').read_bytes()).hexdigest()==EXPECTED
for key in ['child','supervisor']:
 v=hold[key];p=pathlib.Path('/proc')/str(v['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert s[19]==str(v['start_ticks']) and s[0]=='T'
 args=(p/'cmdline').read_bytes().split(b'\\0')
 assert (str(root/'probe/make_inputs.py') if key=='child' else str(root/'bounded_task.py')).encode() in args
# The old timeout includes the deliberate hold; retire only its stopped parent.
parent=hold['supervisor']['pid'];os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
time.sleep(.2)
with (root/'inputs_resume_v3_supervisor.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u','-B',str(root/'resume_inputs_watch_v3.py'),str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps({'account':'A','new_input_observer_pid':p.pid,'original_input_pid':hold['child']['pid'],'new_model_forwards':0}),flush=True)
'''
 result=ctl.execute(client,'A','PROBE_ROOT='+repr(root)+'\nEXPECTED='+repr(hashlib.sha256(raw).hexdigest())+'\n'+program,timeout=30)
 p=LOCAL/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_resume_inputs_v3.json');p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False));assert result['returncode']==0
finally:
 passwords.clear()
 if client:client.close()
