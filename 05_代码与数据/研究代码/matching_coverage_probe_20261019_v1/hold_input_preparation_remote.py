"""Honor latest user hold: pause only this new CPU data-preparation task."""
import pathlib,json,os,signal,time,datetime
root=pathlib.Path(CONFIG['storage'])/'DEIM/experiments/matching_coverage_probe_20261019_v1'
assert not list(root.glob('results_seed*')),'Inspect any forward before further action'
task=root/'task_records/inputs.json'
result={'account':CONFIG['account'],'reason':'User requested explain current research line first; do not rush experiments',
        'no_new_model_forward':True,'no_training_started':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if task.is_file():
    state=json.loads(task.read_text());result['original_task']=state
    if state['state']=='running':
        child=state['pid'];p=pathlib.Path('/proc')/str(child);st=(p/'stat').read_text().rsplit(')',1)[1].split()
        assert st[19]==str(state['start_ticks']) and st[0]!='Z'
        args=(p/'cmdline').read_bytes().split(b'\0');assert str(root/'probe/make_inputs.py').encode() in args
        launch=json.loads((root/'inputs_launch.json').read_text());supervisor=launch['supervisor_pid']
        parent=pathlib.Path('/proc')/str(supervisor);ps=(parent/'stat').read_text().rsplit(')',1)[1].split()
        pargs=(parent/'cmdline').read_bytes().split(b'\0')
        assert str(root/'bounded_task.py').encode() in pargs and str(root/'inputs_task.json').encode() in pargs
        assert int(st[1])==supervisor and os.getpgid(child)==child
        os.kill(supervisor,signal.SIGSTOP);os.killpg(child,signal.SIGSTOP)
        deadline=time.monotonic()+10
        while True:
            cs=(p/'stat').read_text().rsplit(')',1)[1].split()
            pp=(parent/'stat').read_text().rsplit(')',1)[1].split()
            if cs[0]=='T' and pp[0]=='T':break
            assert time.monotonic()<deadline;time.sleep(.05)
        result.update(state='CPU_input_preparation_paused_before_first_forward',child={'pid':child,'start_ticks':st[19],'state':'T'},supervisor={'pid':supervisor,'start_ticks':ps[19],'state':'T'})
    else:result['state']='no_active_CPU_preparation'
else:result['state']='no_CPU_preparation_on_this_account'
dest=root/'user_hold_before_discussion_20261020.json'
with dest.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
