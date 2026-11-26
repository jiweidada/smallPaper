"""Repair a proven archive path error without modifying or rerunning training."""
from pathlib import Path
import ast,difflib,hashlib,json
ROOT=Path.cwd();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v1'
new=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2'
new.mkdir(exist_ok=False)
for name in ('collect.py','verify_primary.py','wait_stopped.py','guards.py'):
    text=(old/name).read_text(encoding='utf8').replace('user_backbone_primary_collection12_20261011_v1','user_backbone_primary_collection12_20261011_v2')
    if name=='collect.py':
        assert text.count("'detector_preflight_start.json'")==1
        text=text.replace("'detector_preflight_start.json'","'detector_preflight_dispatch.json'")
        text=text.replace('from guards import ARMS,identity_plan,positive_terminal','from guards import ARMS,identity_plan,positive_terminal,preflight_launch')
        before="    output=base/'user_backbone_primary_collection12_20261011_v2'"
        text=text.replace(before,"    launch=read(screen/'detector_preflight_dispatch.json')\n    preflight_launch(launch,read(screen/'detector_preflight_completion.json'),prefix,seed,bundle)\n    assert not live(launch,'pid','start_ticks')\n"+before)
    if name=='verify_primary.py':
        text=text.replace('from guards import ARMS,identity_plan,positive_terminal','from guards import ARMS,identity_plan,positive_terminal,preflight_launch')
        before="    full=read(destination/'screen/detector_preflight_v1/report.json')"
        text=text.replace(before,"    prefix={1:'/25085400120',2:'/24085403014'}[seed]\n    preflight_launch(read(destination/'screen/detector_preflight_dispatch.json'),read(destination/'screen/detector_preflight_completion.json'),prefix,seed,PurePosixPath(receipt['paths']['screen_root'])/'bundle')\n"+before)
    if name=='guards.py':
        text+="\n\ndef preflight_launch(launch, completion, prefix, seed, bundle):\n    assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]\n    assert isinstance(launch['pid'],int) and launch['pid']>0 and str(launch['start_ticks']).isdigit()\n    python=prefix+('/envs/deim/bin/python' if seed==1 else '/deim_migration_runtime/envs/deim/bin/python')\n    assert launch['command']==[python,'-B',prefix+'/DEIM/experiments/user_backbone12_20261011_v1/code/preflight.py','--bundle',str(bundle)]\n    assert completion['exit_code']==completion['training_epochs']==completion['optimizer_updates']==0\n"
    ast.parse(text)
    (new/name).write_text(text,encoding='utf8')
protocol=json.loads((old/'protocol.json').read_bytes())
protocol.update(schema='user_backbone_read_only_primary_collection12_v2',collector_sha256=sha(new/'collect.py'),local_verifier_sha256=sha(new/'verify_primary.py'))
protocol['remote_sources']={n:sha(new/n) for n in ('collect.py','wait_stopped.py','guards.py')}
protocol['repair_scope']='v1 requested nonexistent detector_preflight_start.json; native dispatcher writes detector_preflight_dispatch.json. Require actual dispatch command and positive zero-update completion. Preserve v1 failure and empty partial folder, exact original training and all endpoint gates. v2 starts no training, evaluator or model.'
(new/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
diff=''.join(''.join(difflib.unified_diff((old/n).read_text(encoding='utf8').splitlines(True),(new/n).read_text(encoding='utf8').splitlines(True),fromfile='v1/'+n,tofile='v2/'+n)) for n in ('collect.py','verify_primary.py','wait_stopped.py','guards.py','protocol.json'))
(new/'repair.diff').write_text(diff,encoding='utf8')
for stem in ('user_backbone_primary_analysis12_20261011','user_backbone_native_order12_20261011'):
    prior=ROOT/('英文小论文/code/'+stem+'_v1');current=ROOT/('英文小论文/code/'+stem+'_v2');current.mkdir(exist_ok=False)
    text=(prior/'analyze.py').read_text(encoding='utf8').replace(stem+'_v1',stem+'_v2').replace('user_backbone_primary_collection12_20261011_v1','user_backbone_primary_collection12_20261011_v2')
    ast.parse(text);(current/'analyze.py').write_text(text,encoding='utf8')
    spec=json.loads((prior/'protocol.json').read_bytes());spec['analysis_sha256']=sha(current/'analyze.py')
    spec['dependencies_sha256']={n.replace('user_backbone_primary_collection12_20261011_v1','user_backbone_primary_collection12_20261011_v2'):sha(ROOT/n.replace('user_backbone_primary_collection12_20261011_v1','user_backbone_primary_collection12_20261011_v2')) for n in spec['dependencies_sha256']}
    spec['primary_collection_repair_only']=True
    (current/'protocol.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
seal=dict(files={p.relative_to(ROOT).as_posix():sha(p) for directory in (new,ROOT/'英文小论文/code/user_backbone_primary_analysis12_20261011_v2',ROOT/'英文小论文/code/user_backbone_native_order12_20261011_v2') for p in directory.iterdir() if p.is_file()},training_code_changes=0,new_training_epochs=0,new_model_forwards=0,new_evaluator_calls=0)
(ROOT/'outputs/user_primary_collection12_20261011_v2_source_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status='SOURCE_FROZEN_PATH_REPAIR_ONLY',remote={n:sha(new/n) for n in (*protocol['remote_sources'],'protocol.json')})))
