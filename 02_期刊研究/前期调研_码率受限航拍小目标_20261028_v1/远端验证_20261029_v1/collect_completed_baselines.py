"""Read and package the completed, immutable baseline artifacts; no model execution."""
from pathlib import Path
import csv,hashlib,json,math,tarfile,time
ROOT=Path('/root/c10_region_restore_20261029_v1')
out=ROOT/'post_baseline_audit_20261029_v1';out.mkdir(exist_ok=True)
queue=json.loads((ROOT/'receipts/queue_completed.json').read_text())
assert queue['status']=='COMPLETED_BASELINES_AND_INITIAL_DIAGNOSTIC'
summary={'status':'COMPLETED_ARTIFACTS_AUDITED','server_unix':time.time(),'arms':{},'queue_elapsed_seconds':queue['elapsed_seconds']}
frozen=json.loads((ROOT/'receipts/code_frozen_at_launch.json').read_text())
amend=json.loads((ROOT/'receipts/evaluator_amendment_v1.json').read_text())
allowed={a['file']:a['corrected_sha256'] for a in amend['changes']}
for name,sha in frozen['files'].items():
    assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==allowed.get(name,sha),name
assert hashlib.sha256((ROOT/'experiment_protocol.json').read_bytes()).hexdigest()==frozen['protocol_sha256']
first_batches={}
for arm in ['D0','DA']:
    folder=ROOT/'results'/arm
    start=json.loads((folder/'start_receipt.json').read_text())
    complete=json.loads((folder/'completed.json').read_text())
    epochs=[json.loads(line) for line in (folder/'epoch_audit.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in epochs]==list(range(1,101))
    assert complete['epoch']==100 and complete['batches']==77200 and complete['amp_skips']==0
    assert complete['actual_optimizer_updates']==10073==complete['ema_updates']
    checkpoint=Path(complete['primary_checkpoint']);h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==complete['primary_checkpoint_sha256']
    rows=list(csv.DictReader((folder/'results.csv').open()))
    numeric=[]
    for row in rows:
        parsed={}
        for key,value in row.items():
            number=float(value);assert math.isfinite(number),(arm,key,value);parsed[key.strip()]=number
        numeric.append(parsed)
    assert len(numeric)==100
    first_batches[arm]=json.loads((folder/'first_batch_contracts.json').read_text())
    groups={}
    for label,a,b in [('epoch61_80',60,80),('epoch81_90',80,90),('epoch91_100',90,100)]:
        selected=numeric[a:b]
        groups[label]={key:sum(r[key] for r in selected)/len(selected) for key in selected[0] if key.startswith(('train/','val/','metrics/'))}
    summary['arms'][arm]={'completed':complete,'start':start,'epoch_rows':len(epochs),'csv_finite':True,
                          'reference_health_curve_groups':groups,'csv_last10':numeric[-10:]}
assert summary['arms']['D0']['start']['initial_state_sha256']==summary['arms']['DA']['start']['initial_state_sha256']
assert [r['geometry_sha256'] for r in first_batches['D0']]==[r['geometry_sha256'] for r in first_batches['DA']]
summary['same_formal_first8_geometry_hashes']=True
summary['code_and_protocol_integrity']=True
diagnostic=json.loads((ROOT/'evaluation/initial_diagnostic_result.json').read_text())
gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text());obs=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())
meta={im['image_id']:im for im in obs['images']}
valid=[a for a in gt['annotations'] if not a['iscrowd']]
summary['diagnostic_dataset']={'images':len(gt['images']),'valid_gt':len(valid),
    'input_area_bins':{
       'area_lt_256':sum(a['area']*meta[a['image_id']]['geometry']['ratio']**2<256 for a in valid),
       'area_256_to_lt_1024':sum(256<=a['area']*meta[a['image_id']]['geometry']['ratio']**2<1024 for a in valid),
       'area_ge_1024':sum(a['area']*meta[a['image_id']]['geometry']['ratio']**2>=1024 for a in valid)},
    'q20_weighted_bpp':obs['bpp']}
summary['cap500']={}
for arm in ['D0','DA']:
    summary['cap500'][arm]={}
    for condition in ['source','q20']:
        predictions=json.loads((ROOT/'evaluation'/arm/(condition+'_predictions_input.json')).read_text())
        by_image={im['id']:[] for im in gt['images']}
        for p in predictions:by_image[p['image_id']].append(p['score'])
        bottoms=[min(scores) for scores in by_image.values() if len(scores)==500]
        summary['cap500'][arm][condition]={'cap_hits':len(bottoms),'highest_last_kept_score':max(bottoms,default=None),
             'last_kept_score_quantiles':__import__('numpy').quantile(bottoms,[0,.25,.5,.75,1]).tolist() if bottoms else [],
             'cap_hits_with_last_score_ge_005':sum(v>=.05 for v in bottoms)}
summary['primary_table']={arm:{'source_AP_small_input_pct':100*diagnostic[arm]['source']['AP_small_input_500'],
                             'q20_AP_small_input_pct':100*diagnostic[arm]['q20']['AP_small_input_500'],
                             'loss_pp':diagnostic[arm]['remaining_small_AP_loss_pp'],
                             'paired_image_bootstrap95_pp':diagnostic[arm]['gap_paired_image_bootstrap95_pp']} for arm in ['D0','DA']}
(out/'baseline_integrity_and_curve_audit.json').write_text(json.dumps(summary,indent=2))
paths=[ROOT/'experiment_protocol.json',ROOT/'receipts',ROOT/'evaluation',ROOT/'observations/q20',ROOT/'dataset/diagnostic_gt_orig.json',
       ROOT/'dataset/tuning_gt_orig.json',ROOT/'dataset/data.yaml',out/'baseline_integrity_and_curve_audit.json']
for arm in ['D0','DA']:
    for name in ['start_receipt.json','completed.json','epoch_audit.jsonl','results.csv','args.yaml','first_batch_contracts.json','weights/last.pt']:
        paths.append(ROOT/'results'/arm/name)
    paths.append(ROOT/'logs'/('train_'+arm+'.log'))
paths.extend(p for p in (ROOT/'code').iterdir() if p.suffix in ['.py','.sh'])
archive=out/'completed_baseline_results.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
    for path in paths:tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=True)
h=hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest(),'checkpoints':['D0 fixed100 EMA','DA fixed100 EMA']}
(out/'completed_archive_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'archive':receipt,'primary_table':summary['primary_table'],'cap500':summary['cap500'],
                  'same_formal_first8_geometry_hashes':True,'code_and_protocol_integrity':True}),flush=True)
