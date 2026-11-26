"""Read existing trainer artifacts and /proc identities; no framework or mutation."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,sys

def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text())
def config(bundle):
    r=read(bundle/'receipt.json');path=bundle/'resolved.yml';assert sha(path.read_bytes())==r['resolved_sha256']
    c=read(path);keys=('criterion','DEIMCriterion','use_amp','warmup_iter','epoches','flat_epoch','no_aug_epoch','lr_warmup_scheduler')
    return dict(receipt_sha256=sha((bundle/'receipt.json').read_bytes()),resolved_sha256=r['resolved_sha256'],selected_values={k:c[k] for k in keys if k in c},optimizer_configuration_sha256=sha(json.dumps(c['optimizer'],sort_keys=True).encode()))
def lines(path):
    if not path.exists():return dict(exists=False,rows=[],incomplete_last_line=False)
    b=path.read_bytes();raw=b.splitlines();partial=bool(b and not b.endswith(b'\n'))
    if partial:raw=raw[:-1]
    rows=[json.loads(x) for x in raw if x]
    result=dict(exists=True,sha256=sha(b),bytes=len(b),incomplete_last_line=partial,source_complete_line_count=len(rows))
    if path.name=='log.txt':
        keys={'epoch','test_coco_eval_bbox','train_loss','train_lr'}
        keys.update('train_loss_'+loss+suffix for loss in ('mal','vfl','bbox','giou','fgl','ddf') for suffix in ('','_pre','_enc_0'))
        keys.update(('train_loss_ddf_aux_0','train_loss_ddf_aux_1'))
        result['selected_log_fields_only']=True
        rows=[{k:v for k,v in r.items() if k in keys} for r in rows if r['epoch']<12]
    result['rows']=rows
    return result
def processes(screen):
    out=[]
    for name,pidkey,tickkey in (('controller_start.json','pid','start_ticks'),('candidate_start.json','child_pid','child_start_ticks')):
        if not (screen/name).exists():continue
        r=read(screen/name);pid=r[pidkey]
        try:
            f=Path(f'/proc/{pid}/stat').read_text().split();out.append(dict(role=name,pid=pid,state=f[2],same_live=f[21]==str(r[tickkey]) and f[2]!='Z'))
        except (FileNotFoundError,ProcessLookupError):out.append(dict(role=name,pid=pid,same_live=False))
    return out

def main(seed):
    prefix=Path('/25085400120' if seed==1 else '/24085403014');exp=prefix/'DEIM/experiments'
    names=dict(original_stem='original_stem12_20261011_v1',native_stem_scratch='native_stem_scratch12_20261011_v2')
    for name in ('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample'):names[name]='stem_component12_20261011_v1'
    out=dict(schema='stem_live_training_process_read_v1',seed=seed,observed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs={},
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,scope='Existing complete log/audit lines and sampled post-collate histograms. Samples are not all training batches; histogram equality is not tensor identity or full trajectory equality.')
    for name,folder in names.items():
        screen=exp/folder/f'seed{seed}_{name}';run=screen/'run'
        item=dict(processes=processes(screen),screen_exists=screen.exists())
        for key,file in dict(logs='log.txt',updates='training_audit_epochs.jsonl',target_sizes='sampled_training_target_sizes.jsonl').items():item[key]=lines(run/file)
        if (screen/'bundle/receipt.json').exists():item['configuration']=config(screen/'bundle')
        if (run/'runtime_start.json').exists():
            r=read(run/'runtime_start.json');item['runtime_start']=dict(identity={k:r['identity'][k] for k in ('arm','seed','protocol_sha256','common_init_sha256')},optimizer_coverage=r['optimizer_coverage'])
        out['jobs'][name]=item
    receipt=read(exp/names['original_stem']/f'seed{seed}_original_stem/bundle/receipt.json')
    legacy=Path(receipt['paths']['legacy_bundle']);lr=read(legacy/'receipt.json')
    assert lr['identity']['seed']==seed and lr['identity']['arm']=='baseline'
    base=Path(lr['paths']['output']);out['baseline']={key:lines(base/file) for key,file in dict(logs='log.txt',updates='training_audit_epochs.jsonl',target_sizes='sampled_training_target_sizes.jsonl').items()}
    out['baseline']['configuration']=config(legacy)
    out['baseline']['updates']['rows']=[x for x in out['baseline']['updates']['rows'] if x['epoch_zero_based']<12]
    out['baseline']['receipt_sha256']=sha((legacy/'receipt.json').read_bytes())
    assert 'torch' not in sys.modules
    text=base64.b64encode(json.dumps(out,allow_nan=False).encode()).decode()
    print('B64_BEGIN');print('\n'.join(text[i:i+64] for i in range(0,len(text),64)));print('B64_END')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();main(a.seed)
