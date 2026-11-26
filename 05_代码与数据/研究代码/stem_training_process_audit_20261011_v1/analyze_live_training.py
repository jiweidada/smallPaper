"""Compare actual logged updates, selected loss fields and sampled target sizes."""
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,math,sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    with Path(p).open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def reader():
    path=ROOT/'英文小论文/code/original_stem12_monitor_20261011_v1/verify_terminal_local.py'
    spec=importlib.util.spec_from_file_location('stem_process_BASE_source',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,m.local_protocol()
def sized(rows):
    seen={}
    for r in rows:
        e,i=r['epoch_zero_based'],r['epoch_attempt'];key=(e,i)
        assert e in (0,4,8,11) and i in (1,101,501,1001,1501) and key not in seen
        assert r['bands']==['under2','2_to_under4','4_to_under8','8_to_under16','16_to_under32','ge32']
        h=r['target_histogram'];assert len(h)==6 and all(type(x) is int and x>=0 for x in h) and sum(h)==r['target_exposures']
        s=r['samples_shape'];assert len(s)==4 and s[:2]==[4,3] and s[2]==s[3] and s[2]>0 and s[2]%32==0
        seen[key]=r
    return seen
def aggregate(rows):
    result={}
    for e in sorted({r['epoch_zero_based'] for r in rows}):
        r=[x for x in rows if x['epoch_zero_based']==e];hist=[sum(x['target_histogram'][j] for x in r) for j in range(6)];total=sum(hist)
        result[str(e+1)]=dict(sampled_batches=len(r),target_exposures=total,histogram=hist,under4_fraction=(hist[0]+hist[1])/total if total else None,
            scope='Only fixed sampled positions from this epoch. Not all-batch prevalence, same image cohort, canonical annotation count, or per-image query capacity.')
    return result

def analyze(snapshot,output):
    o=read(snapshot);m,p=reader();summaries={};rows=[];histpairs=[];baselines={}
    for seed in (1,2):
        s=o['seed'+str(seed)];assert s['seed']==seed and s['schema']=='stem_live_training_process_read_v1'
        ev=p['reused_local_evidence'][str(seed)];packet=ROOT/ev['packet_relative_path']
        assert sha(ROOT/ev['report_relative_path'])==ev['report_sha256'] and sha(packet/'manifest.json')==ev['manifest_sha256']
        m.reuse_sources(read(ROOT/ev['report_relative_path'])['identity'],p)
        b=s['baseline'];path=packet/'reference/baseline_log.txt';entry=read(packet/'manifest.json')['files']['reference/baseline_log.txt']
        assert sha(path)==entry['sha256']==b['logs']['sha256'] and path.stat().st_size==b['logs']['bytes']==entry['bytes']
        log=[json.loads(x) for x in path.read_text().splitlines()];assert [x['epoch'] for x in log]==list(range(24))
        assert b['logs']['source_complete_line_count']==24 and not b['logs']['incomplete_last_line']
        audit=b['updates']['rows'];assert [x['epoch_zero_based'] for x in audit]==list(range(12))
        assert all(x['attempts_total']==(i+1)*1617 and x['EMA_updates']==x['attempts_total'] and 0<x['actual_optimizer_steps_total']<=x['attempts_total'] for i,x in enumerate(audit))
        cfg=b['configuration'];assert cfg['selected_values']['DEIMCriterion']['losses']==['mal','boxes','local']
        baselines[seed]=packet
        full=sized(s['jobs']['original_stem']['target_sizes']['rows']);sizes={}
        for arm,item in s['jobs'].items():
            assert item['configuration']['selected_values']==cfg['selected_values']
            assert item['configuration']['optimizer_configuration_sha256']==cfg['optimizer_configuration_sha256']
            candidate=item['logs']['rows'];updates=item['updates']['rows']
            assert [r['epoch'] for r in candidate]==list(range(len(candidate))) and len(candidate)<=12
            assert [r['epoch_zero_based'] for r in updates]==list(range(len(updates))) and len(updates)<=12
            for e,r in enumerate(updates):
                assert r['attempts_total']==(e+1)*1617 and r['EMA_updates']==r['attempts_total']
                assert 0<r['actual_optimizer_steps_total']<=r['attempts_total']
                assert r['attempts_without_optimizer_step']==r['attempts_total']-r['actual_optimizer_steps_total']
            for e,r in enumerate(candidate):
                assert len(r['test_coco_eval_bbox'])==12 and all(math.isfinite(v) for v in r['test_coco_eval_bbox'])
                l=log[e];assert r['train_lr']==l['train_lr']
                delta={k:r[k]-l[k] for k in r if k.startswith('train_loss') and k in l}
                v=r['test_coco_eval_bbox'];base=l['test_coco_eval_bbox']
                row=dict(seed=seed,arm=arm,epoch=e+1,AP=v[0]*100,APs=v[3]*100,BASE_AP=base[0]*100,BASE_APs=base[3]*100,
                    delta_BASE_AP_pp=(v[0]-base[0])*100,delta_BASE_APs_pp=(v[3]-base[3])*100,
                    train_loss_mal=r['train_loss_mal'],BASE_train_loss_mal=l['train_loss_mal'],
                    train_loss_bbox=r['train_loss_bbox'],BASE_train_loss_bbox=l['train_loss_bbox'],
                    train_loss_giou=r['train_loss_giou'],BASE_train_loss_giou=l['train_loss_giou'],
                    actual_steps=None,BASE_actual_steps=audit[e]['actual_optimizer_steps_total'],step_difference=None,selected_loss_deltas=delta)
                if e<len(updates):
                    row['actual_steps']=updates[e]['actual_optimizer_steps_total'];row['step_difference']=row['actual_steps']-row['BASE_actual_steps']
                rows.append(row)
            current=sized(item['target_sizes']['rows']);sizes[arm]=aggregate(item['target_sizes']['rows'])
            if arm!='original_stem':
                comparisons=[]
                for key in sorted(set(current)&set(full)):
                    x,y=current[key],full[key];same=x['target_histogram']==y['target_histogram'] and x['samples_shape']==y['samples_shape']
                    comparisons.append(dict(seed=seed,arm=arm,epoch=key[0]+1,epoch_attempt=key[1],histogram_and_input_shape_equal=same,
                        full_histogram=y['target_histogram'],arm_histogram=x['target_histogram'],full_shape=y['samples_shape'],arm_shape=x['samples_shape']))
                histpairs.extend(comparisons)
        summaries[str(seed)]=dict(observed_remote_UTC=s['observed_remote_UTC'],baseline_sampled_size_history_exists=b['target_sizes']['exists'],
            baseline_log_sha256=entry['sha256'],baseline_updates_read_sha256=b['updates']['sha256'],all_selected_training_configurations_equal=True,
            sampled_sizes=sizes,live_processes={arm:item['processes'] for arm,item in s['jobs'].items()})
    criterion=baselines[1]/'native/engine/deim/deim_criterion.py'
    assert sha(criterion)=='16cdd80b2197b8def8ee3f5f19f6311f5221c3da08abacb99f50d346610a2c30'
    functions={n.name:n for n in ast.walk(ast.parse(criterion.read_text(encoding='utf8'))) if isinstance(n,ast.FunctionDef)}
    f=ast.unparse(functions['loss_labels_mal'])
    assert 'target_score = target_score.pow(self.gamma)' in f and 'ious = torch.diag(ious).detach()' in f
    gamma=o['seed1']['baseline']['configuration']['selected_values']['DEIMCriterion']['gamma']
    toy=[]
    for iou in (.2,.5):
        y=iou**gamma;h=-y*math.log(y)-(1-y)*math.log(1-y);toy.append(dict(IoU=iou,positive_soft_target=y,minimum_single_positive_BCE=h))
    assert toy[0]['minimum_single_positive_BCE']<toy[1]['minimum_single_positive_BCE']
    latest=[r for r in rows if r['epoch']==max(x['epoch'] for x in rows if x['seed']==r['seed'] and x['arm']==r['arm'])]
    out=dict(status='PROVISIONAL_REAL_LOG_UPDATE_AND_SAMPLED_SIZE_AUDIT',summaries=summaries,rows=rows,latest_rows=latest,histogram_pair_checks=histpairs,
        matching_sampled_histogram_shape_pairs=sum(x['histogram_and_input_shape_equal'] for x in histpairs),compared_sampled_pairs=len(histpairs),
        MAL=dict(criterion_source_sha256=sha(criterion),gamma=gamma,positive_soft_target='matched IoU ** gamma; default path uses detached IoU',
            toy=toy,interpretation='Lower logged MAL is not evidence of better class discrimination or AP. This scalar soft-target example does not establish the cause of any observed loss/AP difference.'),
        snapshot_sha256=sha(snapshot),analysis_code_sha256=sha(__file__),collector_source_sha256=sha(HERE/'read_live_training.py'),
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,full_input_trajectory_equivalence_proven=False,
        causal_root_proven=False,fixed12_saved_EMA_verified=False,
        limitations='Live file snapshots can be incomplete at their boundaries. Update counters do not diagnose why an attempt skipped its optimizer step. Equal sampled histograms/shapes do not prove equal input tensors, GT identities, augmentations or gradients. Historical BASE has no saved sampled-size history here; its target sizes cannot be compared from this snapshot. Existing first100 identity proofs cover only those first100. Every efficacy decision remains at independently verified fixed12 EMA.')
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    write(output,out)
    with Path(output).with_suffix('.csv').open('x',encoding='utf-8-sig',newline='') as f:
        flat=[{k:v for k,v in r.items() if k!='selected_loss_deltas'} for r in rows];w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(dict(status=out['status'],matching_size_pairs=out['matching_sampled_histogram_shape_pairs'],compared_size_pairs=len(histpairs),latest_rows=latest),ensure_ascii=False))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.snapshot,a.output)
