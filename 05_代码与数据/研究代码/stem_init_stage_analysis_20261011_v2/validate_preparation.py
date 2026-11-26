"""Check existing BASE cache joins and count partitions; certify no new endpoint."""
from pathlib import Path
import argparse,sys
import numpy as np
from compare_saved_stages import HERE,PROTOCOL,prepared,cached_baseline,groups,read,sha,write

def validate(output):
    p,g,onset,rank,readers=prepared();checks=[]
    for seed in (1,2):
        base=cached_baseline(seed,p,readers['full']);masks=groups(base['annotation'],base['values'])
        original=base['original_stage_report']['groups']
        for name,mask in masks.items():
            for t,ti in ((.5,0),(.75,5)):
                current=onset(base['values'],mask,t,ti)
                assert current==original[name][str(t)]['baseline']
        # Only arithmetic on already saved arrays; no ranking model or evaluator.
        counts=base['rank_report']['groups']
        for name in ('all','COCO_small','input_under4','input_4_to_under8'):
            for tag in ('50','75'):assert sum(counts[name][tag]['cells'].values())==int(masks[name].sum())
        checks.append(dict(seed=seed,GT=len(base['values']['annotation_id']),stage_onset_matches_saved_report=True,
            native_flags_joined_by_GT_ID=True,ranking_partition_totals_exact=True,baseline_AP_percent=base['summary']['overall_percent']))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    out=dict(status='PASS_EXISTING_BASE_CACHES_STAGE_COUNTS_AND_IDENTITY_JOINS',checks=checks,
        protocol_sha256=sha(PROTOCOL),new_terminal_packets_analyzed=0,new_model_forwards=0,new_evaluator_calls=0,
        limitation='Preparation checks only. Actual new full/scratch/control fixed12 stage analysis, ranking score checks and any efficacy conclusion remain pending.')
    write(output,out);print(out['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();validate(a.output)
