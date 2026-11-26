from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,sys
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=read(B/'A37_局部差分编码与语义判别证据核查_v1'/'semantic_probe_sample_FROZEN.json')
verified=read(D/'independent_support_verification.json')
assert verified['primary_val_pairs']==78 and verified['data_probe_fits']==0 and not verified['semantic_AUC_estimated']
assert not (D/'analysis'/'probe_fit_entries.jsonl').exists()
seed=old['seed']
def rank(value):return hashlib.sha256((seed+'|'+value).encode()).hexdigest()
old_by_name={r['file']:r for r in old['cases']}
train_prefixes={r['prefix'] for r in old['cases'] if r['split']=='train'}
old_val_excluded=set(old['exclusions']['old_val_prefixes'])
prior_train=set()
for folder in ['A33_固定可见内容的采样相位诊断_v1','A34_相位干预剩余76图复核_v1']:
    prior_train.update(r['file_name'] for r in read(B/folder/'phase_protocol_FROZEN.json')['cases'])
root=Path(r'D:\数据集\visdrone2019');pools={}
for split in ['train','val']:
    groups=defaultdict(list)
    for p in (root/('VisDrone2019-DET-'+split)/'images').glob('*.jpg'):
        prefix=p.stem.split('_')[0]
        if split=='train' and (prefix not in train_prefixes or p.name in prior_train):continue
        if split=='val' and prefix in old_val_excluded:continue
        groups[prefix].append(p)
    pools[split]=groups
assert len(pools['train'])==64 and len(pools['val'])==64
rows=[];new_count=0
for split in ['train','val']:
    for prefix in sorted(pools[split],key=rank):
        for frame_rank,p in enumerate(sorted(pools[split][prefix],key=lambda p:rank(p.name))[:2]):
            annotation=p.parent.parent/'annotations'/(p.stem+'.txt')
            with Image.open(p) as image:width,height=image.size
            source=old_by_name.get(p.name)
            r=dict(prospective_index=len(rows),split=split,prefix=prefix,frame_rank_in_group=frame_rank,
                file=p.name,image_path=str(p),annotation_path=str(annotation),width=width,height=height,
                image_sha256=sha(p),annotation_sha256=sha(annotation),reuse_A38_capture=source is not None)
            if source is not None:
                assert r['image_sha256']==source['image_sha256'] and frame_rank==0
                r.update(A38_index=source['index'],A38_capture=str(D/'downloaded_evidence'/'captures'/('case_%03d.npz'%source['index'])))
            else:r['new_capture_ordinal']=new_count;new_count+=1
            rows.append(r)
assert {r['file'] for r in old['cases']}<={r['file'] for r in rows}
assert not ({r['prefix'] for r in rows if r['split']=='train'}&{r['prefix'] for r in rows if r['split']=='val'})
counts={s:sum(r['split']==s for r in rows) for s in ['train','val']}
plan=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_SAMPLING_DRAFT_NOT_AN_EXECUTION_ALLOCATION',
    reason='A38 passed data collection but only78primaryvalpairs versus preregistered100;no data probes or semanticAUC were fitted/estimated.',
    decision_type='Count-only enrollment feasibility adjustment before any comparison of the two representations,not performance-guided resampling.',
    selection='Keep64trainprefixes;useall64eligiblevalprefixes;uptotwoSHA256-rankedimagesperprefixunderexistingA37seed,withsingletonsgivenone. Labels and feature/probe scores do not select frames.',
    seed=seed,cases=rows,case_counts=counts,prefix_counts={'train':64,'val':64},reused_A38_cases=112,
    prospective_new_unique_images=new_count,possible_new_whole_detector_cap=new_count+1,
    possible_repeat='One predetermined firstNEWcase repeatedafternewcollection to checkfullqueryfeatureidentity; no rerun of old112images.',
    inference_unit='Bootstrap64validationfilename-prefixgroupswithalltheirframes,notindividualframes;prefix is a proxy,not verified flight identity.',
    retained_scientific_rules=['Samefine/query/contrast/geometry/RFFviews','SamepredictionboxROI','Sameforeground/backgroundandnative-supportdefinitions',
        'Samefixedlambda1and7datafitcap','SameprimaryAUCgapandadvancementthresholds','Keepminimum100primarypairsand20contributingvalgroups'],
    additional_justification='Additionaltrainframesstabilizethe512Dbasiscontrol;no fitted-performance result was available when selecting this plan.',
    required_before_execution=['Newphase/protocolandexactcapmustbefrozen','Clusterbootstrapdrawsandcodefixed','OldA38114capstaysclosed','Noautomaticfurtherexpansionifnewcoveragefails'],
    new_model_call_allocation=0,new_data_probe_fit_allocation=0,new_detector_optimizer_update_allocation=0,
    mainline_validated=False,new_algorithm_approved=None)
with (D/'coverage_followup_sample_DRAFT.json').open('x',encoding='utf-8') as output:json.dump(plan,output,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=plan['status'],case_counts=counts,reused=112,new_unique_images=new_count,
    possible_later_cap=new_count+1,allocated_now=0),ensure_ascii=False))
