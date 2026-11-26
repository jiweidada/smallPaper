"""Freeze local inputs and package; no model import, forward, training or SSH."""
import datetime,hashlib,json,pathlib,random,tarfile
HERE=pathlib.Path(__file__).resolve().parent
WORK=HERE.parents[1]
DEST=WORK/'data/08_匹配聚合覆盖探针_20261019_v1'
DEST.mkdir(exist_ok=True)
NATIVE=WORK/'code/local_candidates_20261004_v2/native'
OLD=WORK/'data/00_实验计划与证据索引/本次24轮双基线模型下载_20261009_v1'
ANN=pathlib.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    ann=json.loads(ANN.read_text(encoding='utf8'));ids=sorted(i['id'] for i in ann['images'])
    random.Random(20261019).shuffle(ids);pools={'development':ids[:len(ids)//2],'heldout':ids[len(ids)//2:]}
    batches=[]
    for group,pool in pools.items():
        for k in range(8):batches.append({'batch':len(batches),'group':group,'main_image_ids':pool[4*k:4*k+4],'augmentation_seed':19261019+len(batches),'forward_DN_seed':29261019+len(batches)})
    plan={'experiment_id':'matching_coverage_probe_20261019_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'User requested epoch-boundary pause, local backups, then bounded64-input probe only; no short training','formal_inputs':64,'batches':batches,'source_pools':pools,'annotation_sha256':sha(ANN),'augmentation_epoch':24,'checkpoints':{},'precision':'FP32, no AMP, TF32 disabled','model_mode':'training outputs, BN buffers restored each batch, parameters require_grad false','optimizer_steps':0,'maximum_checkpoints':2,'maximum_smoke_inputs_each':8,'primary_comparisons':['native_union','coverage_constrained','last_anchor_fill'],'additional_diagnostic':['last_hungarian','own_layer_hungarian'],'r_definition':'(V_native-V_candidate)/V_native; diagnostic only','quality_metrics':['per-image mean and lower-quartile matched IoU','L1/GIoU/FGL using original weights and K normalization','regression GT vs branch own classification GT mismatch rate','per-GT duplicate count'],'source_group_unit':'connected components of same-batch and shared augmentation sources; development/heldout sources disjoint','early_stop':'failure, invalid isolation, or resource preflight; no automatic experiment retry','no_sample_extension':True,'no_automatic_training':True}
    configs={}
    for seed in [1,2]:
        source=OLD/f'seed{seed}/run';args=json.loads((source/'args.json').read_text(encoding='utf8'));y=args['yaml_cfg'];configs[str(seed)]=y
        plan['checkpoints'][str(seed)]={'path':str(source/'last.pth'),'sha256':sha(source/'last.pth'),'args_sha256':sha(source/'args.json'),'expected_last_epoch':23,'state':'raw model, not EMA','recipe_epochs':132}
    for key in ['HGNetv2','HybridEncoder','DFINETransformer','DEIMCriterion']:
        assert configs['1'][key]==configs['2'][key]
    save(DEST/'sample_plan.json',plan)
    save(DEST/'baseline_recipes.json',configs)
    files=[]
    for p in sorted((NATIVE/'engine').rglob('*.py')):files.append((p,'native/'+p.relative_to(NATIVE).as_posix()))
    for p in HERE.glob('*.py'):files.append((p,'probe/'+p.name))
    files += [(DEST/'sample_plan.json','sample_plan.json'),(DEST/'baseline_recipes.json','baseline_recipes.json')]
    smoke=WORK/'data/00_实验计划与证据索引/完整增强输入容量与尺寸核查_20261008_v1/augseed810081'
    for k in range(8):
        for suffix in ['.png','_targets.npz']:
            p=smoke/(f'{k:03d}'+suffix);assert p.is_file();files.append((p,'smoke/'+p.name))
    records=[{'relative':rel,'bytes':p.stat().st_size,'sha256':sha(p)} for p,rel in files]
    save(DEST/'package_manifest.json',{'files':records,'native_sources':len([r for r in records if r['relative'].startswith('native/')]),'plan_sha256':sha(DEST/'sample_plan.json')})
    with tarfile.open(DEST/'probe_package.tar.gz','w:gz') as tar:
        for p,rel in files:tar.add(p,arcname=rel,recursive=False)
        tar.add(DEST/'package_manifest.json',arcname='package_manifest.json')
    print(json.dumps({'destination':str(DEST),'plan_sha256':sha(DEST/'sample_plan.json'),'package_bytes':(DEST/'probe_package.tar.gz').stat().st_size,'package_sha256':sha(DEST/'probe_package.tar.gz'),'files':len(files)},ensure_ascii=False))
if __name__=='__main__':main()
