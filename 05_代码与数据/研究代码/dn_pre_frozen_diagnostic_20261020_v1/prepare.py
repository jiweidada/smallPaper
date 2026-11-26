"""Lock assets and batch selection before the new derivative diagnostic."""
import datetime, hashlib, json, pathlib, tarfile

CODE = pathlib.Path(__file__).resolve().parent
WORK = CODE.parents[1]
OLD = WORK / 'data/08_匹配聚合覆盖探针_20261019_v1'
OUT = WORK / 'data/09_DN初始化冻结诊断_20261020_v1'

def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    plan = json.loads((OLD/'sample_plan.json').read_text(encoding='utf-8-sig'))
    manifest = json.loads((OLD/'shared_inputs/inputs/manifest.json').read_text(encoding='utf-8-sig'))
    batches = sorted((v for v in manifest['batches'] if v['group']=='development'), key=lambda v:v['batch'])[:2]
    assert [v['batch'] for v in batches] == [0, 1]
    for row in batches:
        assert sha(OLD/'shared_inputs/inputs'/row['file']) == row['sha256']
    for seed, row in plan['checkpoints'].items(): assert sha(row['path']) == row['sha256']
    package = json.loads((OLD/'package_manifest.json').read_text(encoding='utf-8-sig'))
    expected = {v['relative']:v['sha256'] for v in package['files']}
    copied=[]
    with tarfile.open(OLD/'probe_package.tar.gz','r:gz') as archive:
        for member in archive:
            if not (member.isfile() and member.name.startswith('native/engine/') and member.name.endswith('.py')): continue
            dest=CODE/member.name
            assert dest.resolve().is_relative_to(CODE.resolve())
            data=archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest()==expected[member.name],member.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(data)
            copied.append({'relative':member.name,'sha256':sha(dest)})
    protocol = {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization':'User: two seeds, two development batches each, original matching/loss; frozen diagnostic; no new training and no impact to other tasks.',
        'selection':'first two development batches by batch ID, selected before new derivative/counterfactual outcomes; previous development analyses make this exploratory, not heldout confirmation',
        'seeds':[1,2], 'batches':batches, 'checkpoints':plan['checkpoints'],
        'weight_state':'online raw model, last_epoch=23, not EMA',
        'device':'local CPU only; CUDA_VISIBLE_DEVICES empty; <=2 threads, below-normal priority, two-CPU affinity when supported',
        'precision':'FP32 native forward/loss/parameter gradients; float64 cached-head loss evaluation of the resulting FP32 virtual weights',
        'forward':'training outputs with DN; restore model buffers and RNG after each batch; no augmentation regeneration',
        'dn_noise':'original stored per-batch integer DN seed, freshly generated on CPU; not bitwise replay of historical CUDA random draws; same CPU noise across model seeds',
        'ordinary_pre_loss':'native criterion boxes pre, indices_go/num_boxes_go at use_uni_set=True; no deduplication',
        'dn_pre_loss':'native criterion boxes dn_pre, indices_dn/(num_boxes*dn_num_group)',
        'weights':{'L1':5,'GIoU':2}, 'epoch_argument':24,
        'fixed_counterfactual_matching':'freeze original native criterion pre regression indices, normalization and cached hidden/input references from this diagnostic forward',
        'update_scope':'only pre_bbox_head.layers[-1].weight and bias; all other parameters held fixed for evaluation',
        'derivatives':'capture full native total-loss gradient norm to apply original global clip_max_norm=0.1; remove only DN pre gradient at the last layer, preserve all other gradients when recomputing counterfactual norm',
        'virtual_updates':'functional cloned AdamW from the same checkpoint state for include/exclude current DN last-layer gradient; no optimizer.step, model parameter writeback or checkpoint modification',
        'directions':'batch00->batch01 and batch01->batch00 for each seed; same-batch descriptive only',
        'primary_effect':'ordinary pre loss after include-DN virtual update minus loss after exclude-DN update; >0 means locally harmful on the other batch',
        'references':'analytic cross-batch gradient interactions, ordinary-self direction and counterfactual decrease with historical Adam momentum explicitly retained',
        'hard_group':'ordinary input-reference matched IoU<0.5, determined before the pre head; original full num_boxes_go retained for this subset contribution',
        'decision':'If effects are weak or inconsistent across directions/seeds, stop prioritizing this explanation. Positive local results only qualify further mechanism checks; no detection benefit, long-term harm, exclusivity of this path, or superiority of preserving upstream gradients is established.',
        'no_numeric_gate_adopted':True, 'no_parameter_training_steps':True,
        'heldout_prediction_reads':False,'remote_operations':False,
        'matching_and_native_source_identity':copied,
        'source_plan_sha256':sha(OLD/'sample_plan.json'),
        'recipe_sha256':sha(OLD/'baseline_recipes.json'),
        'input_manifest_sha256':sha(OLD/'shared_inputs/inputs/manifest.json'),
        'prepare_script_sha256':sha(__file__),
    }
    with (OUT/'01_锁定资产与诊断口径.json').open('x',encoding='utf-8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
    print(json.dumps({'native_sources':len(copied),'selected_batches':[v['batch'] for v in batches],'checkpoint_hashes_verified':2,'protocol_sha256':sha(OUT/'01_锁定资产与诊断口径.json')}))

if __name__=='__main__':main()
