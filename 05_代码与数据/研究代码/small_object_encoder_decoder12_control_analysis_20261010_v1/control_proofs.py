"""Bind downloaded controls to the original two verified fine runs on CPU."""
from pathlib import Path
from copy import deepcopy
import ast, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[3]
FINE = ROOT/'outputs/small_object_encoder_decoder12_analysis_20261010_v1'
CODE = ROOT/'英文小论文/code/small_object_encoder_decoder12_20261010_v5'
PLAN = ROOT/'outputs/small_object_encoder_decoder_20261010_v2/prospective_screen_plan.json'
PLAN_SHA = '061c395a28fa77f29262091945988a92f2d4f6b504acf21a6e9485413b0f7282'
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def fine_local(seed):
    index = read(CODE/'promotion/index.json'); entry = index['seeds'][str(seed)]
    parent = FINE/f'terminal_collection/seed{seed}'
    packet = parent/'extracted'; report_path = parent/'verification_report.json'
    assert sha(report_path) == entry['local_verification_sha256']
    report = read(report_path); done = read(packet/'run/training_completion.json')
    assert report['status'] == 'PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    assert report['seed'] == seed and report['files_verified'] == 2202
    assert report['archive_sha256'] == entry['terminal_archive']['sha256']
    assert report['all548_saved_weight_stage_arrays_exact'] and report['stage_geometry_TP_conflicts'] == 0
    assert report['native_gate_passed_this_seed']
    assert report['candidate_overall_percent']['APs'] > report['baseline_overall_percent']['APs']
    assert report['candidate_overall_percent']['AP'] >= report['baseline_overall_percent']['AP']
    assert done['identity']['arm'] == 'fine' and done['identity']['seed'] == seed
    assert done['completed_epochs'] == 12 and done['entered_epochs'] == list(range(12))
    assert sha(packet/'run/training_completion.json') == entry['training_completion_sha256']
    assert sha(packet/'candidate_reload_v1/verification.json') == entry['reload_verification_sha256']
    assert sha(packet/'run/last.pth') == entry['checkpoint_sha256'] == done['last_checkpoint_sha256']
    assert report['cost']['candidate_parameters'] == 13806625 and report['cost']['added_parameters'] == 3574628
    return packet, report, done, report_path


def comparable_config(config):
    config = deepcopy(config)
    config.pop('output_dir')
    variant = config['SmallObjectDEIM'].pop('variant')
    assert variant in ('fine', 'degraded')
    assert config['SmallObjectDEIM']['initialization_seed'] == 713
    assert config['epoches'] == 132 and config['flat_epoch'] == 64 and config['no_aug_epoch'] == 12
    assert config['train_dataloader']['total_batch_size'] == 4
    assert config['train_dataloader']['dataset']['transforms']['policy']['epoch'] == [4,64,120]
    assert config['resume'] is None and config['use_amp'] and config['use_ema']
    return config


def verify_promotion_packet(packet):
    packet = Path(packet); protocol = read(packet/'code/protocol.json')
    index_path = packet/'code/promotion/index.json'; index = read(index_path)
    receipt = read(packet/'prepared/receipt.json'); identity = receipt['identity']; seed = identity['seed']
    assert seed in (1,2) and identity['arm'] == 'degraded'
    assert sha(PLAN) == PLAN_SHA == protocol['scientific_plan_sha256'] == index['scientific_plan_sha256']
    assert sha(index_path) == sha(CODE/'promotion/index.json') == protocol['fine_promotion_sha256'] == identity['fine_promotion_sha256']
    assert index['status'] == 'BOTH_FINE_FIXED12_RELOAD_AND_LOCAL_VERIFICATION_PASS'
    proof = read(packet/'screen/dual_seed_fine_promotion.json')
    assert proof['both_same_seed_native_gates_passed'] and proof['both_saved_weight_reload_and_local_verification_passed']
    assert proof['protocol_sha256'] == sha(packet/'code/protocol.json') == identity['protocol_sha256']
    assert proof['fine_promotion_sha256'] == sha(index_path) and proof['automatic_retry'] is False
    fine_packets = {}
    for n in (1,2):
        local, report, done, report_path = fine_local(n); fine_packets[n] = local
        entry = index['seeds'][str(n)]
        for name,expected in [(f'seed{n}_verification_report.json',entry['local_verification_sha256']),
                              (f'seed{n}_training_completion.json',entry['training_completion_sha256']),
                              (f'seed{n}_reload_verification.json',entry['reload_verification_sha256'])]:
            assert sha(packet/'code/promotion'/name) == expected
        assert read(packet/'code/promotion'/f'seed{n}_verification_report.json') == report
    fine = fine_packets[seed]; original_identity = read(fine/'prepared/receipt.json')['identity']
    for key in ('common_init_sha256','train_ann_sha256','val_ann_sha256','baseline_checkpoint_sha256',
                'baseline_first100_sha256','image_manifest_sha256','evaluator_binding'):
        assert identity[key] == original_identity[key], key
    unchanged = []
    for name in ('small_object_model.py','geometry_memory.py','results.py','stage_trace.py',
                 'evaluator_binding.py','evaluator_sources.json','source_binding.json','baseline_eval.py',
                 'training_audit.py','supervise.py'):
        left=fine/'code'/name; right=packet/'code'/name
        assert sha(left)==original_identity['implementation'][name]
        assert sha(right)==identity['implementation'][name]
        if sha(left)==sha(right): unchanged.append(name); continue
        assert seed==2 and name in ('training_audit.py','supervise.py'),name
        a=left.read_text(encoding='utf8'); b=right.read_text(encoding='utf8')
        if name=='supervise.py':
            before="        phase(root,'baseline',[sys.executable,'-B',str(Path(__file__).with_name('baseline_eval.py')),'--bundle',a.bundle],1800)"
            after="        from reuse_baseline import attach_cached_baseline\n        attach_cached_baseline(receipt,plan,paths)"
            assert a.count(before)==1
            a=a.replace(before,after)
        trees=[ast.parse(s) for s in (a,b)]
        for tree in trees:
            assert isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str)
            tree.body.pop(0)
        assert ast.dump(trees[0],include_attributes=False)==ast.dump(trees[1],include_attributes=False),name
    assert comparable_config(read(packet/'prepared/resolved.yml')) == comparable_config(read(fine/'prepared/resolved.yml'))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return {'status':'PASS_ORIGINAL_DUAL_FINE_GATES_AND_EQUAL_CAPACITY_BINDING',
            'scientific_plan_sha256':PLAN_SHA,'promotion_sha256':sha(index_path),
            'seed':seed,'byte_identical_bound_files':unchanged,
            'approved_wrapper_only_differences':([] if seed==1 else ['training_audit.py:module docstring only','supervise.py:replace baseline GPU evaluation with verified cache attach; candidate phase unchanged']),
            'model_and_optimizer_training_behavior_equal':True,'resolved_config_equal_except_output_and_variant':True,
            'model_forwards':0,'optimizer_updates':0,'evaluator_calls':0,
            'claim_boundary':'Degraded control changes only the parameter-free spatial preprocessing of P2; this does not separately isolate geometry-query splitting, chart features or initial proposals.'}
