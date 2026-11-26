"""Materialize the authorized orders and seal inputs before any B optimization."""
import ast, difflib, importlib, math
C = importlib.import_module('345_context')
import torch
T = importlib.import_module('train344')

def record(path):
    return {'path': str(path), 'sha256': C.sha(path), 'bytes': path.stat().st_size}

def main():
    a = C.read(C.HERE/'345_result_A合同总表_v2.json')
    assert len(a['contracts']) == 12
    flops = C.read(C.HERE/'345_result_FLOPs覆盖补足_v2.json')
    assert flops['all_8_match_previous_stdout'] and len(flops['rows']) == 8
    group_path = C.S5/'342_八run配置清单.json'
    group = C.read(group_path)
    seal = C.read(C.S5/'342_阶段5_交付封存SHA256.json')['artifacts']
    for p, item in seal.items():
        assert C.sha(p) == item['sha256'], ('stage5 sealed file changed',p)
    ids = T.load_order()
    assert len(ids)==3235 and len(set(ids))==3235
    shuffle = {}
    for seed in (1,2):
        config = next(r for r in group['runs'] if r['arm']=='N' and r['global_seed']==seed)
        epochs = []
        for epoch in range(config['epochs']):
            generator = torch.Generator(device='cpu').manual_seed(config['head_seed']+epoch)
            epochs.append([ids[j] for j in torch.randperm(len(ids), generator=generator).tolist()])
        formal_name = f'345_shuffle_development_seed{seed}.json'
        C.write(formal_name, {'kind':'development_6epoch','global_seed':seed,'head_seed':config['head_seed'],
                'algorithm':'CPU torch.Generator.manual_seed(head_seed + zero_based_epoch); torch.randperm over 327 development_ids order',
                'development_index_path':T.P327,'development_index_sha256':C.sha(T.P327),
                'torch_version':torch.__version__,'batch_size':config['batch'],'drop_last':config['drop_last'],
                'epochs':epochs})
        formal_path = C.HERE/formal_name
        mini_name = f'345_shuffle_mini_acceptance_seed{seed}.json'
        C.write(mini_name, {'kind':'mini_acceptance_only','global_seed':seed,'head_seed':config['head_seed'],
                'parent_formal_sha256':C.sha(formal_path),'batch_size':config['batch'],
                'epochs':[row[:2*config['batch']] for row in epochs],
                'restriction':'CPU acceptance only; formal loader must reject this document kind'})
        shuffle[str(seed)] = {'formal':record(formal_path),'mini':record(C.HERE/mini_name)}
    configs = {}
    for cfg in group['runs']:
        path = C.S5/f"342_预定配置_{cfg['arm']}_seed{cfg['global_seed']}.json"
        assert C.sha(path)==seal[str(path)]['sha256'] and C.read(path)['config']==cfg
        configs[cfg['run_id']] = record(path)
    # All math remains imported by identity. The diff is a launcher replacement,
    # not a copied and independently edited version of the loss or optimizer.
    old = (C.S344/'train344.py').read_text(encoding='utf-8')
    new = (C.HERE/'train345.py').read_text(encoding='utf-8')
    ast.parse(new)
    C.text('345_train344_to_train345.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
           fromfile=str(C.S344/'train344.py'),tofile=str(C.HERE/'train345.py'))))
    C.write('345_B_预定验收口径.json', {
        'authority':[str(C.SPEC/'345_授权_继承合同重跑与正式入口验收.txt'),str(C.SPEC/'345c_补充裁定_FLOPs写盘修正.txt')],
        'math_inheritance':'train345 calls train344.make_common_init/build_optimizer/compute_losses_340/load_entry/collate/lr_at/restore_rng by object identity; no new coordinate conversion',
        'shuffle':'per seed one six-epoch file, shared by all four arms; independent regeneration/permutation audit before optimization',
        'continuous_vs_resume':'all eight arm/seed combinations: six consecutive mini steps versus three + new-process resume to absolute endpoint six',
        'strict_comparison':['every model tensor byte/dtype/shape','all optimizer state including param_groups and lr','torch/numpy/python RNG','step/cursor/shuffle/initial binding','345_steps.json exact UTF-8 bytes'],
        'wall_times':'actual measured epoch wall times in separate 345_epoch_times.json; never claimed bit-identical',
        'formal':'eight 2-step runs through actual CLI, true 3235-window epoch order; image_ids and actual collated rgb/P tensor SHA must match for same seed across four arms',
        'negative':['configuration SHA','manifest SHA','shuffle bytes','existing run directory','max_steps=1219','formal mode rejects miniature order','CLI refuses unlisted hyperparameter'],
        'limits':{'mini_endpoint':6,'formal_endpoint':2,'hard_scientific_budget':1218,'real_training_steps':0},
        'planned_actual_optimizer_steps':8*(6+3+3+2),'planned_head_forwards':112,'planned_cache_loads':1792,
        '未执行':['B optimizations','真实训练','GPU'],'source_sha256':C.sources()})
    sources = C.sources()
    sources[str(C.SPEC/'345b_补充裁定_A5测试构造修正.txt')] = C.sha(C.SPEC/'345b_补充裁定_A5测试构造修正.txt')
    sources[str(C.SPEC/'345c_补充裁定_FLOPs写盘修正.txt')] = C.sha(C.SPEC/'345c_补充裁定_FLOPs写盘修正.txt')
    sources[str(T.P327)] = C.sha(T.P327)
    C.write('345_B_冻结入口绑定.json', {'configs':configs,'config_group':record(group_path),
            'stage5_seal':record(C.S5/'342_阶段5_交付封存SHA256.json'),
            'lr_curve':record(C.S5/'342_预定学习率曲线1218点.json'),'shuffle':shuffle,'sources':sources,
            'authorization':'CPU bounded acceptance only; real training 需另行授权'})
    bundle_sha = C.sha(C.HERE/'345_B_冻结入口绑定.json')
    C.text('345_bindings_anchor.py',f'"""Exclusive 345 launch trust anchor, see final delivery seal."""\nBUNDLE_SHA256 = {bundle_sha!r}\n')
    print('B orders, frozen config/source bundle and acceptance protocol materialized; optimizer steps=0',flush=True)

if __name__=='__main__':
    main()
