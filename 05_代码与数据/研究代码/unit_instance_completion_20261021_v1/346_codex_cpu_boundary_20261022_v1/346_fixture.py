"""Construct synthetic cursor fixtures from real untrained initial states."""
import importlib,random
C=importlib.import_module('346_context')
import numpy as np
import torch
E=C.E;T=C.T

def build_all():
    root=C.HERE/'346_fixtures'
    root.mkdir(exist_ok=False)
    records={}
    for seed in (1,2):
        for arm in 'NSGR':
            cfg,order,_,base_binding=E.validate_bindings(arm,seed,'formal')
            for case,spec in C.CASES.items():
                np.random.seed(cfg['head_seed']);random.seed(cfg['head_seed'])
                model=T.make_common_init(arm,seed).train()
                initial=E.state_sha(model.state_dict())
                assert initial==cfg['untrained_initial_state_sha256']
                optimizer=E.make_optimizer(model,cfg)
                assert not optimizer.state
                start=spec['start']
                epoch,batch_index=divmod(start,cfg['steps_per_epoch'])
                position=batch_index*cfg['batch']
                E.check_cursor(start,epoch,batch_index,order,cfg)
                binding=C.binding_for(base_binding,case)
                name=f'346_{arm}_seed{seed}_{case}'
                path=root/(name+'.pt')
                checkpoint={'format':C.KIND,'binding':binding,'initial_state_sha256':initial,
                    'model_state_sha256':initial,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'param_group_lr':[g['lr'] for g in optimizer.param_groups],
                    'step':start,'epoch':epoch,'batch_index':batch_index,'position_in_epoch':position,
                    'shuffle_sha256':binding['shuffle_sha256'],'rng':T.capture_rng(),
                    'step_logs':[],'epoch_times':[],'epoch_partial_seconds':0.0,
                    'synthetic_origin_step':start,'actual_optimizer_updates_since_fixture':0,
                    'real_training_steps':0,
                    'fixture_note':'Only the cursor is synthetic. Model/optimizer/RNG are actual untrained initial states; no prior batches or historical logs were executed or fabricated.'}
                with open(path,'xb') as f:torch.save(checkpoint,f)
                receipt_path=root/(name+'_receipt.json')
                E.write_json(receipt_path,{'checkpoint_path':str(path),'checkpoint_sha256':C.sha(path),
                    'binding':binding,'step':start,'epoch':epoch,'batch_index':batch_index,
                    'position_in_epoch':position,'execution_kind':C.KIND,'case':case,
                    'actual_optimizer_updates_since_fixture':0,'model_is_untrained_initial':True})
                records[f'{arm}_seed{seed}/{case}']={'path':str(receipt_path),'sha256':C.sha(receipt_path)}
    C.write('346_构造断点清单.json',{'execution_kind':C.KIND,'records':records,
            'actual_counts':{'models_constructed':16,'implicit_common_init_reference_models':16,
                             'optimizers_constructed':16,'head_forwards':0,'optimizer_updates':0},
            'note':'游标不代表已训练步数；16份断点全为未训练初态＋显式构造位置。',
            'source_sha256':C.source_sha()})
    print('346 fixtures: 16 untrained synthetic-cursor checkpoints sealed; optimizer updates=0',flush=True)
    return records
