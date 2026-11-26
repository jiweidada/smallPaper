"""Actual native output-path decision and full-state resume, tiny disposable CPU fixture."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from uavdt_protocol import install_native_adapter
from uavdt_stage2 import install_stage2_checkpointing
install_native_adapter();install_stage2_checkpointing()
from engine.solver.det_solver import DetSolver
from train_uavdt import validate_paths

class EMA:
    def __init__(self,model):self.module=copy.deepcopy(model);self.updates=7
    def to(self,device):self.module.to(device);return self
    def state_dict(self):return {'module':self.module.state_dict(),'updates':self.updates}
    def load_state_dict(self,state):self.module.load_state_dict(state['module']);self.updates=state['updates']

def solver(output,resume=None):
    model=torch.nn.Linear(2,2)
    cfg=SimpleNamespace(device='cpu',model=model,tuning=None,sync_bn=False,find_unused_parameters=False,
                        criterion=torch.nn.Identity(),postprocessor=torch.nn.Identity(),ema=EMA(model),
                        scaler=torch.amp.GradScaler('cpu',init_scale=1024.),last_epoch=-1,output_dir=str(output),resume=resume,writer=None)
    s=DetSolver.__new__(DetSolver);s.cfg=cfg;s._is_pruned=False;s.afss=None;s.metric_lr_reducer=None;s.swanlab_run_id=None
    s._setup();s.train_dataloader=SimpleNamespace(dataset=SimpleNamespace(category2name={0:'vehicle'},remap_mscoco_category=False))
    s.optimizer=torch.optim.AdamW(s.model.parameters(),lr=1e-4)
    s.lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(s.optimizer,milestones=[2],gamma=.1)
    s.best_stat={'epoch':84,'avg_metric':.4,'uavdt_eval_bbox':.4};s._uavdt_stage2_best_stat=copy.deepcopy(s.best_stat)
    return s

root=Path(sys.argv[1]);scratch=root/'artifacts/runtime_contract';scratch.mkdir(exist_ok=False)
fresh=solver(scratch/'fresh_c');assert fresh.output_dir.resolve()==(scratch/'fresh_c').resolve()
for step in range(3):
    fresh.optimizer.zero_grad();loss=fresh.model(torch.ones(2,2)).square().sum()
    fresh.scaler.scale(loss).backward();fresh.scaler.step(fresh.optimizer);fresh.scaler.update();fresh.lr_scheduler.step()
fresh.last_epoch=84;state=fresh.state_dict();assert state['optimizer']['state'] and state['scaler']['scale']==1024.
path=fresh.output_dir/'last.pth';torch.save(state,path)
restored=solver(scratch/'deliberately_wrong_output',str(path));assert restored.output_dir.resolve()==fresh.output_dir.resolve()
restored.load_resume_state(str(path))
assert restored.last_epoch==84 and restored._uavdt_stage2_best_stat['epoch']==84
assert restored.scaler.state_dict()==fresh.scaler.state_dict()
assert restored.lr_scheduler.state_dict()==fresh.lr_scheduler.state_dict()
assert restored.ema.updates==7
for k,v in fresh.model.state_dict().items():torch.testing.assert_close(restored.model.state_dict()[k],v,rtol=0,atol=0)
for index,state in fresh.optimizer.state_dict()['state'].items():
    for k,v in state.items():torch.testing.assert_close(restored.optimizer.state_dict()['state'][index][k],v,rtol=0,atol=0)
args=SimpleNamespace(config=str(root/'code/configs/experiments/uavdt/c_mean_full96.yml'),seed=0,
                     output_dir=str(root/'outputs/uavdt_c_mean_full96_seed0'),resume=str(root/'outputs/uavdt_c_full96_seed0/last.pth'))
try:validate_paths(args)
except AssertionError:pass
else:raise AssertionError('Cross-job resume path was not rejected')
result={'state':'passed','CPU_fixture_only':True,'native_fresh_path_verified':True,'native_resume_parent_override_verified':True,
        'cross_job_resume_rejected':True,'model_optimizer_scheduler_AMP_EMA_epoch_stage2_roundtrip_verified':True,
        'formal_training_updates':0}
with (root/'runtime_contract_receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
