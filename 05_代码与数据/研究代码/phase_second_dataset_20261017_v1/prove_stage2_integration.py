"""CPU integration check against the actual frozen native solver; in-memory checkpoint capture only."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
account=sys.argv[1]
storage=Path('/25085400120' if account=='A' else '/24085403014')
root=storage/('DEIM/experiments' if account=='A' else 'experiments')/'phase_second_dataset_tinyperson_20261017_v1'
sys.path.insert(0,str(root/'code'));os.chdir(root/'code')
import torch
assert not torch.cuda.is_available()
from tiny_protocol import install_native_adapter,TinyPersonEvaluator
install_native_adapter()
from engine.solver import det_solver as native
from tiny_stage2 import install_stage2_checkpointing

class EMAFixture:
    def __init__(self):self.decay=.999;self.marker=42
    def state_dict(self):return {'marker':self.marker}
    def load_state_dict(self,state):self.marker=state['marker']

def refresh(solver):
    solver.last_epoch=30;solver.ema.marker=7;solver._tiny_stage2_best_stat=None
    return True

native.DetSolver._load_best_stg1_if_available=refresh
install_stage2_checkpointing()
saved=[]
native.dist_utils.save_on_master=lambda state,path:saved.append(copy.deepcopy(state))
s=native.DetSolver.__new__(native.DetSolver)
s.writer=None;s.output_dir=Path('/tmp/unused_tiny_integration_capture')
s.model=torch.nn.Linear(1,1);s.ema=EMAFixture()
s.criterion=SimpleNamespace(reset_model=lambda:None,set_model=lambda model:None)
s.train_dataloader=SimpleNamespace(dataset=SimpleNamespace(category2name={0:'person'},remap_mscoco_category=False),
                                  collate_fn=SimpleNamespace(stop_epoch=84))
s.evaluator=TinyPersonEvaluator.__new__(TinyPersonEvaluator)
s._is_pruned=False;s.afss=None;s.metric_lr_reducer=None;s.swanlab_run_id=None
s.best_stat={'epoch':30,'avg_metric':.9,'tiny_eval_bbox':.9};s.last_epoch=84
s.save_best_model({'tiny_eval_bbox':[.4]+[.99]*41},s.best_stat,84)
assert len(saved)==1 and saved[0]['last_epoch']==84 and saved[0]['ema']['marker']==42
assert saved[0]['best_stat']['tiny_eval_bbox']==.4
assert s.last_epoch==84 and s.ema.marker==7 and s._tiny_stage2_best_stat['epoch']==84
s.last_epoch=85;s.ema.marker=50
s.save_best_model({'tiny_eval_bbox':[.45]+[.99]*41},s.best_stat,85)
assert saved[-1]['last_epoch']==85 and saved[-1]['ema']['marker']==50
state=s.state_dict();s._tiny_stage2_best_stat=None;s.last_epoch=10;s.load_state_dict(state)
assert s.last_epoch==85 and s._tiny_stage2_best_stat['avg_metric']==.45
s.last_epoch=86;s.ema.marker=70
s.save_best_model({'tiny_eval_bbox':[.95]+[.99]*41},s.best_stat,86)
assert len(saved)==4 and saved[-1]['last_epoch']==86 and saved[-1]['best_stat']['epoch']==86
assert saved[-1]['ema']['marker']==70 and saved[-1]['tiny_stage2_best_stat']['avg_metric']==.95
print(json.dumps({'account':account,'actual_native_solver_integration_passed':True,
                  'cases':['snapshot_before_epoch84_refresh','native_refresh_keeps_real_clock','selection_resume','native_global_save_matches_selected_stage2'],
                  'GPU_work_started':False,'training_started':False,'checkpoint_files_written':0}))
