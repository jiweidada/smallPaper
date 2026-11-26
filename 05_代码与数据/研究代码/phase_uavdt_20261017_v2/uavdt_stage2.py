"""Select the actual AP70-evaluated EMA in epochs84..95, preserving native training state."""
import copy
import math

def choose_stage2(previous,stats,epoch,stop_epoch=84):
    if epoch<stop_epoch:return previous
    score=float(stats['uavdt_eval_bbox'][0])
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid native UAVDT AP70')
    if previous is not None and score<=previous['avg_metric']:return previous
    return {'epoch':epoch,'avg_metric':score,'uavdt_eval_bbox':score,'selection_metric':'AP70','selection_window':[stop_epoch,95]}

def stage2_snapshot(state,selected,epoch):
    assert state['last_epoch']==epoch==selected['epoch']
    result=dict(state);result['best_stat']=copy.deepcopy(selected);result['uavdt_stage2_best_stat']=copy.deepcopy(selected)
    return result

def preserve_clock_during_refresh(solver,refresh):
    epoch=solver.last_epoch;selected=copy.deepcopy(getattr(solver,'_uavdt_stage2_best_stat',None))
    try:return refresh(solver)
    finally:solver.last_epoch=epoch;solver._uavdt_stage2_best_stat=selected

def install_stage2_checkpointing():
    from engine.solver import det_solver as native
    from uavdt_protocol import UAVDTEvaluator
    cls=native.DetSolver
    if getattr(cls,'_uavdt_stage2_installed',False):return
    old_save,old_state,old_load=cls.save_best_model,cls.state_dict,cls.load_state_dict
    old_refresh=cls._load_best_stg1_if_available
    def state_dict(self):
        state=old_state(self);state['uavdt_stage2_best_stat']=copy.deepcopy(getattr(self,'_uavdt_stage2_best_stat',None));return state
    def load_state_dict(self,state):
        result=old_load(self,state);self._uavdt_stage2_best_stat=copy.deepcopy(state.get('uavdt_stage2_best_stat'));return result
    def save_best_model(self,stats,best,epoch):
        if isinstance(getattr(self,'evaluator',None),UAVDTEvaluator):
            stop=self.train_dataloader.collate_fn.stop_epoch;previous=getattr(self,'_uavdt_stage2_best_stat',None)
            selected=choose_stage2(previous,stats,epoch,stop);self._uavdt_stage2_best_stat=selected
            if selected is not previous and self.output_dir:
                with native.ModelSaverFunc(self):native.dist_utils.save_on_master(stage2_snapshot(self.state_dict(),selected,epoch),self.output_dir/'best_stg2.pth')
        return old_save(self,stats,best,epoch)
    cls.state_dict,cls.load_state_dict,cls.save_best_model=state_dict,load_state_dict,save_best_model
    cls._load_best_stg1_if_available=lambda self:preserve_clock_during_refresh(self,old_refresh)
    cls._uavdt_stage2_installed=True
