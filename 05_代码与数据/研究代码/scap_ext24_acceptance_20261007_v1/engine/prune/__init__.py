"""DEIM's self-contained structural-pruning API."""
  
import sys
     
from ..logger_module import get_logger
from . import torch_pruning as _tp     
    
# A bare ``import torch_pruning`` anywhere in the process must resolve to the  
# vendored copy so the official package cannot be mixed into the same model.     
_existing_tp = sys.modules.get("torch_pruning")    
if _existing_tp is not None and _existing_tp is not _tp:     
    get_logger(__name__).warning(     
        "检测到 sys.modules['torch_pruning'] 来自 %s，与内置剪枝运行时不一致；"   
        "已将其替换为 engine.prune.torch_pruning，请勿与官方包混用，否则会导致剪枝失效。",
        getattr(_existing_tp, "__file__", "<unknown>"),
    )    
sys.modules["torch_pruning"] = _tp  
del _existing_tp
 
from .checkpoint import (    
    checkpoint_state,
    load_checkpoint,  
    load_model_checkpoint,     
    load_pruned_model,    
    save_pruned_checkpoint,
) 
from .deim_pruner import ( 
    RepConvPruner,
    build_pruner,  
    example_inputs_from_batch,   
    ignored_layers_and_params, 
    model_stats,
    run_pruning,     
)     

__all__ = [
    "RepConvPruner",
    "build_pruner",     
    "checkpoint_state",
    "example_inputs_from_batch",
    "ignored_layers_and_params",
    "load_checkpoint",
    "load_model_checkpoint",
    "load_pruned_model",
    "model_stats",
    "run_pruning",     
    "save_pruned_checkpoint",   
]
