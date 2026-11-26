"""Small structural-pruning integration around the vendored Torch-Pruning runtime."""     

from __future__ import annotations 
   
import copy
from collections.abc import Iterable 
from typing import Any

import torch
from torch import nn

from ..logger_module import get_logger  
from ..misc.profiler_utils import count_model_stats     
from . import torch_pruning as tp 

_PRUNE_RETRY_ERRORS = (RuntimeError, IndexError, getattr(torch, "AcceleratorError", RuntimeError))

   
logger = get_logger(__name__) 
_SUPPORTED_METHODS = {"random", "l1", "lamp", "group_norm"}   
_UNSUPPORTED_METHODS = {"group_hessian", "group_taylor"}   
_IGNORE_CLASS_NAMES = {    
    "TransformerEncoderBlock",
    "MaxSigmoidAttnBlock", 
    "TextAdapter",  
    # CARAFE requires the predictor's fixed k*k mask channels; pruning this 
    # block's internal lpr_conv breaks that invariant.
    "LPRMAlignUpModule",
    # The standalone CARAFE implementation also predicts a fixed-size mask.
    "CARAFE",
}   
_STRUCTURAL_ERROR_FRAGMENTS = (
    "invalid for input of size",
    "shapes cannot be multiplied",     
    "shape mismatch",
    "size mismatch",  
    "sizes of tensors must match",
    "size of tensor",   
    "split_with_sizes",
    "index out of range",
    "out of bounds",
    "dimension out of range",
    "must be divisible by",   
    "normalized_shape",    
    "broadcast shape",
    "groups=", 
    "dependency index expansion",
    "dependency index",
    "device-side assert",  
    "acceleratorerror",
    "indices element is out of data bounds",   
)


class RepConvPruner(tp.pruner.BasePruningFunc):   
    """Prune both branches of DEIM's re-parameterizable ``VGGBlock`` together."""
 
    def prune_out_channels(self, layer: nn.Module, idxs: list[int]) -> nn.Module:    
        layer.ch_out -= len(idxs) 
        tp.prune_conv_out_channels(layer.conv1.conv, idxs)     
        tp.prune_conv_out_channels(layer.conv2.conv, idxs)    
        tp.prune_batchnorm_out_channels(layer.conv1.norm, idxs)    
        tp.prune_batchnorm_out_channels(layer.conv2.norm, idxs)
        return layer    

    def prune_in_channels(self, layer: nn.Module, idxs: list[int]) -> nn.Module:
        layer.ch_in -= len(idxs)
        tp.prune_conv_in_channels(layer.conv1.conv, idxs)
        tp.prune_conv_in_channels(layer.conv2.conv, idxs)
        return layer     

    def get_out_channels(self, layer: nn.Module) -> int:  
        return layer.ch_out
  
    def get_in_channels(self, layer: nn.Module) -> int: 
        return layer.ch_in

     
class _OpaqueIgnoredPruner(tp.pruner.BasePruningFunc): 
    """Trace an auto-ignored block as one opaque, non-prunable node.""" 

    opaque = True
    
    def prune_out_channels(self, layer: nn.Module, idxs: list[int]) -> nn.Module:
        return layer   

    def prune_in_channels(self, layer: nn.Module, idxs: list[int]) -> nn.Module:
        return layer

    def get_out_channels(self, layer: nn.Module) -> None:
        return None     

    def get_in_channels(self, layer: nn.Module) -> None:   
        return None
    

def _vgg_types() -> tuple[type[nn.Module], ...]:
    from ..deim.hybrid_encoder import VGGBlock   
  
    types: list[type[nn.Module]] = [VGGBlock]   
    try:
        from ..deim.hybrid_encoder_deimv2 import VGGBlock as DEIMV2VGGBlock   
        types.append(DEIMV2VGGBlock)
    except ImportError:
        pass 
    return tuple(types)    
   

def _first_batch_item(batch: Any) -> Any:
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (dict, torch.Tensor)):  
        return batch[0]
    return batch     
     
    
def _slice_one(value: Any) -> Any:     
    if torch.is_tensor(value):  
        return value[:1]
    if isinstance(value, dict):
        return {key: _slice_one(item) for key, item in value.items()}    
    if isinstance(value, list):
        return [_slice_one(item) for item in value]
    if isinstance(value, tuple):    
        return tuple(_slice_one(item) for item in value)    
    return value 
   

def example_inputs_from_batch(batch: Any, model: nn.Module, device: torch.device | str) -> tuple[Any, ...]:
    """Adapt one real validation batch to the one-positional-argument model API."""  
    # Delayed to avoid importing ``engine.solver`` while DetSolver imports this module.    
    from ..solver.sample_adapter import move_samples_to_device, select_model_input_for_model
    
    samples = move_samples_to_device(_first_batch_item(batch), torch.device(device))  
    return (_slice_one(select_model_input_for_model(samples, model=model, key="rgb")),)    
    
  
def _is_yaml_block(module: nn.Module) -> bool:
    return hasattr(module, "i") and hasattr(module, "f")    

  
def _trace_forward_fn(model: nn.Module, example_inputs: Any):
    """Trace decoder outputs plus executed YAML blocks that later layers skip."""
    captured: list[torch.Tensor] = []     

    def _capture(_module, _inputs, output):
        captured.extend(tensor for tensor in _output_tensors(output) if tensor.requires_grad)

    hooks = [module.register_forward_hook(_capture) for module in model.modules() if _is_yaml_block(module)] 
    try: 
        if isinstance(example_inputs, dict):
            result = model(**example_inputs)
        else:   
            try:   
                result = model(*example_inputs)
            except TypeError:     
                result = model(example_inputs)  
    finally:
        for hook in hooks:   
            hook.remove()  
    
    seen = {id(tensor) for tensor in _output_tensors(result)}  
    extras = [tensor for tensor in captured if id(tensor) not in seen and tensor.grad_fn is not None]
    return (result, extras) if extras else result    
  

def ignored_layers_and_params(  
    model: nn.Module,
    extra_class_names: Iterable[str] = (),
    extra_module_paths: Iterable[str] = (),
) -> tuple[list[nn.Module | nn.Parameter], dict[type[nn.Module] | nn.Module, tp.pruner.BasePruningFunc]]: 
    """Protect decoder, attention/text dimensions, and direct auxiliary parameters."""
    ignored: list[nn.Module | nn.Parameter] = []     
    extra_class_names = set(extra_class_names)
    extra_modules = {dict(model.named_modules())[path] for path in extra_module_paths}    
    ignored_class_names = _IGNORE_CLASS_NAMES | extra_class_names    
    decoder = getattr(model, "decoder", None)
    if isinstance(decoder, nn.Module): 
        ignored.append(decoder)
    ignored.extend(extra_modules)     
    for module in model.modules():
        name = type(module).__name__
        if name in ignored_class_names:
            ignored.append(module)
        elif name == "RepNCSPELAN4": 
            ignored.extend(child for child in (getattr(module, "cv1", None), getattr(module, "cv4", None)) if isinstance(child, nn.Module))
    modules = list(dict.fromkeys(item for item in ignored if isinstance(item, nn.Module))) 
    # MetaPruner expands ignored modules itself, but not their bare Parameters.   
    direct_params = list(dict.fromkeys(
        parameter for root in modules for module in root.modules()    
        for parameter in module.parameters(recurse=False)    
    ))
    customized_pruners = {kind: RepConvPruner() for kind in _vgg_types()}
    customized_pruners.update({type(module): _OpaqueIgnoredPruner() for module in modules if type(module).__name__ in extra_class_names})
    customized_pruners.update({module: _OpaqueIgnoredPruner() for module in extra_modules})
    return [*modules, *direct_params], customized_pruners


def _importance_and_pruner(method: str): 
    if method in _UNSUPPORTED_METHODS:
        raise ValueError(f"{method} requires a calibration loop and is not supported.")
    if method not in _SUPPORTED_METHODS:   
        raise ValueError(f"Unsupported prune_method {method!r}; choose {sorted(_SUPPORTED_METHODS)}.")
    if method == "random":
        return tp.importance.RandomImportance(), tp.pruner.MagnitudePruner   
    if method == "l1":   
        return tp.importance.MagnitudeImportance(p=1), tp.pruner.MagnitudePruner     
    if method == "lamp":
        return tp.importance.LAMPImportance(p=2), tp.pruner.MagnitudePruner
    return tp.importance.GroupNormImportance(p=2), tp.pruner.GroupNormPruner    
  

def build_pruner(   
    model: nn.Module, 
    example_inputs: tuple[Any, ...],
    method: str,
    iterative_steps: int = 20,    
    global_pruning: bool = False, 
    round_to: int | None = 8,    
    extra_ignored_class_names: Iterable[str] = (),
    extra_ignored_module_paths: Iterable[str] = (),
):
    """Build the dependency graph using the real model input tuple."""    
    importance, pruner_type = _importance_and_pruner(method)     
    ignored, customized_pruners = ignored_layers_and_params(
        model, extra_ignored_class_names, extra_ignored_module_paths
    )   
    return pruner_type(     
        model,
        example_inputs,    
        importance=importance,  
        global_pruning=global_pruning,
        pruning_ratio=1.0, 
        max_pruning_ratio=1.0,    
        iterative_steps=iterative_steps,    
        ignored_layers=ignored, 
        customized_pruners=customized_pruners,
        root_module_types=[nn.Conv2d, *_vgg_types()],     
        round_to=round_to,
        forward_fn=_trace_forward_fn,  
    )
    

def model_stats(model: nn.Module, example_inputs: tuple[Any, ...]) -> tuple[int, int]:     
    """Count FLOPs and parameters for exactly this input."""     
    flops, _, params = count_model_stats(model, example_inputs)  
    return flops, params
   
   
def _format_count(value: int) -> str:     
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if value >= scale:
            return f"{value / scale:.3f}{suffix}"
    return str(value)
    

def _forward_check(model: nn.Module, example_inputs: tuple[Any, ...]) -> None:    
    with torch.no_grad():  
        output = model(*example_inputs)
    for tensor in _output_tensors(output):    
        if not torch.isfinite(tensor).all(): 
            raise RuntimeError("Pruned model produced NaN or Inf.")
    

def _output_tensors(output: Any) -> Iterable[torch.Tensor]:     
    if torch.is_tensor(output):   
        yield output 
    elif isinstance(output, dict): 
        for item in output.values():
            yield from _output_tensors(item)
    elif isinstance(output, (list, tuple)):
        for item in output:
            yield from _output_tensors(item)   

   
def _is_structural_error(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" not in message and any(fragment in message for fragment in _STRUCTURAL_ERROR_FRAGMENTS)   
  
   
def _auto_ignore_candidates(model: nn.Module, error: Exception) -> list[tuple[str, str, nn.Module]]:    
    """Return the failing prune root, its parent, then its parsed YAML block."""    
    named_modules = dict(model.named_modules())
    names_by_id = {id(module): name for name, module in named_modules.items()}

    def module_for(value: nn.Module | nn.Parameter | None) -> nn.Module | None:
        if isinstance(value, nn.Parameter):     
            name = next((name for name, parameter in model.named_parameters() if parameter is value), None)
            return named_modules.get(name.rpartition(".")[0]) if name else None     
        return value if isinstance(value, nn.Module) and id(value) in names_by_id else None
  
    def yaml_block(path: str) -> tuple[str, nn.Module] | None:
        if not path:
            return None     
        parts = path.split(".")   
        fallback = None
        for end in range(len(parts), 0, -1):    
            name = ".".join(parts[:end])
            candidate = named_modules[name]     
            if hasattr(candidate, "i") and hasattr(candidate, "f"):
                return name, candidate    
            parent = named_modules.get(".".join(parts[:end - 1]), model)
            if isinstance(parent, (nn.Sequential, nn.ModuleList)):
                fallback = (name, candidate)  
        return fallback 
 
    module = module_for(getattr(error, "root_module", None)) or module_for(getattr(error, "failed_module", None))   

    if module is None: 
        traceback = error.__traceback__  
        while traceback is not None:     
            for candidate in traceback.tb_frame.f_locals.values():
                if isinstance(candidate, nn.Module) and id(candidate) in names_by_id:   
                    module = candidate
            traceback = traceback.tb_next
    if module is None:    
        return []   
 
    path = names_by_id[id(module)]  
    if not path:
        return []  
    candidates = [("root", path, module)]   
    parent_path = path.rpartition(".")[0]     
    if parent_path:     
        candidates.append(("parent", parent_path, named_modules[parent_path])) 
    block = yaml_block(path)
    if block is not None:
        candidates.append(("yaml", *block))  
    return list(dict.fromkeys(candidates))


def _run_pruning_once(
    model: nn.Module,
    example_inputs: tuple[Any, ...],   
    method: str, 
    speed_up: float,
    before_flops: int,     
    before_params: int,     
    iterative_steps: int,
    global_pruning: bool,  
    round_to: int | None,
    auto_ignored_module_paths: list[str],  
) -> dict[str, Any]: 
    pruner = build_pruner(  
        model, example_inputs, method, iterative_steps, global_pruning, round_to,    
        extra_ignored_module_paths=auto_ignored_module_paths,
    )    
    after_flops, after_params = before_flops, before_params 
    while pruner.current_step < iterative_steps and before_flops / after_flops < speed_up:  
        pruner.step()
        _forward_check(model, example_inputs)
        previous_flops, previous_params = after_flops, after_params
        after_flops, after_params = model_stats(model, example_inputs)
        flops_change = (after_flops / previous_flops - 1) * 100 if previous_flops else 0.0    
        params_change = (after_params / previous_params - 1) * 100 if previous_params else 0.0    
        logger.info( 
            f"[Prune step {pruner.current_step:02d}/{iterative_steps}] "  
            f"FLOPs {_format_count(previous_flops)} -> {_format_count(after_flops)} ({flops_change:+.2f}%), "
            f"Params {_format_count(previous_params)} -> {_format_count(after_params)} ({params_change:+.2f}%), "
            f"speed-up {before_flops / after_flops:.2f}x"
        )
    if after_flops >= before_flops or after_params >= before_params:
        raise RuntimeError("Pruning made no measurable channel/FLOPs change.")
    return {   
        "model": model,
        "pruner": pruner,
        "before_flops": before_flops,
        "after_flops": after_flops,
        "before_params": before_params, 
        "after_params": after_params,
        "steps": pruner.current_step,
        "speed_up": before_flops / after_flops,
    }     


def run_pruning(
    model: nn.Module,    
    example_inputs: tuple[Any, ...],
    method: str,    
    speed_up: float,    
    iterative_steps: int = 20,
    global_pruning: bool = False,
    round_to: int | None = 8,
) -> dict[str, Any]:
    """Prune until the requested measured speed-up, then prove the model still forwards."""     
    if speed_up <= 1:
        raise ValueError("speed_up must be greater than 1 for structural pruning.")
    model.eval()
    _forward_check(model, example_inputs)     
    before_flops, before_params = model_stats(model, example_inputs)  
    if before_flops <= 0:    
        raise RuntimeError("Could not measure positive FLOPs before pruning.")    
    clean_model = copy.deepcopy(model)
    auto_ignored_classes: list[str] = []
    auto_ignored_module_paths: list[str] = []   
    auto_ignore_levels: dict[str, str] = {}   
    auto_ignore_reasons: dict[str, str] = {}     
    while True:
        try:   
            result = _run_pruning_once(
                model, example_inputs, method, speed_up, before_flops, before_params,
                iterative_steps, global_pruning, round_to, auto_ignored_module_paths,
            )
            result.update({
                "auto_ignored_classes": auto_ignored_classes,   
                "auto_ignored_module_paths": auto_ignored_module_paths,
                "auto_ignore_levels": auto_ignore_levels,     
                "auto_ignore_reasons": auto_ignore_reasons,
                "auto_ignore_attempts": len(auto_ignored_module_paths),  
            })
            return result    
        except _PRUNE_RETRY_ERRORS as error:
            candidates = _auto_ignore_candidates(model, error) if _is_structural_error(error) else []
            selected = next((candidate for candidate in candidates if not any(
                candidate[1] == ignored or candidate[1].startswith(f"{ignored}.")
                for ignored in auto_ignored_module_paths   
            )), None)  
            if selected is None:
                raise
            level, module_path, module = selected
            class_name = type(module).__name__
            if class_name in _IGNORE_CLASS_NAMES:
                raise 
            reason = f"{module_path}: {error}"   
            auto_ignored_module_paths.append(module_path)
            auto_ignore_levels[module_path] = level  
            if class_name not in auto_ignored_classes:     
                auto_ignored_classes.append(class_name)    
            auto_ignore_reasons[class_name] = reason     
            logger.warning(
                f"Pruning broke {module_path} ({class_name}); retrying from the clean model with this {level} ignored: {error}"
            )
            model.__dict__, clean_model.__dict__ = clean_model.__dict__, model.__dict__
            clean_model = copy.deepcopy(model)
