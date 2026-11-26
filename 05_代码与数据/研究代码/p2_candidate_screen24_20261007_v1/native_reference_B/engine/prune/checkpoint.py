"""Trusted local checkpoint helpers for structurally pruned DEIM models.""" 
  
from __future__ import annotations   
    
from pathlib import Path  
from typing import Any    

import torch    
from torch import nn

     
def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:  
    """Load a trusted local checkpoint, including its optional full prune model."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):     
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)!r}.")
    return checkpoint

 
def load_pruned_model(path: str | Path, device: torch.device | str | None = None) -> tuple[nn.Module, dict[str, Any]]:    
    """Return the complete model serialized by a structural-pruning checkpoint."""
    checkpoint = load_checkpoint(path)
    model = checkpoint.get("prune_model")
    if not isinstance(model, nn.Module): 
        raise ValueError(f"{path} does not contain a complete 'prune_model'.")   
    return (model.to(device) if device is not None else model), checkpoint 
  
     
def checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Pick the normal training state, preferring EMA when present."""
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]  
    state = checkpoint.get("model")     
    if not isinstance(state, dict):     
        raise ValueError("Checkpoint has neither ema.module nor model state_dict.")    
    return state

    
def load_model_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,  
    device: torch.device | str | None = None,
) -> tuple[nn.Module, dict[str, Any]]: 
    """Load a prune model directly, or a normal checkpoint into ``model``."""   
    checkpoint = load_checkpoint(path)
    pruned = checkpoint.get("prune_model")   
    if isinstance(pruned, nn.Module):
        return (pruned.to(device) if device is not None else pruned), checkpoint
    if model is None:    
        raise ValueError("A model is required for a checkpoint without 'prune_model'.")   
    model.load_state_dict(checkpoint_state(checkpoint))
    return (model.to(device) if device is not None else model), checkpoint    


def save_pruned_checkpoint(
    path: str | Path,    
    state: dict[str, Any],
    prune_model: nn.Module,
) -> Path:
    """Save normal solver state together with the complete pruned architecture."""   
    if not isinstance(prune_model, nn.Module):     
        raise TypeError("prune_model must be an nn.Module.") 
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["prune_model"] = prune_model
    torch.save(payload, target) 
    return target  
