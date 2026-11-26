"""Every executed nn.Module output, at return time; evaluation observer only."""
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
import torch


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


class RawModuleRecorder:
    def __init__(self, model, destination):
        assert not model.training
        self.model = model
        self.root = Path(destination)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'blobs').mkdir()
        self.handles, self.stack, self.calls = [], [], []
        self.counts, self.blobs = {}, {}
        self.active = False
        names = {}
        for name, module in model.named_modules(remove_duplicate=False):
            names.setdefault(id(module), []).append(name or 'model')
        self.inventory = {}
        for name, module in model.named_modules():
            name = name or 'model'
            self.inventory[name] = dict(class_name=type(module).__name__,
                registered_aliases=names[id(module)], direct_parameter_count=sum(
                    p.numel() for p in module.parameters(recurse=False)))
            self.handles.append(module.register_forward_pre_hook(self._pre(name), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self._post(name), with_kwargs=True))

    def tensor(self, value):
        assert value.layout == torch.strided and not value.is_quantized
        cpu = value.detach().cpu().contiguous()
        original_dtype = str(cpu.dtype)
        if cpu.dtype == torch.bfloat16:
            array = cpu.view(torch.uint16).numpy()
            encoding = 'bfloat16_raw_bits_as_uint16'
        else:
            array = cpu.numpy()
            encoding = 'numpy_exact_dtype'
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        raw = buffer.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        name = digest + '.npy'
        if digest not in self.blobs:
            with (self.root / 'blobs' / name).open('xb') as stream:
                stream.write(raw)
            self.blobs[digest] = dict(file='blobs/' + name, bytes=len(raw))
        return dict(kind='tensor', original_dtype=original_dtype,
            storage_numpy_dtype=str(array.dtype), shape=list(value.shape),
            original_device=str(value.device), encoding=encoding,
            blob_sha256=digest, blob_file='blobs/' + name, tensor_bytes=int(array.nbytes))

    def pack(self, value):
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if isinstance(value, dict):
            return dict(kind='dict', entries=[dict(key=self.pack(key), value=self.pack(item))
                                             for key, item in value.items()])
        if isinstance(value, (tuple, list)):
            return dict(kind=type(value).__name__, items=[self.pack(item) for item in value])
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return dict(kind='nonfinite_float', repr=repr(value))
            return dict(kind='scalar', python_type=type(value).__name__, value=value)
        raise TypeError('Unsupported output type; no silent omission: ' + type(value).__name__)

    def begin(self, metadata, model_input, targets):
        assert not self.active and not self.stack and not self.model.training
        assert not torch.is_grad_enabled()
        self.active = True
        self.metadata = metadata
        self.input = self.pack(model_input)
        self.targets = self.pack(targets)

    def _pre(self, name):
        def hook(module, args, kwargs):
            if not self.active:
                return None
            assert not module.training and not torch.is_grad_enabled()
            invocation = self.counts.get(name, 0)
            self.counts[name] = invocation + 1
            event = dict(call_id=len(self.calls), module=name, invocation=invocation,
                parent_call_id=self.stack[-1] if self.stack else None,
                class_name=type(module).__name__)
            self.calls.append(event)
            self.stack.append(event['call_id'])
            return None
        return hook

    def _post(self, name):
        def hook(module, args, kwargs, output):
            if not self.active:
                return None
            event = self.calls[self.stack.pop()]
            assert event['module'] == name
            # Serialize now: downstream in-place operations may mutate output storage.
            event['output'] = self.pack(output)
            return None
        return hook

    def finish(self):
        assert self.active and not self.stack
        assert all('output' in row for row in self.calls)
        assert self.counts.get('model') == 1
        self.active = False
        for name, row in self.inventory.items():
            row['actual_calls'] = self.counts.get(name, 0)
        record = dict(schema='candidate_all_called_module_raw_outputs_v1',
            status='ALL_CALLED_MODULE_OUTPUTS_SERIALIZED', metadata=self.metadata,
            normalized_model_input=self.input, native_targets=self.targets,
            registered_modules=self.inventory, invocation_events=self.calls,
            unique_physical_blobs=self.blobs, total_module_calls=len(self.calls),
            registered_module_count=len(self.inventory),
            called_module_count=sum(n > 0 for n in self.counts.values()),
            structure_calls_never_merged=True,
            blob_deduplication_only=True, original_output_dtypes_preserved=True,
            functional_operations_outside_nn_module_hooks_not_claimed=True,
            uncalled_registered_containers_or_branches_not_invented=True,
            no_extra_head_matcher_loss_or_evaluator_calls=True,
            no_backward_or_optimizer_or_training_calls=True,
            causal_claim=False, population_efficacy_claim=False)
        write(self.root / 'record.json', record)
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False
