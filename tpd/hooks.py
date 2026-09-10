"""
Hook engine for TPD.
Hooks into PyTorch modules to capture forward/backward tensor data.
"""

import os
import sys
import weakref
from typing import Any

import torch
import torch.nn as nn

from .config import Config
from .sampler import safe_sample
from .stack import capture_current_stack, get_stack_manager
from .storage import StorageManager
from .summary import safe_compute_summary


class HookEngine:
    """
    Hooks into all nn.Module instances to capture forward/backward data.
    Manages step counting, periodic saving, and signal handling.
    """

    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.stack_manager = get_stack_manager()

        # Counters
        self._forward_count = 0
        self._backward_count = 0
        self._total_hook_count = 0
        self._stopped = False

        # Track hooked modules
        self._hooked_modules: weakref.WeakSet = weakref.WeakSet()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        # Module filter
        self._module_filter = Config.get_module_filter_list()

    def _should_hook_module(self, module: nn.Module) -> bool:
        """Check if a module should be hooked based on filter."""
        if not self._module_filter:
            return True
        module_type = type(module).__name__
        full_name = f"{type(module).__module__}.{module_type}"
        for prefix in self._module_filter:
            if prefix in module_type or prefix in full_name:
                return True
        return False

    def _should_stop(self) -> bool:
        """Check if we've reached the max step limit."""
        if self._stopped:
            return True
        if Config.MAX_STEPS > 0 and self._total_hook_count >= Config.MAX_STEPS:
            self._stopped = True
            return True
        return False

    def _should_save(self) -> bool:
        """Check if we should save data now."""
        return self._total_hook_count % Config.SAVE_INTERVAL == 0

    def _get_module_name(self, module: nn.Module) -> str:
        """Get a descriptive name for a module."""
        return f"{type(module).__module__}.{type(module).__name__}"

    def _extract_tensors(self, data: Any) -> list[tuple[str, torch.Tensor]]:
        """Extract all tensors from a nested structure (args/kwargs/output)."""
        tensors = []
        if isinstance(data, torch.Tensor):
            tensors.append(("", data))
        elif isinstance(data, (tuple, list)):
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    tensors.append((f"[{i}]", item))
                elif isinstance(item, (tuple, list)):
                    for j, sub_item in enumerate(item):
                        if isinstance(sub_item, torch.Tensor):
                            tensors.append((f"[{i}][{j}]", sub_item))
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    tensors.append((f".{key}", value))
        return tensors

    def _forward_pre_hook(self, module: nn.Module, args: tuple, kwargs: dict | None = None):
        """Hook called before forward pass. Captures input tensors."""
        if self._should_stop():
            return

        self._forward_count += 1
        self._total_hook_count += 1

        # Capture stack trace
        frames = capture_current_stack()
        stack_id = self.stack_manager.get_stack_id(frames)

        # Extract and process input tensors
        input_records = []
        all_args = list(args) if args else []
        if kwargs:
            all_args.append(kwargs)

        for arg_idx, arg in enumerate(args or []):
            for path, tensor in self._extract_tensors(arg):
                record = {
                    "step": self._forward_count,
                    "hook_type": "forward_input",
                    "module_name": self._get_module_name(module),
                    "module_id": id(module),
                    "tensor_path": f"args[{arg_idx}]{path}",
                    "stack_id": stack_id,
                    "summary": safe_compute_summary(tensor),
                    "samples": safe_sample(tensor),
                }
                input_records.append(record)

        if kwargs:
            for key, value in kwargs.items():
                for path, tensor in self._extract_tensors(value):
                    record = {
                        "step": self._forward_count,
                        "hook_type": "forward_input",
                        "module_name": self._get_module_name(module),
                        "module_id": id(module),
                        "tensor_path": f"kwargs.{key}{path}",
                        "stack_id": stack_id,
                        "summary": safe_compute_summary(tensor),
                        "samples": safe_sample(tensor),
                    }
                    input_records.append(record)

        # Store records
        if input_records:
            self.storage.append_records(input_records)

        # Periodic save
        if self._should_save():
            self._periodic_save()

    def _forward_hook(self, module: nn.Module, args: tuple, output: Any):
        """Hook called after forward pass. Captures output tensors."""
        if self._stopped:
            return

        # Capture stack trace
        frames = capture_current_stack()
        stack_id = self.stack_manager.get_stack_id(frames)

        # Extract and process output tensors
        output_records = []
        for path, tensor in self._extract_tensors(output):
            record = {
                "step": self._forward_count,
                "hook_type": "forward_output",
                "module_name": self._get_module_name(module),
                "module_id": id(module),
                "tensor_path": f"output{path}",
                "stack_id": stack_id,
                "summary": safe_compute_summary(tensor),
                "samples": safe_sample(tensor),
            }
            output_records.append(record)

        # Store records
        if output_records:
            self.storage.append_records(output_records)

    def _backward_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple):
        """Hook called during backward pass. Captures gradient tensors."""
        if self._should_stop():
            return

        self._backward_count += 1
        self._total_hook_count += 1

        # Capture stack trace
        frames = capture_current_stack()
        stack_id = self.stack_manager.get_stack_id(frames)

        records = []

        # Process grad_input
        for idx, grad in enumerate(grad_input or []):
            if grad is not None and isinstance(grad, torch.Tensor):
                record = {
                    "step": self._backward_count,
                    "hook_type": "backward_grad_input",
                    "module_name": self._get_module_name(module),
                    "module_id": id(module),
                    "tensor_path": f"grad_input[{idx}]",
                    "stack_id": stack_id,
                    "summary": safe_compute_summary(grad),
                    "samples": safe_sample(grad),
                }
                records.append(record)

        # Process grad_output
        for idx, grad in enumerate(grad_output or []):
            if grad is not None and isinstance(grad, torch.Tensor):
                record = {
                    "step": self._backward_count,
                    "hook_type": "backward_grad_output",
                    "module_name": self._get_module_name(module),
                    "module_id": id(module),
                    "tensor_path": f"grad_output[{idx}]",
                    "stack_id": stack_id,
                    "summary": safe_compute_summary(grad),
                    "samples": safe_sample(grad),
                }
                records.append(record)

        # Store records
        if records:
            self.storage.append_records(records)

        # Periodic save
        if self._should_save():
            self._periodic_save()

    def _periodic_save(self):
        """Save stack traces periodically."""
        try:
            self.stack_manager.save_stacks(self.storage.get_stacks_path())
        except Exception as e:
            print(f"[TPD] Warning: Failed to save stacks: {e}", file=sys.stderr)

    def hook_module(self, module: nn.Module):
        """Register hooks on a single module."""
        if module in self._hooked_modules:
            return
        if not self._should_hook_module(module):
            return

        # Register forward pre-hook (captures inputs)
        handle1 = module.register_forward_pre_hook(self._forward_pre_hook, with_kwargs=True)
        self._handles.append(handle1)

        # Register forward hook (captures outputs)
        handle2 = module.register_forward_hook(self._forward_hook)
        self._handles.append(handle2)

        # Register backward hook (captures gradients)
        handle3 = module.register_full_backward_hook(self._backward_hook)
        self._handles.append(handle3)

        self._hooked_modules.add(module)

    def hook_all_modules(self, root_module: nn.Module):
        """Recursively hook all submodules."""
        for module in root_module.modules():
            self.hook_module(module)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hooked_modules.clear()

    def finalize(self):
        """Final save of all data."""
        self._periodic_save()
        print(
            f"[TPD] Finalized: {self._forward_count} forward, "
            f"{self._backward_count} backward, "
            f"{self._total_hook_count} total hook calls",
            file=sys.stderr,
        )


class ModuleCreationMonitor:
    """
    Monitors module creation to automatically hook new modules.
    Uses sys.settrace to intercept __init__ calls.
    """

    def __init__(self, hook_engine: HookEngine):
        self.hook_engine = hook_engine
        self._original_trace = None
        self._active = False

    def start(self):
        """Start monitoring module creation."""
        if self._active:
            return
        self._active = True
        self._original_trace = sys.gettrace()
        sys.settrace(self._trace_callback)

    def stop(self):
        """Stop monitoring module creation."""
        if not self._active:
            return
        self._active = False
        sys.settrace(self._original_trace)
        self._original_trace = None

    def _trace_callback(self, frame, event, arg):
        """Trace callback to intercept module creation."""
        if event == "call":
            # Check if this is an nn.Module.__init__ call
            if "self" in frame.f_locals:
                self_obj = frame.f_locals["self"]
                if isinstance(self_obj, nn.Module):
                    try:
                        self.hook_engine.hook_module(self_obj)
                    except Exception:
                        pass
        return self._trace_callback
