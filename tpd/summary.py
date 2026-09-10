"""
Tensor summary statistics.
Computes shape, stride, device, dtype, and numerical statistics.
"""

import math
import torch


def compute_summary(tensor: torch.Tensor) -> dict:
    """
    Compute summary statistics for a tensor.
    Returns a dictionary with:
    - shape, stride, device, dtype (basic info)
    - max, min, mean, var (numerical stats)
    - nan_count, inf_count (special value counts)
    """
    summary = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
    }

    if tensor.numel() == 0:
        summary.update({
            "max": None,
            "min": None,
            "mean": None,
            "var": None,
            "nan_count": 0,
            "inf_count": 0,
        })
        return summary

    # Handle non-floating point tensors
    if not tensor.is_floating_point():
        tensor_float = tensor.float()
    else:
        tensor_float = tensor

    # Count NaN and Inf
    nan_count = torch.isnan(tensor_float).sum().item()
    inf_count = torch.isinf(tensor_float).sum().item()

    # Compute statistics (handle NaN/Inf gracefully)
    finite_mask = torch.isfinite(tensor_float)
    finite_values = tensor_float[finite_mask]

    if finite_values.numel() > 0:
        max_val = torch.max(finite_values).item()
        min_val = torch.min(finite_values).item()
        mean_val = torch.mean(finite_values).item()
        var_val = torch.var(finite_values).item() if finite_values.numel() > 1 else 0.0
    else:
        max_val = None
        min_val = None
        mean_val = None
        var_val = None

    # Also compute raw max/min including NaN/Inf for reference
    try:
        raw_max = torch.max(tensor_float).item()
        raw_min = torch.min(tensor_float).item()
    except Exception:
        raw_max = None
        raw_min = None

    summary.update({
        "max": max_val,
        "min": min_val,
        "mean": mean_val,
        "var": var_val,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "raw_max": raw_max,
        "raw_min": raw_min,
    })

    return summary


def safe_compute_summary(tensor: torch.Tensor) -> dict:
    """Safely compute summary, handling edge cases."""
    try:
        return compute_summary(tensor)
    except Exception as e:
        return {
            "shape": list(tensor.shape) if hasattr(tensor, "shape") else [],
            "stride": list(tensor.stride()) if hasattr(tensor, "stride") else [],
            "device": str(tensor.device) if hasattr(tensor, "device") else "unknown",
            "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else "unknown",
            "numel": tensor.numel() if hasattr(tensor, "numel") else 0,
            "max": None,
            "min": None,
            "mean": None,
            "var": None,
            "nan_count": 0,
            "inf_count": 0,
            "raw_max": None,
            "raw_min": None,
            "error": str(e),
        }
