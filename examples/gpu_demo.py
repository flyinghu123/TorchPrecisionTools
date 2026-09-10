"""
Test script for TPD - Torch Precision Debugger (single GPU).

Run the fp32 baseline and the --amp (fp16 autocast) run back to back with the
same seed, then use `tpd compare` to locate the precision differences.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        self.norm = nn.LayerNorm(10)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x


def autocast_ctx(use_amp: bool):
    """Autocast context that is active only when AMP is enabled."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)


def test_basic(device, use_amp):
    """Test basic functionality."""
    print("=" * 80)
    print(f"Testing TPD basic functionality (amp={'on' if use_amp else 'off'})")
    print("=" * 80)

    # Create model
    model = SimpleModel().to(device)

    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 10, device=device)
    y = torch.randn(batch_size, 10, device=device)

    # Forward pass
    print("\n[TEST] Forward pass...")
    with autocast_ctx(use_amp):
        output = model(x)
        loss = F.mse_loss(output, y)
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"Loss: {loss.item()}")

    # Backward pass
    print("\n[TEST] Backward pass...")
    loss.backward()

    # Check gradients
    print("\n[TEST] Checking gradients...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad shape={param.grad.shape}, mean={param.grad.mean().item():.6f}")

    print("\n[TEST] Basic test completed!")


def test_multiple_forwards(device, use_amp):
    """Test multiple forward/backward passes."""
    print("\n" + "=" * 80)
    print("Testing multiple forward/backward passes")
    print("=" * 80)

    model = SimpleModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_steps = 5
    for step in range(num_steps):
        print(f"\n[TEST] Step {step + 1}/{num_steps}")

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 10, device=device)

        optimizer.zero_grad()
        with autocast_ctx(use_amp):
            output = model(x)
            loss = F.mse_loss(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Loss: {loss.item():.6f}")

    print("\n[TEST] Multiple passes test completed!")


def test_nan_inf(device, use_amp):
    """Test NaN and Inf handling."""
    print("\n" + "=" * 80)
    print("Testing NaN/Inf handling")
    print("=" * 80)

    # Create tensor with NaN and Inf
    x = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf'), -float('inf')], device=device)
    print(f"Input tensor: {x}")
    print(f"NaN count: {torch.isnan(x).sum().item()}")
    print(f"Inf count: {torch.isinf(x).sum().item()}")

    # Test with model
    model = SimpleModel().to(device)

    # Inject NaN into input
    x_with_nan = torch.randn(4, 10, device=device)
    x_with_nan[0, 0] = float('nan')
    x_with_nan[1, 1] = float('inf')

    print("\n[TEST] Forward with NaN/Inf in input...")
    with autocast_ctx(use_amp):
        output = model(x_with_nan)
    print(f"Output has NaN: {torch.isnan(output).any().item()}")
    print(f"Output has Inf: {torch.isinf(output).any().item()}")

    print("\n[TEST] NaN/Inf test completed!")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="TPD single GPU test suite")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="run forward/backward under fp16 autocast (mixed precision)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available, use cpu_demo.py for CPU-only testing")
        sys.exit(1)
    device = torch.device("cuda:0")

    # Fixed seed so the fp32 baseline and the --amp run see identical data:
    # any difference reported by `tpd compare` comes from precision only
    torch.manual_seed(42)

    print("TPD GPU Test Suite")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"AMP: {'enabled (fp16 autocast)' if args.amp else 'disabled (fp32)'}")
    print(f"TPD_ENABLED: {os.environ.get('TPD_ENABLED', '0')}")
    print(f"TPD_OUTPUT_DIR: {os.environ.get('TPD_OUTPUT_DIR', './tpd_results')}")
    print(f"TPD_SAMPLE_COUNT: {os.environ.get('TPD_SAMPLE_COUNT', '50')}")
    print(f"TPD_SAMPLE_MODE: {os.environ.get('TPD_SAMPLE_MODE', 'uniform')}")

    test_basic(device, args.amp)
    test_multiple_forwards(device, args.amp)
    test_nan_inf(device, args.amp)

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
