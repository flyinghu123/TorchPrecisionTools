"""
Test script for TPD - Torch Precision Debugger.
"""

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


def test_basic():
    """Test basic functionality."""
    print("=" * 80)
    print("Testing TPD basic functionality")
    print("=" * 80)

    # Create model
    model = SimpleModel()

    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 10)

    # Forward pass
    print("\n[TEST] Forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Compute loss
    loss = F.mse_loss(output, y)
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


def test_multiple_forwards():
    """Test multiple forward/backward passes."""
    print("\n" + "=" * 80)
    print("Testing multiple forward/backward passes")
    print("=" * 80)

    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_steps = 5
    for step in range(num_steps):
        print(f"\n[TEST] Step {step + 1}/{num_steps}")

        x = torch.randn(4, 10)
        y = torch.randn(4, 10)

        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.6f}")

    print("\n[TEST] Multiple passes test completed!")


def test_nan_inf():
    """Test NaN and Inf handling."""
    print("\n" + "=" * 80)
    print("Testing NaN/Inf handling")
    print("=" * 80)

    # Create tensor with NaN and Inf
    x = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf'), -float('inf')])
    print(f"Input tensor: {x}")
    print(f"NaN count: {torch.isnan(x).sum().item()}")
    print(f"Inf count: {torch.isinf(x).sum().item()}")

    # Test with model
    model = SimpleModel()

    # Inject NaN into input
    x_with_nan = torch.randn(4, 10)
    x_with_nan[0, 0] = float('nan')
    x_with_nan[1, 1] = float('inf')

    print("\n[TEST] Forward with NaN/Inf in input...")
    output = model(x_with_nan)
    print(f"Output has NaN: {torch.isnan(output).any().item()}")
    print(f"Output has Inf: {torch.isinf(output).any().item()}")

    print("\n[TEST] NaN/Inf test completed!")


def main():
    """Run all tests."""
    print("TPD Test Suite")
    print(f"TPD_ENABLED: {os.environ.get('TPD_ENABLED', '0')}")
    print(f"TPD_OUTPUT_DIR: {os.environ.get('TPD_OUTPUT_DIR', './tpd_results')}")
    print(f"TPD_SAMPLE_COUNT: {os.environ.get('TPD_SAMPLE_COUNT', '50')}")
    print(f"TPD_SAMPLE_MODE: {os.environ.get('TPD_SAMPLE_MODE', 'uniform')}")

    test_basic()
    test_multiple_forwards()
    test_nan_inf()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
