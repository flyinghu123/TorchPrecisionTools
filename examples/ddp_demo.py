"""
Test script for TPD - Torch Precision Debugger (multi-GPU DDP via torchrun).

Launch with:
    torchrun --nproc_per_node=2 ddp_demo.py

TPD picks up RANK/LOCAL_RANK/WORLD_SIZE exported by torchrun and writes one
set of result files per rank:
    rank{N}.jsonl / stacks_rank{N}.json / config_rank{N}.json
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SimpleDataset(Dataset):
    """Simple dataset with a fixed seed so every run sees the same data."""

    def __init__(self, size: int = 16):
        generator = torch.Generator().manual_seed(42)
        self.inputs = torch.randn(size, 10, generator=generator)
        self.targets = torch.randn(size, 10, generator=generator)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


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


def main():
    """Run a small DDP training loop."""
    # torchrun exports RANK / LOCAL_RANK / WORLD_SIZE for every process
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        print("This demo must be launched with torchrun, e.g.:")
        print("  torchrun --nproc_per_node=2 ddp_demo.py")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA is not available, this demo requires GPUs")
        sys.exit(1)

    # Optional: inject NaN on one rank to simulate a precision issue
    inject_nan_rank = int(os.environ.get("INJECT_NAN_RANK", "-1"))

    print("TPD DDP Test Suite")
    print(f"Rank: {rank}/{world_size}, local rank: {local_rank}")
    print(f"Device: cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})")
    print(f"TPD_ENABLED: {os.environ.get('TPD_ENABLED', '0')}")
    print(f"TPD_OUTPUT_DIR: {os.environ.get('TPD_OUTPUT_DIR', './tpd_results')}")
    print(f"TPD_SAMPLE_COUNT: {os.environ.get('TPD_SAMPLE_COUNT', '50')}")
    print(f"TPD_SAMPLE_MODE: {os.environ.get('TPD_SAMPLE_MODE', 'uniform')}")
    if inject_nan_rank >= 0:
        print(f"INJECT_NAN_RANK: {inject_nan_rank} (simulated precision issue)")

    # NCCL backend for GPU communication, bind each process to its GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    # Same seed on every rank so initial weights are identical
    torch.manual_seed(42)

    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    dataset = SimpleDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    num_epochs = 2
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            # Simulate a precision issue on the configured rank
            if rank == inject_nan_rank:
                x[0, 0] = float("nan")

            optimizer.zero_grad()
            output = ddp_model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"[TEST] Epoch {epoch} step {step}: loss={loss.item():.6f}")

    if rank == 0:
        print("\n[TEST] DDP test completed!")

    dist.destroy_process_group()

    # TPD writes one set of files per rank (RANK is set by torchrun)
    output_dir = os.environ.get("TPD_OUTPUT_DIR", "./tpd_results")
    print(f"[Rank {rank}] Records: {output_dir}/rank{rank}.jsonl")
    print(f"[Rank {rank}] Stacks:  {output_dir}/stacks_rank{rank}.json")


if __name__ == "__main__":
    main()
