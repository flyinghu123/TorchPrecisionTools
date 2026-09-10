"""最小可复现的 MLP 训练脚本：用于验证 pprobe 的 forward/backward 采集。

用法（不需要 import pprobe，靠环境变量注入）::

    PPROBE_ENABLE=1 PPROBE_OUT=./runs/base python examples/mlp_simple.py --device cuda
    PPROBE_ENABLE=1 PPROBE_OUT=./runs/low   python examples/mlp_simple.py --device cuda --eps 1e-8

数值差异来源用命令行参数模拟：
* ``--eps``       改 LayerNorm 的 epsilon（真实场景中常见：不同框架默认 eps 不同）
* ``--tf32``      打开 cuBLAS TF32（Ampere 及以上会有 ~1e-3 级别的矩阵乘差异）
* ``--dtype``     计算精度 fp32 / fp16 / bf16
* ``--device``    cpu / cuda（跨平台对比）
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.res = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return self.res(x) + h


class TinyModel(nn.Module):
    def __init__(self, dim: int = 64, layers: int = 2, eps: float = 1e-5):
        super().__init__()
        self.embed = nn.Linear(32, dim)
        self.blocks = nn.ModuleList([Block(dim, eps=eps) for _ in range(layers)])
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("cuda 不可用，跳过", file=sys.stderr)
            return 2
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.manual_seed(args.seed)

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = TinyModel(dim=64, layers=2, eps=args.eps).to(device=device, dtype=dtype)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    losses = []
    for step in range(args.steps):
        x = torch.randn(args.batch, 32, device=device, dtype=dtype)
        y = torch.randint(0, 10, (args.batch,), device=device)
        opt.zero_grad(set_to_none=False)
        logits = model(x)
        loss = crit(logits.float(), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        print(f"step {step}: loss={losses[-1]:.6f} dtype={args.dtype} device={args.device} "
              f"eps={args.eps} tf32={args.tf32}")
    print("mean_loss=%.8f" % (sum(losses) / len(losses)))
    if os.environ.get("PPROBE_ENABLE"):
        print("（pprobe 已通过 .pth 自动注入）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
