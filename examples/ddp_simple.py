"""多卡 / 多进程示例：验证 pprobe 按 rank 分目录采集，并可分 rank 对比。

单机 2 进程（没有 GPU 时用 gloo，一样能验证 rank 目录与对比链路）::

    PPROBE_ENABLE=1 PPROBE_OUT=runs/ddp_base \\
      torchrun --nproc_per_node=2 examples/ddp_simple.py
    PPROBE_ENABLE=1 PPROBE_OUT=runs/ddp_eps \\
      torchrun --nproc_per_node=2 examples/ddp_simple.py --eps 1e-8
    pprobe compare runs/ddp_base runs/ddp_eps --out diff.md

``--bad-rank`` 模拟“只有一张卡不对”这种最难查的场景（某张卡的数据分片/
通信 dtype/驱动版本不同）：只有该 rank 的 LayerNorm eps 被放大 10 倍：::

    PPROBE_ENABLE=1 PPROBE_OUT=runs/ddp_bad \\
      torchrun --nproc_per_node=2 examples/ddp_simple.py --bad-rank 1
    pprobe compare runs/ddp_base runs/ddp_bad      # 报告里 rank1 从第 0 次前向就发散

真实框架里一行代码都不用改，只要::

    PPROBE_ENABLE=1 PPROBE_OUT=runs/ddp_base \\
      torchrun --nproc_per_node=8 pretrain_gpt.py ...

注意 pprobe 完全不需要修改这段代码：每个 rank 进程都会通过 ``pprobe.pth`` 自动注入，
并各自写 ``runs/.../rankN/``。
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim * 2, eps=eps)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.act(self.norm(self.fc1(x)))
        return self.fc2(h) + x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--bad-rank", type=int, default=-1, help="只把该 rank 的 eps 放大 10 倍")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    want_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if want_cuda:
        device = torch.device(f"cuda:{local_rank % max(1, torch.cuda.device_count())}")
        torch.cuda.set_device(device)
        backend = "nccl"
        try:  # 驱动/算子库不匹配时（比如 sm 太老）直接退回 gloo，不然报错信息很难看
            torch.zeros(1, device=device) + 1
        except Exception as e:  # noqa: BLE001
            print(f"[rank {rank}] cuda 不可用（{type(e).__name__}），改用 cpu/gloo", flush=True)
            device, backend = torch.device("cpu"), "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=120))

    torch.manual_seed(args.seed)  # 所有 rank 相同的初始化 → 差异只可能来自计算
    eps = args.eps * 10 if rank == args.bad_rank else args.eps
    model = TinyMLP(args.dim, eps=eps).to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None if device.type == "cpu" else [device])
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.MSELoss()

    for step in range(args.steps):
        # 每个 rank 用自己的数据分片：真实训练里 rank 间的差异往往来自数据/通信
        g = torch.Generator(device="cpu").manual_seed(args.seed * 100 + step * 7 + rank)
        x = torch.randn(8, args.dim, generator=g).to(device)
        y = torch.randn(8, args.dim, generator=g).to(device)
        opt.zero_grad(set_to_none=False)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        extra = ""
        if world > 1:
            t = loss.detach().clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            extra = f" avg={float(t) / world:.8f}"
        print(f"[rank {rank}/{world}] step {step} loss={float(loss.detach()):.8f} "
              f"device={device} eps={eps:g}{extra}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print(f"dist-ok rank={rank} world={world} backend={backend} device={device.type} eps={eps:g}",
          flush=True)
    print(f"[rank {rank}] done -> {os.environ.get('PPROBE_OUT', '(未启用探针)')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
