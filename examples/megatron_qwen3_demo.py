"""
Minimal Qwen3-structure demo that pretrains Megatron-LM on a single small GPU
(tested on a 4GB GTX 1050 Ti) with TPD enabled.

It drives Megatron-LM's official entry point ``pretrain_gpt.py`` so the full
Megatron training stack is exercised (arg validation, initialization, mcore
DistributedDataParallel, Adam optimizer, LR scheduler, dataloader, JIT fusion
warmup), while keeping the model tiny enough for 4GB of GPU memory.

Qwen3 architecture features covered (at tiny scale, ~1.3M params):
  - RMSNorm for pre-attention / pre-MLP / final norm (eps 1e-6)
  - QK LayerNorm (per-head RMSNorm on queries and keys)
  - Rotary position embeddings (base 1e6, full rotary percent)
  - Grouped-query attention (8 query heads, 4 KV groups)
  - SwiGLU MLP without bias
  - Tied input/output word embeddings

No tokenizer file or dataset is needed: NullTokenizer + --mock-data generate
everything locally, so the demo runs fully offline.

Usage (normally launched via megatron_qwen3_demo.sh, which also runs the
comparisons):

    TPD_ENABLED=1 TPD_OUTPUT_DIR=saves/run1 \
        python megatron_qwen3_demo.py [--seed 1234] [--no-deterministic]

Extra unknown args are forwarded to Megatron (e.g. --train-iters 8).
"""

import argparse
import os
import runpy
import sys

# Default: the Megatron-LM checkout living next to torch-precision-debugger.
DEFAULT_MEGATRON_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Megatron-LM")
)

# Tiny Qwen3-shaped model + the training hyper-parameters of the demo.
# Everything that is NOT a Qwen3 structure flag is chosen so that a 4GB Pascal
# GPU (no TE, no Triton, no flash-attn) can run it in fp32.
QWEN3_ARGS = [
    # ---- model: Qwen3 structure ----
    "--num-layers", "2",                     # Qwen3-0.6B has 28; 2 is enough here
    "--hidden-size", "256",
    "--ffn-hidden-size", "512",              # SwiGLU intermediate size
    "--num-attention-heads", "8",
    "--group-query-attention",
    "--num-query-groups", "4",               # GQA: 8 query heads, 4 KV groups
    "--normalization", "RMSNorm",
    "--norm-epsilon", "1e-6",
    "--qk-layernorm",                        # Qwen3-specific QK RMSNorm
    "--position-embedding-type", "rope",
    "--rotary-percent", "1.0",
    "--rotary-base", "1000000",              # Qwen3 rope theta
    "--no-rope-fusion",                      # rope fusion needs TransformerEngine
    "--swiglu",
    "--disable-bias-linear",                 # Qwen3 linears have no bias
    # ---- offline data ----
    "--tokenizer-type", "NullTokenizer",     # no tokenizer file needed
    "--vocab-size", "512",
    "--make-vocab-size-divisible-by", "1",
    "--mock-data",                           # no dataset files needed
    "--seq-length", "128",
    "--max-position-embeddings", "128",
    # ---- keep dropout at Qwen3's training default (0.0) ----
    "--attention-dropout", "0.0",
    "--hidden-dropout", "0.0",
    # ---- local (mcore-only) implementation: no TE / flash-attn required ----
    "--transformer-impl", "local",
    "--no-masked-softmax-fusion",
    "--no-bias-swiglu-fusion",
    "--no-bias-dropout-fusion",
    "--no-persist-layer-norm",               # not supported by torch RMSNorm
    "--no-gradient-accumulation-fusion",     # needs APEX CUDA extension
    # ---- tiny training loop ----
    "--micro-batch-size", "2",
    "--global-batch-size", "4",
    "--train-iters", "4",
    "--lr", "1e-4",
    "--lr-decay-style", "constant",
    "--optimizer", "adam",
    "--weight-decay", "0.01",
    "--clip-grad", "1.0",
    "--eval-interval", "1000",               # skip validation runs
    "--num-workers", "0",
    "--log-interval", "1",
    "--exit-interval", "4",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiny Qwen3-structure Megatron-LM pretraining demo (TPD-friendly)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Megatron --seed value")
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="drop Megatron --deterministic-mode (control experiment)",
    )
    return parser


def main():
    demo_args, megatron_extra = build_parser().parse_known_args()

    megatron_root = os.environ.get("MEGATRON_LM_PATH", DEFAULT_MEGATRON_ROOT)
    if not os.path.isfile(os.path.join(megatron_root, "pretrain_gpt.py")):
        print(
            f"[demo] pretrain_gpt.py not found under {megatron_root}. "
            "Clone Megatron-LM there or set MEGATRON_LM_PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Single-process launch: no torchrun needed, set the env vars Megatron reads.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # Megatron's JIT-fusion warmup compiles with torch.compile -> Triton, which
    # rejects pre-Volta GPUs (sm_61 here). Disabling dynamo keeps the fused
    # helpers in eager mode; RNG consumption stays identical across runs, so it
    # does not affect the determinism comparison.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    megatron_args = list(QWEN3_ARGS) + ["--seed", str(demo_args.seed)]
    if not demo_args.no_deterministic:
        megatron_args.append("--deterministic-mode")

    # Run Megatron's official entry point as __main__.
    entry = os.path.join(megatron_root, "pretrain_gpt.py")
    sys.path.insert(0, megatron_root)  # for gpt_builders / model_provider
    sys.argv = [entry] + megatron_args + megatron_extra
    print(f"[demo] launching Megatron: {entry}")
    print(f"[demo] args: {' '.join(megatron_args)}")
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
