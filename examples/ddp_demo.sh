# Multi-GPU DDP demo launched with torchrun.
# INJECT_NAN_RANK=1 needs at least 2 GPUs.
# Usage: NPROC=2 bash ddp_demo.sh

NPROC=${NPROC:-2}

export TPD_ENABLED=1
export TPD_SAMPLE_COUNT=3
export TPD_MAX_STEPS=64

# Run 1: healthy baseline, each rank writes rank{N}.jsonl / stacks_rank{N}.json
export TPD_OUTPUT_DIR=saves/ddp_result0
torchrun --nproc_per_node=$NPROC ddp_demo.py

# Run 2: inject NaN on rank 1 to simulate a precision issue
export TPD_OUTPUT_DIR=saves/ddp_result1
INJECT_NAN_RANK=1 torchrun --nproc_per_node=$NPROC ddp_demo.py

# Compare rank 1: the forward_input records show nan_count diffs on rank 1
# (gradients are all-reduced, so other ranks show diffs as well)
tpd compare saves/ddp_result0 saves/ddp_result1 -r 1 -o saves/ddp_comparison_rank1.json
