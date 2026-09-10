# Single GPU demo: run an fp32 baseline and an AMP (fp16 autocast) run,
# then compare them with the TPD CLI.
# Usage: bash gpu_demo.sh

export TPD_ENABLED=1
export TPD_SAMPLE_COUNT=3
export TPD_MAX_STEPS=64

# Run 1: fp32 baseline
export TPD_OUTPUT_DIR=saves/gpu_result0
python gpu_demo.py

# Run 2: fp16 autocast (AMP). Both runs use seed 42, so any difference
# reported by `tpd compare` comes from precision, not from different data
export TPD_OUTPUT_DIR=saves/gpu_result1
python gpu_demo.py --amp

# Compare the two runs
tpd compare saves/gpu_result0 saves/gpu_result1 -o saves/gpu_comparison.json
