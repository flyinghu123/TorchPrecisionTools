# Megatron-LM Qwen3 demo: locate randomness that survives --deterministic-mode.
#
# Pretrains a tiny Qwen3-structure model (RMSNorm + RoPE + GQA + QK-LayerNorm +
# SwiGLU, ~1.3M params) with Megatron-LM on a single small GPU and uses TPD to
# compare repeated runs:
#
#   Experiment 1 (the question to answer): --deterministic-mode ON, same seed,
#     two runs -> any differing tensor = randomness that deterministic mode
#     did NOT eliminate.
#   Experiment 2 (control): --deterministic-mode OFF, same seed, two runs ->
#     shows whether this hardware/config is bit-reproducible even without it.
#   Experiment 3 (positive control): different seeds -> shows what TPD reports
#     when randomness IS present (data sampling order depends on --seed), i.e.
#     a sanity check that the comparison can detect differences at all.
#
# Requirements:
#   - conda env `tpd` (torch + `pip install -e torch-precision-debugger` +
#     `pip install pybind11`, plus g++ for the dataset helpers)
#   - Megatron-LM cloned at ../../Megatron-LM (or set MEGATRON_LM_PATH)
#   - ~4GB free GPU memory, ~2 minutes wall time
#
# Usage:
#   conda activate tpd
#   bash megatron_qwen3_demo.sh

set -euo pipefail

PYTHON=${TPD_PYTHON:-python}
# Run the TPD CLI with injection disabled: the .pth auto-import would also
# hook the CLI process itself and pollute its output.
tpd_cli() {
    TPD_ENABLED=0 "$PYTHON" -m tpd.cli "$@"
}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"$(cd "$(dirname "$0")/../../Megatron-LM" && pwd)"}
export MEGATRON_LM_PATH

# Megatron compiles its dataset helpers at runtime via `make`, which resolves
# python3/pybind11 from PATH: put the active env's bin directory first.
ENV_BIN="$(dirname "$(command -v "$PYTHON")")"
export PATH="$ENV_BIN:$PATH"

export TPD_ENABLED=1
export TPD_SAMPLE_COUNT=8
export TPD_MAX_STEPS=1000
export TPD_SAVE_INTERVAL=200

echo "=== Megatron-LM Qwen3 demo ==="
echo "python:        $($PYTHON --version) ($PYTHON)"
echo "Megatron-LM:   $MEGATRON_LM_PATH"

# Compile the C++ dataset helpers up front (no-op once built).
make -C "$MEGATRON_LM_PATH/megatron/core/datasets" >/dev/null

# ---------------------------------------------------------------------------
# Experiment 1: --deterministic-mode ON, seed 1234, run twice
# ---------------------------------------------------------------------------
echo
echo "=== Experiment 1: deterministic-mode ON, same seed, two runs ==="
export TPD_OUTPUT_DIR=saves/megatron_det_run1
$PYTHON megatron_qwen3_demo.py --seed 1234 2>&1 | grep -E "lm loss|exiting|TPD. Finalized" || true
export TPD_OUTPUT_DIR=saves/megatron_det_run2
$PYTHON megatron_qwen3_demo.py --seed 1234 2>&1 | grep -E "lm loss|exiting|TPD. Finalized" || true
tpd_cli compare saves/megatron_det_run1 saves/megatron_det_run2 \
    -o saves/megatron_det_comparison.json | tail -6

# ---------------------------------------------------------------------------
# Experiment 2 (control): deterministic-mode OFF, same seed, run twice
# ---------------------------------------------------------------------------
echo
echo "=== Experiment 2 (control): deterministic-mode OFF, same seed, two runs ==="
export TPD_OUTPUT_DIR=saves/megatron_nodet_run1
$PYTHON megatron_qwen3_demo.py --seed 1234 --no-deterministic 2>&1 | grep -E "lm loss|exiting|TPD. Finalized" || true
export TPD_OUTPUT_DIR=saves/megatron_nodet_run2
$PYTHON megatron_qwen3_demo.py --seed 1234 --no-deterministic 2>&1 | grep -E "lm loss|exiting|TPD. Finalized" || true
tpd_cli compare saves/megatron_nodet_run1 saves/megatron_nodet_run2 \
    -o saves/megatron_nodet_comparison.json | tail -6

# ---------------------------------------------------------------------------
# Experiment 3 (positive control): different seeds -> differences expected
# ---------------------------------------------------------------------------
echo
echo "=== Experiment 3 (positive control): seed 1234 vs 4321 (deterministic ON) ==="
export TPD_OUTPUT_DIR=saves/megatron_seed4321
$PYTHON megatron_qwen3_demo.py --seed 4321 2>&1 | grep -E "lm loss|exiting|TPD. Finalized" || true
tpd_cli compare saves/megatron_det_run1 saves/megatron_seed4321 \
    -o saves/megatron_seeddiff_comparison.json | tail -6
echo "--- first difference TPD locates (seed only changed) ---"
tpd_cli cmp first saves/megatron_seeddiff_comparison.json --type large-diff \
    --threshold 0.1 --window 0 | sed -n '1,14p'
echo "--- full report for the seed-diff experiment ---"
tpd_cli report saves/megatron_seeddiff_comparison.json \
    -o saves/megatron_seeddiff.report.txt | tail -3

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "=== Summary ==="
echo "Outputs written under examples/saves/:"
echo "  megatron_det_run{1,2}     experiment 1 raw TPD data (deterministic ON)"
echo "  megatron_nodet_run{1,2}   experiment 2 raw TPD data (deterministic OFF)"
echo "  megatron_seed4321         experiment 3 raw TPD data (seed 4321)"
echo "  megatron_*_comparison.json  compare results"
echo "  megatron_seeddiff.report.txt  full report for the seed-diff experiment"
echo
echo "Interpretation:"
echo "  - Experiment 1 with 0 differences: every hooked component (embedding,"
echo "    RMSNorm, GQA attention, SwiGLU MLP, DDP wrapper, loss, ...) is"
echo "    bit-identical across runs -> no residual randomness under"
echo "    --deterministic-mode for this config."
echo "  - Differences in experiment 1: those modules still hold randomness;"
echo "    inspect with 'tpd cmp first/list/show saves/megatron_det_comparison.json'"
echo "    and 'tpd stack saves/megatron_det_run1 <stack_id>'."
echo "  - Experiment 3 proves the comparison detects real differences (here:"
echo "    --seed also drives mock-data sampling order)."
